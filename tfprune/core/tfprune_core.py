"""Functional selection, evidence transport and visual-token compaction."""

from typing import Optional

import torch
from loguru import logger as eval_logger

from .cache_utils import merge_prune_kv_cache
from .functional_coverage import FunctionalCoverageSelector
from .future_functional import FutureFunctionalResponseBuilder


class TFPruneCore:
    def __init__(self, *, target_vision_tokens=None, target_retention_ratio=None, device="cpu"):
        if (target_vision_tokens is None) == (target_retention_ratio is None):
            raise ValueError("Specify exactly one token budget or retention ratio")
        if target_vision_tokens is not None and (
            int(target_vision_tokens) != target_vision_tokens or target_vision_tokens < 1
        ):
            raise ValueError("target_vision_tokens must be a positive integer")
        if target_retention_ratio is not None and (not 0 < target_retention_ratio <= 1):
            raise ValueError("target_retention_ratio must be in (0, 1]")
        self.device = torch.device(device)
        self.target_vision_tokens = target_vision_tokens
        self.target_retention_ratio = target_retention_ratio
        self.selector = FunctionalCoverageSelector()
        self.future_response_builder = FutureFunctionalResponseBuilder()
        self.reset_state()

    def set_vision_token_range(self, start: int, end: int):
        if not 0 <= start < end:
            raise ValueError(f"invalid visual token range [{start}, {end})")
        self.vision_token_start = int(start)
        self.vision_token_end = int(end)

    def set_mandatory_vision_indices(self, indices):
        if indices is None:
            self.mandatory_vision_indices = None
            return
        self.mandatory_vision_indices = [int(index) for index in indices]

    def apply_attention_mass_bias(self, attention_mask: Optional[torch.Tensor]):
        bias = self.pruned_prompt_attention_bias
        if attention_mask is None or bias is None or attention_mask.ndim != 4:
            return attention_mask
        key_length = attention_mask.shape[-1]
        prefix = min(key_length, bias.numel())
        additive = torch.zeros(key_length, device=attention_mask.device, dtype=attention_mask.dtype)
        additive[:prefix] = bias[:prefix].to(attention_mask.device, attention_mask.dtype)
        return attention_mask + additive.view(1, 1, 1, -1)

    def should_prune(self, completed_layer_idx: int) -> bool:
        if self.is_pruned or self.vision_token_start is None:
            return False
        idx = int(completed_layer_idx)
        return idx + 1 >= 4

    def _finish_noop_pruning(
        self, hidden_states: torch.Tensor, past_key_values, num_visual: int, requested_tokens: int
    ):
        eval_logger.info(
            "[TFPrune] full-budget no-op: requested={}, native={}; skipping functional selection",
            requested_tokens,
            num_visual,
        )
        keep_indices = list(range(num_visual))
        sequence_keep = torch.ones(
            hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device
        )
        self.is_pruned = True
        self.keep_vision_indices = keep_indices
        self.last_sequence_keep_mask = sequence_keep
        self.num_removed_tokens = 0
        self.last_selection = None
        self.pruned_prompt_attention_bias = torch.zeros(
            hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device
        )
        return (hidden_states, past_key_values, keep_indices)

    def execute_pruning(
        self,
        hidden_states: torch.Tensor,
        past_key_values=None,
        future_decoder_layer=None,
        position_embeddings=None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if self.is_pruned:
            return (hidden_states, past_key_values, self.keep_vision_indices)
        if hidden_states.shape[0] != 1:
            raise NotImplementedError("mid-decoder TFPrune currently supports batch_size=1")
        start, end = (self.vision_token_start, self.vision_token_end)
        visual_states = hidden_states[0, start:end]
        num_visual = visual_states.shape[0]
        mandatory_mask = torch.zeros(num_visual, dtype=torch.bool, device=visual_states.device)
        if self.mandatory_vision_indices:
            mandatory = torch.as_tensor(
                self.mandatory_vision_indices, device=visual_states.device, dtype=torch.long
            )
            mandatory = mandatory[(mandatory >= 0) & (mandatory < num_visual)]
            mandatory_mask[mandatory] = True
        candidate_indices = torch.where(~mandatory_mask)[0]
        candidate_count = int(candidate_indices.numel())
        if self.target_vision_tokens is not None:
            candidate_target = int(self.target_vision_tokens)
        else:
            candidate_target = max(1, int(round(self.target_retention_ratio * candidate_count)))
        # Structural tokens are preserved in addition to the content-token budget.
        if candidate_target >= candidate_count:
            return self._finish_noop_pruning(
                hidden_states, past_key_values, num_visual, candidate_target
            )
        if future_decoder_layer is None:
            raise RuntimeError(
                "future functional response requires the decoder layer after the pruning boundary; the model must contain at least five decoder blocks"
            )
        functional_signatures = self.future_response_builder.build(
            future_decoder_layer,
            hidden_states,
            start,
            end,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            response_dim=64,
            max_text_probes=16,
        )
        candidate_functional_signatures = functional_signatures.index_select(0, candidate_indices)
        metric_signatures = self.future_response_builder.sketch(
            candidate_functional_signatures, 128
        )
        result = self.selector.select_one(
            candidate_functional_signatures,
            visual_states.index_select(0, candidate_indices),
            target_tokens=candidate_target,
            metric_signatures=metric_signatures,
        )
        selected_candidates = candidate_indices.index_select(
            0, torch.as_tensor(result.indices, device=candidate_indices.device)
        )
        keep_indices = (
            torch.cat([selected_candidates, torch.where(mandatory_mask)[0]], dim=0)
            .unique(sorted=True)
            .tolist()
        )
        sequence_keep = torch.ones(
            hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device
        )
        vision_keep = torch.zeros(end - start, dtype=torch.bool, device=hidden_states.device)
        vision_keep[torch.as_tensor(keep_indices, device=hidden_states.device)] = True
        sequence_keep[start:end] = vision_keep
        selected_candidates_ordered = candidate_indices.index_select(
            0, torch.as_tensor(result.indices, device=candidate_indices.device)
        )
        mandatory_indices = torch.where(mandatory_mask)[0]
        cluster_representatives = torch.cat([selected_candidates_ordered, mandatory_indices], dim=0)
        cluster_assignments = torch.empty(num_visual, dtype=torch.long, device=visual_states.device)
        cluster_assignments[candidate_indices] = result.assignments
        for offset, mandatory_idx in enumerate(mandatory_indices):
            cluster_assignments[mandatory_idx] = len(result.indices) + offset
        cluster_member_weights = torch.ones(
            num_visual, dtype=torch.float32, device=visual_states.device
        )
        cluster_member_weights[candidate_indices] = result.transport_weights
        candidate_masses = result.effective_masses
        cluster_masses = torch.cat(
            [
                candidate_masses.to(visual_states.device, dtype=torch.float32),
                torch.ones(
                    mandatory_indices.numel(), dtype=torch.float32, device=visual_states.device
                ),
            ]
        )
        merged_visual = visual_states.clone()
        merged_visual[selected_candidates_ordered] = result.merged_states.to(visual_states.dtype)
        pruned_hidden = hidden_states.clone()
        pruned_hidden[0, start:end] = merged_visual
        pruned_hidden = pruned_hidden[:, sequence_keep, :]
        pruned_cache = past_key_values
        if past_key_values is not None:
            pruned_cache = merge_prune_kv_cache(
                past_key_values,
                sequence_keep,
                start,
                cluster_representatives,
                cluster_assignments,
                member_weights=cluster_member_weights,
            )
        prompt_bias = torch.zeros(
            int(sequence_keep.sum().item()), dtype=torch.float32, device=hidden_states.device
        )
        if cluster_masses is not None:
            visual_bias = torch.zeros(num_visual, dtype=torch.float32, device=hidden_states.device)
            visual_bias[cluster_representatives] = cluster_masses.float().clamp_min(1e-08).log()
            prompt_bias[start : start + len(keep_indices)] = visual_bias[vision_keep]
        self.pruned_prompt_attention_bias = prompt_bias
        self.is_pruned = True
        self.keep_vision_indices = keep_indices
        self.last_sequence_keep_mask = sequence_keep
        self.num_removed_tokens = int(sequence_keep.numel() - sequence_keep.sum().item())
        self.last_selection = result
        return (pruned_hidden, pruned_cache, keep_indices)

    def reset_state(self):
        self.is_pruned = False
        self.keep_vision_indices = None
        self.vision_token_start = None
        self.vision_token_end = None
        self.mandatory_vision_indices = None
        self.last_selection = None
        self.pruned_prompt_attention_mask = None
        self.pruned_prompt_attention_bias = None
        self.last_sequence_keep_mask = None
        self.num_removed_tokens = 0
