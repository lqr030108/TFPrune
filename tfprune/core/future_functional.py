"""Next-block routing and projected-value signatures for functional selection."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def _attention_geometry(self_attn, projected_width: int):
    config = getattr(self_attn, "config", None)
    num_heads = getattr(self_attn, "num_heads", None)
    if num_heads is None and config is not None:
        num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(self_attn, "num_key_value_heads", None)
    if num_kv_heads is None and config is not None:
        num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_heads is None:
        raise AttributeError("Cannot determine num_attention_heads")
    num_heads = int(num_heads)
    num_kv_heads = int(num_kv_heads if num_kv_heads is not None else num_heads)
    head_dim = getattr(self_attn, "head_dim", None)
    if head_dim is None and config is not None:
        head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = projected_width // num_kv_heads
    head_dim = int(head_dim)
    if projected_width != num_kv_heads * head_dim:
        raise RuntimeError(
            f"Unexpected K/V projection width {projected_width}; expected {num_kv_heads} * {head_dim}"
        )
    if num_heads % num_kv_heads:
        raise RuntimeError(
            f"num_attention_heads={num_heads} is not divisible by num_key_value_heads={num_kv_heads}"
        )
    return (num_heads, num_kv_heads, head_dim)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _broadcast_rope(tensor: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    while embedding.ndim > 0 and embedding.shape[0] == 1 and (embedding.ndim > 3):
        embedding = embedding.squeeze(0)
    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(0).unsqueeze(1)
    elif embedding.ndim == 3:
        embedding = embedding.unsqueeze(1)
    elif embedding.ndim != 4:
        raise RuntimeError(f"Unsupported rotary embedding shape: {embedding.shape}")
    return embedding.to(device=tensor.device, dtype=tensor.dtype)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    self_attn=None,
):
    if position_embeddings is None:
        return (query, key)
    cosine, sine = position_embeddings
    if cosine.ndim == 4 and cosine.shape[0] == 3:
        rope_scaling = getattr(self_attn, "rope_scaling", None) or {}
        section = rope_scaling.get("mrope_section")
        if section is None:
            config = getattr(self_attn, "config", None)
            scaling = getattr(config, "rope_scaling", None) or {}
            section = scaling.get("mrope_section")
        if section is None:
            raise RuntimeError("Qwen multimodal RoPE is missing mrope_section")
        split_sizes = [int(value) for value in section] * 2
        cosine_parts = cosine.split(split_sizes, dim=-1)
        sine_parts = sine.split(split_sizes, dim=-1)
        cosine = torch.cat(
            [part[index % 3] for index, part in enumerate(cosine_parts)], dim=-1
        ).unsqueeze(1)
        sine = torch.cat(
            [part[index % 3] for index, part in enumerate(sine_parts)], dim=-1
        ).unsqueeze(1)
        cosine = cosine.to(device=query.device, dtype=query.dtype)
        sine = sine.to(device=query.device, dtype=query.dtype)
        return (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )
    cosine = _broadcast_rope(query, cosine)
    sine = _broadcast_rope(query, sine)
    return (query * cosine + _rotate_half(query) * sine, key * cosine + _rotate_half(key) * sine)


def _select_position_embeddings(
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]], indices: torch.Tensor
):
    if position_embeddings is None:
        return None
    cosine, sine = position_embeddings
    return (
        cosine.index_select(-2, indices.to(cosine.device)),
        sine.index_select(-2, indices.to(sine.device)),
    )


def _balanced_block(block: torch.Tensor, eps: float) -> torch.Tensor:
    flat = block.reshape(block.shape[0], -1).float()
    rms = flat.square().mean().sqrt().clamp_min(eps)
    return flat / (rms * math.sqrt(max(flat.shape[1], 1)))


class FutureFunctionalResponseBuilder:
    def __init__(self, seed: int = 17, eps: float = 1e-08):
        self.seed = int(seed)
        self.eps = float(eps)
        self._projection_cache = {}
        self._sketch_cache = {}

    def _projection(self, input_dim: int, output_dim: int, device) -> torch.Tensor:
        key = (int(input_dim), int(output_dim), str(device))
        matrix = self._projection_cache.get(key)
        if matrix is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + 1009 * input_dim + output_dim)
            matrix = torch.randn(
                input_dim, output_dim, generator=generator, dtype=torch.float32
            ) / math.sqrt(output_dim)
            matrix = matrix.to(device)
            self._projection_cache[key] = matrix
        return matrix

    def sketch(self, signatures: torch.Tensor, output_dim: Optional[int]) -> torch.Tensor:
        if output_dim is None or int(output_dim) <= 0 or int(output_dim) >= signatures.shape[-1]:
            return signatures
        input_dim = int(signatures.shape[-1])
        output_dim = int(output_dim)
        key = (input_dim, output_dim, str(signatures.device))
        matrix = self._sketch_cache.get(key)
        if matrix is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + 7919 * input_dim + 104729 * output_dim)
            matrix = torch.empty(input_dim, output_dim, dtype=torch.float32)
            matrix.bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
            matrix.div_(math.sqrt(output_dim))
            matrix = matrix.to(signatures.device)
            self._sketch_cache[key] = matrix
        return signatures.float() @ matrix

    @torch.no_grad()
    def build(
        self,
        decoder_layer,
        hidden_states: torch.Tensor,
        vision_start: int,
        vision_end: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        response_dim: int,
        max_text_probes: int,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError("future functional response requires hidden_states [1, S, D]")
        if not 0 <= vision_start < vision_end <= hidden_states.shape[1]:
            raise ValueError("invalid visual-token range")
        self_attn = decoder_layer.self_attn
        seq_len = hidden_states.shape[1]
        probe_indices = torch.arange(
            vision_end, seq_len, device=hidden_states.device, dtype=torch.long
        )
        if probe_indices.numel() == 0:
            probe_indices = torch.arange(
                0, vision_start, device=hidden_states.device, dtype=torch.long
            )
        if probe_indices.numel() == 0:
            raise RuntimeError("No text token is available as a functional query probe")
        probe_indices = probe_indices[-max(1, int(max_text_probes)) :]
        normalized = decoder_layer.input_layernorm(hidden_states)
        probe_states = normalized.index_select(1, probe_indices)
        visual_states = normalized[:, vision_start:vision_end]
        query = self_attn.q_proj(probe_states)
        key = self_attn.k_proj(normalized)
        value = self_attn.v_proj(visual_states)
        num_heads, num_kv_heads, head_dim = _attention_geometry(self_attn, value.shape[-1])
        query = query.view(1, probe_indices.numel(), num_heads, head_dim).transpose(1, 2)
        key = key.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        value = value.view(1, vision_end - vision_start, num_kv_heads, head_dim).transpose(1, 2)
        query_positions = _select_position_embeddings(position_embeddings, probe_indices)
        query, _ = _apply_rope(query, query, query_positions, self_attn=self_attn)
        _, key = _apply_rope(key, key, position_embeddings, self_attn=self_attn)
        if num_kv_heads != num_heads:
            repeats = num_heads // num_kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        probe_query = query[0].float()
        all_key = key[0].float()
        if probe_query.shape[1] != probe_indices.numel() or all_key.shape[1] != seq_len:
            raise RuntimeError(
                f"Future response Q/K shape mismatch: query={tuple(probe_query.shape)}, key={tuple(all_key.shape)}, probes={probe_indices.numel()}, seq_len={seq_len}"
            )
        logits = torch.einsum("hqd,hkd->hqk", probe_query, all_key)
        logits = logits / math.sqrt(head_dim)
        mask_applied = False
        if attention_mask is not None and attention_mask.ndim == 4:
            mask = attention_mask[0]
            if mask.shape[-2] == seq_len:
                mask = mask[:, probe_indices, :]
            elif mask.shape[-2] != probe_indices.numel():
                mask = None
            if mask is not None:
                logits = logits + mask.to(logits.device, torch.float32)
                mask_applied = True
        if not mask_applied:
            key_positions = torch.arange(seq_len, device=logits.device)
            blocked = key_positions.unsqueeze(0) > probe_indices.unsqueeze(1)
            logits = logits.masked_fill(blocked.unsqueeze(0), float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        visual_probability = probabilities[:, :, vision_start:vision_end]
        projection = self._projection(normalized.shape[-1], int(response_dim), hidden_states.device)
        output_weight = self_attn.o_proj.weight.float()
        head_weight = output_weight.T.reshape(num_heads, head_dim, output_weight.shape[0])
        head_maps = torch.matmul(head_weight, projection)
        projected_value = torch.einsum("hsd,hdr->hsr", value[0].float(), head_maps)
        visual_value = projected_value
        contribution = torch.einsum("hqn,hnr->qnr", visual_probability, visual_value).permute(
            1, 0, 2
        )
        routing = visual_probability.permute(2, 1, 0)
        blocks = []
        blocks.append(_balanced_block(contribution, self.eps))
        blocks.append(_balanced_block(routing, self.eps))
        value_potential = visual_value.sum(dim=0)
        blocks.append(_balanced_block(value_potential, self.eps))
        return torch.cat(blocks, dim=-1)
