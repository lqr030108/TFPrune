"""Qwen2.5-VL decoder compression with native multimodal rotary positions."""

import types

import torch
from loguru import logger as eval_logger
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..core.cache_utils import cache_sequence_length


def _is_prefill(past_key_values) -> bool:
    return cache_sequence_length(past_key_values) == 0


def _rebuild_decode_mask(attention_mask, batch_size, seq_length, past_length, tfprune_core):
    prompt_mask = tfprune_core.pruned_prompt_attention_mask
    if prompt_mask is None:
        return attention_mask
    prompt_mask = prompt_mask.to(attention_mask.device)
    generated_past = max(0, past_length - prompt_mask.shape[1])
    generated_mask = torch.ones(
        batch_size, generated_past, dtype=attention_mask.dtype, device=attention_mask.device
    )
    return torch.cat([prompt_mask, generated_mask, attention_mask[:, -seq_length:]], dim=1)


def _ensure_mass_causal_mask(mask, hidden_states, query_length, past_length, tfprune_core):
    if tfprune_core.pruned_prompt_attention_bias is None:
        return mask
    if mask is None:
        key_length = past_length + query_length
        mask = torch.zeros(
            1, 1, query_length, key_length, dtype=hidden_states.dtype, device=hidden_states.device
        )
        if query_length > 1:
            query_positions = torch.arange(query_length, device=hidden_states.device) + past_length
            key_positions = torch.arange(key_length, device=hidden_states.device)
            blocked = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            mask.masked_fill_(blocked.view(1, 1, query_length, key_length), float("-inf"))
    return tfprune_core.apply_attention_mass_bias(mask)


def _patch_qwen_text_forward(text_model, tfprune_core):

    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        requested_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length = inputs_embeds.shape[:2]
        if batch_size != 1:
            raise NotImplementedError("TFPrune Qwen runtime currently requires batch_size=1")
        prefill = _is_prefill(past_key_values)
        if use_cache and past_key_values is None and (not torch.jit.is_tracing()):
            past_key_values = DynamicCache()
        past_seen_tokens = cache_sequence_length(past_key_values)
        if not prefill and attention_mask is not None and (attention_mask.ndim == 2):
            attention_mask = _rebuild_decode_mask(
                attention_mask, batch_size, seq_length, past_seen_tokens, tfprune_core
            )
        attention_mask_2d = attention_mask
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, batch_size, -1)
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, requested_attentions
        )
        hidden_states = inputs_embeds
        if not prefill:
            causal_mask = _ensure_mass_causal_mask(
                causal_mask, hidden_states, seq_length, past_seen_tokens, tfprune_core
            )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if requested_attentions else None
        next_decoder_cache = None
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_output_attentions = bool(requested_attentions)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=layer_output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if layer_output_attentions else 1]
            if requested_attentions:
                all_self_attns += (layer_outputs[1],)
            if prefill and tfprune_core.should_prune(completed_layer_idx=layer_idx):
                hidden_states, next_decoder_cache, _ = tfprune_core.execute_pruning(
                    hidden_states,
                    past_key_values=next_decoder_cache,
                    future_decoder_layer=self.layers[layer_idx + 1]
                    if layer_idx + 1 < len(self.layers)
                    else None,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                )
                past_key_values = next_decoder_cache
                keep_mask = tfprune_core.last_sequence_keep_mask
                position_ids = position_ids[..., keep_mask]
                cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
                if causal_mask is not None:
                    if causal_mask.ndim != 4:
                        raise RuntimeError(
                            "Qwen pruning requires a 4D additive mask when a mask is present"
                        )
                    causal_mask = causal_mask[:, :, keep_mask, :][:, :, :, keep_mask]
                causal_mask = _ensure_mass_causal_mask(
                    causal_mask, hidden_states, hidden_states.shape[1], 0, tfprune_core
                )
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                if attention_mask_2d is None:
                    tfprune_core.pruned_prompt_attention_mask = torch.ones(
                        batch_size,
                        hidden_states.shape[1],
                        dtype=torch.long,
                        device=hidden_states.device,
                    )
                else:
                    tfprune_core.pruned_prompt_attention_mask = attention_mask_2d[
                        :, keep_mask
                    ].detach()
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                (
                    value
                    for value in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                    if value is not None
                )
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    text_model.forward = types.MethodType(patched_forward, text_model)


def _patch_qwen_outer_forward(qwen_vl_model, tfprune_core):
    original_forward = qwen_vl_model.forward

    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if _is_prefill(past_key_values) and input_ids is not None:
            if input_ids.shape[0] != 1:
                raise NotImplementedError("TFPrune Qwen runtime currently requires batch_size=1")
            tfprune_core.reset_state()
            visual_mask = input_ids[0].eq(self.config.image_token_id)
            visual_positions = torch.where(visual_mask)[0]
            if visual_positions.numel() == 0:
                return original_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    **kwargs,
                )
            if visual_positions.numel() > 1 and (
                not torch.all(visual_positions[1:] == visual_positions[:-1] + 1)
            ):
                raise NotImplementedError(
                    "Interleaved/multi-span Qwen images are not supported yet"
                )
            tfprune_core.set_vision_token_range(
                int(visual_positions[0].item()), int(visual_positions[-1].item()) + 1
            )
        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    qwen_vl_model.forward = types.MethodType(patched_forward, qwen_vl_model)
    qwen_vl_model._tfprune_original_forward = original_forward


def patch_qwen_attention_layers(qwen_model, tfprune_core):
    qwen_vl_model = qwen_model.model
    text_model = qwen_vl_model.language_model
    _patch_qwen_text_forward(text_model, tfprune_core)
    _patch_qwen_outer_forward(qwen_vl_model, tfprune_core)
    eval_logger.info("[TFPrune-Qwen] Patched native outer model and text decoder at layer {}", 4)
