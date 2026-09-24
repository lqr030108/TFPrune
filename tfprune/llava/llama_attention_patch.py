# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
"""Insert functional compression after the fourth LLaMA decoder block."""

import types

import torch
from loguru import logger as eval_logger

try:
    from transformers.cache_utils import Cache, DynamicCache

    _HAS_TRANSFORMERS_CACHE = True
except ImportError:
    Cache = DynamicCache = None
    _HAS_TRANSFORMERS_CACHE = False
try:
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

    _HAS_OFFICIAL_MASK_UTIL = True
except ImportError:
    _HAS_OFFICIAL_MASK_UTIL = False


def _is_kv_empty(past_key_values):
    if past_key_values is None:
        return True
    if hasattr(past_key_values, "__len__"):
        return len(past_key_values) == 0
    return False


def _is_cache_object(past_key_values):
    if past_key_values is None:
        return False
    return not isinstance(past_key_values, tuple)


def _decoder_uses_cache_class(llama_model):
    if not _HAS_TRANSFORMERS_CACHE or not llama_model.layers:
        return False
    self_attn = getattr(llama_model.layers[0], "self_attn", None)
    return self_attn is not None and hasattr(self_attn, "layer_idx")


def _get_past_seq_len(past_key_values):
    if _is_kv_empty(past_key_values):
        return 0
    if isinstance(past_key_values, tuple):
        return past_key_values[0][0].shape[2]
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    if hasattr(past_key_values, "key_cache"):
        return past_key_values.key_cache[0].shape[2]
    return 0


def _get_layer_kv(past_key_values, layer_idx):
    if _is_kv_empty(past_key_values):
        return None
    return past_key_values[layer_idx]


def _compute_rope_embeddings(model, hidden_states, position_ids):
    if not hasattr(model, "rotary_emb") or model.rotary_emb is None:
        return None
    try:
        cos, sin = model.rotary_emb(hidden_states, position_ids)
    except TypeError:
        seq_len = hidden_states.shape[1]
        cos, sin = model.rotary_emb(seq_len, hidden_states.device, hidden_states.dtype)
    return (cos, sin)


def _make_4d_causal_attention_mask(
    attention_mask_2d, input_shape, inputs_embeds, past_key_values_length=0
):
    batch_size, target_len = input_shape
    total_key_len = target_len + past_key_values_length
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    if attention_mask_2d is not None and attention_mask_2d.dim() == 4:
        return attention_mask_2d
    causal_mask = torch.full((target_len, total_key_len), float("-inf"), dtype=dtype, device=device)
    causal_mask = torch.triu(causal_mask, diagonal=past_key_values_length + 1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
    if attention_mask_2d is not None:
        key_pad_mask = torch.ones((batch_size, total_key_len), dtype=torch.bool, device=device)
        if past_key_values_length > 0:
            key_pad_mask[:, :past_key_values_length] = True
        key_pad_mask[:, past_key_values_length:] = attention_mask_2d.bool()
        pad_mask = torch.zeros((batch_size, 1, 1, total_key_len), dtype=dtype, device=device)
        pad_mask = pad_mask.masked_fill(~key_pad_mask, float("-inf"))
        expanded_mask = causal_mask + pad_mask
    else:
        expanded_mask = causal_mask.expand(batch_size, 1, target_len, total_key_len)
    if torch.isnan(expanded_mask).any():
        eval_logger.error("[TFPrune] NaN detected in attention mask! Falling back to causal only.")
        expanded_mask = causal_mask.expand(batch_size, 1, target_len, total_key_len)
    return expanded_mask


def patch_llama_attention_layers(model, tfprune_core):
    llama_model = model.get_model()
    _patch_llama_model_forward(llama_model, tfprune_core)
    eval_logger.info("[TFPrune] LLaMA decoder patched for compression after block 4.")


def _patch_llama_model_forward(llama_model, tfprune_core):

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
        **kwargs,
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
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return_legacy_cache = False
        if use_cache and _decoder_uses_cache_class(self):
            return_legacy_cache = not isinstance(past_key_values, Cache)
            if return_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        is_cache_obj = _is_cache_object(past_key_values)
        past_seq_len = _get_past_seq_len(past_key_values)
        if position_ids is None:
            position_ids = torch.arange(
                past_seq_len,
                seq_length + past_seq_len,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        attention_mask_2d = attention_mask
        is_prefill = _is_kv_empty(past_key_values)
        if (
            not is_prefill
            and attention_mask is not None
            and (attention_mask.dim() == 2)
            and (tfprune_core.pruned_prompt_attention_mask is not None)
        ):
            prompt_mask = tfprune_core.pruned_prompt_attention_mask.to(attention_mask.device)
            generated_past = max(0, past_seq_len - prompt_mask.shape[1])
            generated_mask = torch.ones(
                batch_size, generated_past, dtype=attention_mask.dtype, device=attention_mask.device
            )
            current_mask = attention_mask[:, -seq_length:]
            attention_mask = torch.cat([prompt_mask, generated_mask, current_mask], dim=1)
            attention_mask_2d = attention_mask
        input_shape = (batch_size, seq_length)
        if _HAS_OFFICIAL_MASK_UTIL:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length=past_seq_len
            )
        else:
            attention_mask = _make_4d_causal_attention_mask(
                attention_mask,
                input_shape=input_shape,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_seq_len,
            )
        if not is_prefill:
            attention_mask = tfprune_core.apply_attention_mass_bias(attention_mask)
        position_embeddings = _compute_rope_embeddings(self, inputs_embeds, position_ids)
        if use_cache:
            if is_cache_obj:
                next_decoder_cache = past_key_values
            else:
                next_decoder_cache = ()
        else:
            next_decoder_cache = None
        all_self_attentions = () if requested_attentions else None
        all_hidden_states = () if output_hidden_states else None
        hidden_states = inputs_embeds
        num_layers = len(self.layers)
        if 4 >= num_layers and is_prefill:
            raise ValueError("TFPrune requires at least five decoder blocks")
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = (
                _get_layer_kv(past_key_values, idx) if not is_cache_obj else past_key_values
            )
            layer_output_attn = bool(requested_attentions)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=layer_output_attn,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if use_cache and (not is_cache_obj):
                cache_idx = 2 if layer_output_attn else 1
                if cache_idx < len(layer_outputs):
                    present_kv = layer_outputs[cache_idx]
                    next_decoder_cache += (present_kv,)
            if requested_attentions:
                all_self_attentions += (layer_outputs[1],)
            if (
                is_prefill
                and (not tfprune_core.is_pruned)
                and tfprune_core.should_prune(completed_layer_idx=idx)
            ):
                hidden_states, pruned_cache, keep_indices = tfprune_core.execute_pruning(
                    hidden_states,
                    past_key_values=next_decoder_cache,
                    future_decoder_layer=self.layers[idx + 1]
                    if idx + 1 < len(self.layers)
                    else None,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                )
                next_decoder_cache = pruned_cache
                vision_start = tfprune_core.vision_token_start
                vision_end = tfprune_core.vision_token_end
                new_seq_len = hidden_states.shape[1]
                keep_mask = torch.ones(seq_length, dtype=torch.bool, device=hidden_states.device)
                vision_len = vision_end - vision_start
                vision_mask = torch.zeros(vision_len, dtype=torch.bool, device=hidden_states.device)
                vision_mask[keep_indices] = True
                keep_mask[vision_start:vision_end] = vision_mask
                if position_ids is not None:
                    position_ids = position_ids[:, keep_mask]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :, keep_mask, :][:, :, :, keep_mask]
                    attention_mask = tfprune_core.apply_attention_mass_bias(attention_mask)
                if attention_mask_2d is not None and attention_mask_2d.dim() == 2:
                    tfprune_core.pruned_prompt_attention_mask = attention_mask_2d[
                        :, keep_mask
                    ].detach()
                else:
                    tfprune_core.pruned_prompt_attention_mask = torch.ones(
                        hidden_states.shape[0],
                        new_seq_len,
                        dtype=torch.long,
                        device=hidden_states.device,
                    )
                position_embeddings = _compute_rope_embeddings(self, hidden_states, position_ids)
                seq_length = new_seq_len
        hidden_states = self.norm(hidden_states)
        output_cache = next_decoder_cache
        if use_cache and return_legacy_cache and (output_cache is not None):
            output_cache = output_cache.to_legacy_cache()
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            outputs = (hidden_states,)
            if use_cache:
                outputs += (output_cache,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if requested_attentions:
                outputs += (all_self_attentions,)
            return outputs
        from transformers.modeling_outputs import BaseModelOutputWithPast

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=output_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions if requested_attentions else None,
        )

    llama_model.forward = types.MethodType(patched_forward, llama_model)
