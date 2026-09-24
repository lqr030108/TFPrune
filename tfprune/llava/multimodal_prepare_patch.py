"""Locate visual spans while preserving native LLaVA preprocessing."""

import types

import torch
from loguru import logger as eval_logger

try:
    from llava.constants import IMAGE_TOKEN_INDEX
except ImportError:
    IMAGE_TOKEN_INDEX = -200


def patch_native_multimodal_prepare(model, tfprune_core):
    original_prepare = model.prepare_inputs_labels_for_multimodal

    def patched_prepare(self, *args, **kwargs):
        outputs = original_prepare(*args, **kwargs)
        input_ids = args[0] if len(args) > 0 else kwargs.get("input_ids")
        attention_mask = args[2] if len(args) > 2 else kwargs.get("attention_mask")
        images = args[5] if len(args) > 5 else kwargs.get("images")
        inputs_embeds = outputs[4]
        if (
            images is None
            or inputs_embeds is None
            or input_ids is None
            or (input_ids.shape[1] == 1)
        ):
            return outputs
        if input_ids.shape[0] != 1:
            raise NotImplementedError("TFPrune LLaVA runtime currently requires batch_size=1")
        tfprune_core.reset_state()
        source_mask = (
            attention_mask[0].bool()
            if attention_mask is not None
            else torch.ones_like(input_ids[0], dtype=torch.bool)
        )
        valid_ids = input_ids[0][source_mask]
        image_positions = torch.where(valid_ids == IMAGE_TOKEN_INDEX)[0]
        if image_positions.numel() != 1:
            raise NotImplementedError(
                f"TFPrune currently supports one contiguous image span per LLaVA sample; found {image_positions.numel()} image placeholders"
            )
        output_attention_mask = outputs[2]
        if output_attention_mask is None:
            output_valid_len = int(inputs_embeds.shape[1])
            output_left_pad = 0
        else:
            output_mask = output_attention_mask[0].bool()
            output_valid_len = int(output_mask.sum().item())
            output_left_pad = int((~output_mask).cumprod(dim=0).sum().item())
        text_tokens_after_expansion = int(valid_ids.numel() - 1)
        vision_len = output_valid_len - text_tokens_after_expansion
        vision_start = output_left_pad + int(image_positions[0].item())
        vision_end = vision_start + vision_len
        if vision_len <= 0 or vision_end > inputs_embeds.shape[1]:
            raise RuntimeError(
                f"Failed to infer LLaVA visual span from native multimodal output: start={vision_start}, end={vision_end}, seq={inputs_embeds.shape[1]}"
            )
        tfprune_core.set_vision_token_range(vision_start, vision_end)
        inner_model = (
            self.get_model() if hasattr(self, "get_model") else getattr(self, "model", None)
        )
        image_newline = getattr(inner_model, "image_newline", None)
        if image_newline is not None:
            visual_embeddings = inputs_embeds[0, vision_start:vision_end]
            newline = image_newline.to(
                device=visual_embeddings.device, dtype=visual_embeddings.dtype
            ).reshape(1, -1)
            newline_mask = torch.isclose(visual_embeddings, newline, rtol=0.0001, atol=1e-05).all(
                dim=-1
            )
            mandatory = torch.where(newline_mask)[0].tolist()
            tfprune_core.set_mandatory_vision_indices(mandatory)
            if mandatory:
                eval_logger.debug(
                    "[TFPrune-LLaVA] preserving {} native image-newline tokens", len(mandatory)
                )
        eval_logger.debug(
            "[TFPrune-LLaVA] native visual span=[{}, {}), tokens={}",
            vision_start,
            vision_end,
            vision_len,
        )
        return outputs

    model.prepare_inputs_labels_for_multimodal = types.MethodType(patched_prepare, model)
    model._tfprune_original_prepare_inputs_labels_for_multimodal = original_prepare
