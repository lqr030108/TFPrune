# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
"""Attach TFPrune without replacing native image preprocessing."""

from ..core import TFPruneCore
from .llama_attention_patch import patch_llama_attention_layers
from .multimodal_prepare_patch import patch_native_multimodal_prepare


def tfprune(model, target_vision_tokens=128):
    core = TFPruneCore(target_vision_tokens=target_vision_tokens, device=model.device)
    patch_native_multimodal_prepare(model, core)
    patch_llama_attention_layers(model, core)
    model._tfprune_core = core
    return model
