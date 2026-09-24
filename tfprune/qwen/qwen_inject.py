# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
"""Attach proportional visual-token compression to Qwen2.5-VL."""

from ..core import TFPruneCore
from .qwen_attention_patch import patch_qwen_attention_layers


def tfprune_qwen2_5_vl(qwen_model, processor=None, target_retention_ratio=0.222):
    core = TFPruneCore(target_retention_ratio=target_retention_ratio, device=qwen_model.device)
    qwen_model.model._tfprune_core = core
    patch_qwen_attention_layers(qwen_model, core)
    return (qwen_model, processor)
