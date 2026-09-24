"""KV-cache aggregation and compaction."""

from typing import Optional

import torch


def prune_kv_cache(past_key_values, keep_mask: torch.Tensor):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, tuple):
        return tuple(
            (
                (key[:, :, keep_mask.to(key.device), :], value[:, :, keep_mask.to(value.device), :])
                for key, value in past_key_values
            )
        )
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for layer_idx in range(len(past_key_values.key_cache)):
            key = past_key_values.key_cache[layer_idx]
            value = past_key_values.value_cache[layer_idx]
            if key is None or value is None or key.numel() == 0:
                continue
            if key.shape[-2] != keep_mask.numel():
                continue
            past_key_values.key_cache[layer_idx] = key[:, :, keep_mask.to(key.device), :]
            past_key_values.value_cache[layer_idx] = value[:, :, keep_mask.to(value.device), :]
        return past_key_values
    raise TypeError(f"Unsupported KV cache type: {type(past_key_values)!r}")


def merge_prune_kv_cache(
    past_key_values,
    keep_mask: torch.Tensor,
    vision_start: int,
    representative_indices,
    assignments: torch.Tensor,
    member_weights: Optional[torch.Tensor] = None,
):
    if past_key_values is None:
        return None
    reps = torch.as_tensor(representative_indices, device=keep_mask.device, dtype=torch.long)

    def merge_pair(key, value):
        if key.shape[-2] != keep_mask.numel():
            return (key, value)
        key = key.clone()
        value = value.clone()
        local_assignments = assignments.to(key.device)
        local_weights = (
            member_weights.to(key.device, dtype=torch.float32)
            if member_weights is not None
            else None
        )
        for cluster_idx, representative in enumerate(reps.to(key.device)):
            members = torch.where(local_assignments == cluster_idx)[0] + vision_start
            if members.numel():
                cluster_key = key.index_select(-2, members)
                cluster_value = value.index_select(-2, members)
                if local_weights is None:
                    merged_key = cluster_key.mean(dim=-2)
                    merged_value = cluster_value.mean(dim=-2)
                else:
                    weights = local_weights.index_select(0, members - vision_start)
                    if float(weights.sum().item()) <= 1e-08:
                        weights = torch.ones_like(weights)
                    weights = weights.to(cluster_key.dtype)
                    shape = [1] * cluster_key.ndim
                    shape[-2] = members.numel()
                    weights = weights.view(shape)
                    denominator = weights.sum(dim=-2).clamp_min(1e-08)
                    merged_key = (cluster_key * weights).sum(dim=-2) / denominator
                    merged_value = (cluster_value * weights).sum(dim=-2) / denominator
                key[:, :, vision_start + representative, :] = merged_key
                value[:, :, vision_start + representative, :] = merged_value
        return (key[:, :, keep_mask.to(key.device), :], value[:, :, keep_mask.to(value.device), :])

    if isinstance(past_key_values, tuple):
        return tuple((merge_pair(key, value) for key, value in past_key_values))
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for layer_idx in range(len(past_key_values.key_cache)):
            key = past_key_values.key_cache[layer_idx]
            value = past_key_values.value_cache[layer_idx]
            if key is None or value is None or key.numel() == 0:
                continue
            key, value = merge_pair(key, value)
            past_key_values.key_cache[layer_idx] = key
            past_key_values.value_cache[layer_idx] = value
        return past_key_values
    raise TypeError(f"Unsupported KV cache type: {type(past_key_values)!r}")


def cache_sequence_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if isinstance(past_key_values, tuple) and past_key_values:
        return int(past_key_values[0][0].shape[-2])
    return 0
