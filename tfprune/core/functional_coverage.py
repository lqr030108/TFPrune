"""Farthest-first functional selection and nonnegative evidence transport."""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class FunctionalCoverageSelection:
    indices: List[int]
    assignments: torch.Tensor
    effective_masses: torch.Tensor
    transport_weights: torch.Tensor
    merged_states: torch.Tensor
    requested_tokens: Optional[int] = None


class FunctionalCoverageSelector:
    """Cover normalized response directions at an exact token budget."""

    @torch.no_grad()
    def select_one(self, signatures, hidden_states, target_tokens, metric_signatures=None):
        if signatures.ndim != 2 or signatures.shape[0] == 0:
            raise ValueError("signatures must have shape [N, R] with N > 0")
        if hidden_states.ndim != 2 or hidden_states.shape[0] != signatures.shape[0]:
            raise ValueError("hidden_states must have shape [N, D]")
        if target_tokens is None or int(target_tokens) < 1:
            raise ValueError("An explicit positive token budget is required")
        if metric_signatures is None:
            metric_signatures = signatures
        if metric_signatures.ndim != 2 or metric_signatures.shape[0] != signatures.shape[0]:
            raise ValueError("metric_signatures must have shape [N, R_metric]")
        n = signatures.shape[0]
        k = min(int(target_tokens), n)
        energy = signatures.float().square().sum(-1)
        normalized = F.normalize(metric_signatures.float(), dim=-1, eps=1e-08)
        similarity_matrix = None
        if n <= 1024 and k >= 16 and (n * n * 4 <= 64 * 1024**2):
            similarity_matrix = (normalized @ normalized.T).clamp(-1.0, 1.0)
        first = torch.argmax(energy)
        centers = torch.empty(k, dtype=torch.long, device=signatures.device)
        centers[0] = first
        selected = torch.zeros(n, dtype=torch.bool, device=signatures.device)
        selected[first] = True
        assignments = torch.zeros(n, dtype=torch.long, device=signatures.device)
        best = (
            similarity_matrix[:, first].clone()
            if similarity_matrix is not None
            else (normalized @ normalized[first]).clamp(-1.0, 1.0)
        )
        for step in range(1, k):
            candidate = torch.argmin(best.masked_fill(selected, torch.inf))
            centers[step] = candidate
            selected[candidate] = True
            similarity = (
                similarity_matrix[:, candidate]
                if similarity_matrix is not None
                else (normalized @ normalized[candidate]).clamp(-1.0, 1.0)
            )
            improved = similarity > best
            best = torch.maximum(best, similarity)
            assignments[improved] = step
        assignments[centers] = torch.arange(k, device=centers.device)
        assigned = signatures.index_select(0, centers.index_select(0, assignments)).float()
        weights = (
            (signatures.float() * assigned).sum(-1) / assigned.square().sum(-1).clamp_min(1e-08)
        ).clamp(0.0, 8.0)
        weights[centers] = 1.0
        masses = torch.zeros(k, device=signatures.device, dtype=torch.float32)
        masses.index_add_(0, assignments, weights)
        merged = torch.zeros(
            k, hidden_states.shape[-1], device=hidden_states.device, dtype=torch.float32
        )
        merged.index_add_(0, assignments, hidden_states.float() * weights[:, None])
        merged = merged / masses.clamp_min(1e-08)[:, None]
        return FunctionalCoverageSelection(
            centers.tolist(), assignments, masses, weights, merged, k
        )
