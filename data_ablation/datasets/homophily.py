"""Graph homophily metrics (Zhu et al. 2020 + Pei et al. 2020)."""

from __future__ import annotations

import torch
from torch_geometric.utils import remove_self_loops


def edge_homophily(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of edges connecting same-class nodes (Zhu et al. 2020)."""
    edge_index, _ = remove_self_loops(edge_index)
    src, dst = edge_index
    same = (labels[src] == labels[dst]).float()
    return float(same.mean().item())


def node_homophily(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """Average per-node homophily ratio (Pei et al. 2020)."""
    edge_index, _ = remove_self_loops(edge_index)
    src, dst = edge_index
    n = labels.size(0)

    deg = torch.zeros(n, dtype=torch.float, device=labels.device)
    same = torch.zeros(n, dtype=torch.float, device=labels.device)

    deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
    matches = (labels[src] == labels[dst]).float()
    same.scatter_add_(0, src, matches)

    mask = deg > 0
    return float((same[mask] / deg[mask]).mean().item())


def class_insensitive_homophily(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """Class-insensitive homophily \\hat{h} from Lim et al. 2021 (LINKX paper).

    Adjusts for class imbalance: a value near 0 means the graph carries no
    label-correlated structure beyond random.
    """
    labels = labels.view(-1)
    edge_index, _ = remove_self_loops(edge_index)
    num_classes = int(labels.max().item()) + 1

    # Build c x c compatibility matrix
    H = torch.zeros((num_classes, num_classes), device=labels.device)
    src, dst = edge_index
    for k in range(num_classes):
        idx = (labels[src] == k).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        targ = labels[dst[idx]]
        for j in range(num_classes):
            H[k, j] = (targ == j).sum().float()
    row_sum = H.sum(dim=1, keepdim=True).clamp(min=1.0)
    H = H / row_sum

    counts = torch.bincount(labels[labels >= 0], minlength=num_classes).float()
    proportions = counts / counts.sum().clamp(min=1.0)

    val = 0.0
    for k in range(num_classes):
        diff = (H[k, k] - proportions[k]).clamp(min=0.0)
        if not torch.isnan(diff):
            val += float(diff.item())
    denom = max(num_classes - 1, 1)
    return val / denom
