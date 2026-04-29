"""Split helpers: pick the i-th split column, attach masks if missing."""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def resolve_split(data: Data, split_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (train, val, test) masks for the given split index."""

    def pick(mask: torch.Tensor) -> torch.Tensor:
        return mask[:, split_idx] if mask.dim() > 1 else mask

    return pick(data.train_mask), pick(data.val_mask), pick(data.test_mask)


def ensure_masks(
    data: Data,
    *,
    num_classes: int,
    train_per_class: int = 20,
    val: int = 500,
    test: int = 1000,
    generator_seed: int = 0,
) -> Data:
    """Attach masks if the dataset doesn't ship with them."""
    if hasattr(data, "train_mask") and data.train_mask is not None:
        return data

    n = data.num_nodes
    g = torch.Generator().manual_seed(generator_seed)
    train_mask = torch.zeros(n, dtype=torch.bool)
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        perm = idx[torch.randperm(idx.size(0), generator=g)]
        train_mask[perm[:train_per_class]] = True

    remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
    perm = remaining[torch.randperm(remaining.size(0), generator=g)]
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[perm[:val]] = True
    test_mask[perm[val : val + test]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data
