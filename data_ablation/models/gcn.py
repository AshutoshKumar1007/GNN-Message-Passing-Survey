"""2-layer GCN (Kipf & Welling 2017) using PyG sparse ops.

The original notebook used a dense adjacency multiplication, which OOMs on
graphs > ~30k nodes. We use `GCNConv` so the same architecture scales to
Yelp / Elliptic without code changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("GCN expects at least 2 layers")

        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim, cached=False, add_self_loops=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_dim, hidden_dim, cached=False, add_self_loops=True)
            )
        self.convs.append(GCNConv(hidden_dim, num_classes, cached=False, add_self_loops=True))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        return_embedding: bool = False,
        **_: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        emb = x
        for conv in self.convs[:-1]:
            emb = F.relu(conv(emb, edge_index))
            emb = self.dropout(emb)
        logits = self.convs[-1](emb, edge_index)
        if return_embedding:
            return logits, emb
        return logits
