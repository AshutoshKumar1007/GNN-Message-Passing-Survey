"""GraphSAGE (Hamilton, Ying, Leskovec 2017) using PyG `SAGEConv`.

The earlier hand-rolled aggregator showed instability because it forgot to
project the self-features and used uneven LR for the two halves of the layer.
PyG's `SAGEConv` handles both correctly and supports neighbour sampling for
scalable training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggregator: str = "mean",
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggregator))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        self.convs.append(SAGEConv(hidden_dim, num_classes, aggr=aggregator))

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
