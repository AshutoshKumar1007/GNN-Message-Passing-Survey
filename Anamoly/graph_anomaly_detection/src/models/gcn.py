"""GCN baseline (Kipf & Welling 2017).

2-layer spectral GCN with symmetric normalized propagation, batch-norm, and dropout.
Best-known homophilic baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        assert num_layers >= 1
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(in_dim, num_classes, cached=False, add_self_loops=True))
        else:
            self.convs.append(GCNConv(in_dim, hidden, cached=False, add_self_loops=True))
            self.bns.append(nn.BatchNorm1d(hidden))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden, hidden, cached=False, add_self_loops=True))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.convs.append(GCNConv(hidden, num_classes, cached=False, add_self_loops=True))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index, edge_weight)
