"""GAT / GATv2 baseline (Veličković et al. 2018, Brody et al. 2022).

Uses GATv2Conv — strictly more expressive than original GAT (dynamic attention)
and the default in PyG's modern examples.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(
                GATv2Conv(in_dim, num_classes, heads=1, concat=False, dropout=dropout)
            )
        else:
            self.convs.append(
                GATv2Conv(in_dim, hidden, heads=heads, concat=True, dropout=dropout)
            )
            self.bns.append(nn.BatchNorm1d(hidden * heads))
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATv2Conv(hidden * heads, hidden, heads=heads, concat=True, dropout=dropout)
                )
                self.bns.append(nn.BatchNorm1d(hidden * heads))
            self.convs.append(
                GATv2Conv(hidden * heads, num_classes, heads=1, concat=False, dropout=dropout)
            )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)
