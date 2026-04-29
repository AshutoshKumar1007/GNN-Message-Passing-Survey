"""GraphSAGE (Hamilton et al. 2017) with mean/max aggregation.

Inductive by design; pairs naturally with NeighborLoader for 4GB-VRAM training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.3, aggr: str = "mean"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(in_dim, num_classes, aggr=aggr))
        else:
            self.convs.append(SAGEConv(in_dim, hidden, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden, hidden, aggr=aggr))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.convs.append(SAGEConv(hidden, num_classes, aggr=aggr))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)
