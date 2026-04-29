"""Feature-only MLP baseline.

Critical for Elliptic: the aggregated features (cols 167..) already encode
1-hop neighbor statistics, so a plain MLP on them can rival or beat many GNNs.
Reporting MLP alongside GNNs is the cleanest way to answer
'does graph structure actually help here?' — see reports/final_report.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        assert num_layers >= 2, "MLP must have at least 2 layers"
        dims = [in_dim] + [hidden] * (num_layers - 1) + [num_classes]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 2)]
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Accepts and ignores edge_index so it can be swapped with GNNs by the trainer.
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)
