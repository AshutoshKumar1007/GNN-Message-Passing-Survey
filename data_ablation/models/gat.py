"""GAT (Velickovic 2018) and GATv2 (Brody, Alon, Yahav 2022).

Heads are concatenated in hidden layers and averaged in the final layer,
matching the canonical Cora setup.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv


class GAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.6,
        heads: int = 8,
        v2: bool = False,
    ) -> None:
        super().__init__()
        Conv = GATv2Conv if v2 else GATConv
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.convs.append(Conv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(
                Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
            )
        self.convs.append(
            Conv(
                hidden_dim * heads,
                num_classes,
                heads=1,
                concat=False,
                dropout=dropout,
            )
        )

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
        emb = self.dropout(x)
        for conv in self.convs[:-1]:
            emb = F.elu(conv(emb, edge_index))
            emb = self.dropout(emb)
        logits = self.convs[-1](emb, edge_index)
        if return_embedding:
            return logits, emb
        return logits
