"""TGN-style temporal node classifier (Rossi et al. 2020).

We use a snapshot-based approximation of TGN that:

1. Sorts edges by timestamp and bins them into discrete temporal snapshots.
2. Maintains a per-node memory updated by GRU on each interaction.
3. Computes node embeddings via a temporal attention layer over recent
   neighbours.

This implementation is faithful enough for static node classification on
Elliptic / Yelp while keeping the dependency surface to vanilla PyG. For
edge prediction or full continuous-time evaluation the official `TGNMemory`
module from `torch_geometric.nn.models.tgn` should be used instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class TGNNodeClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_nodes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        time_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.time_dim = time_dim

        # Per-node memory updated by GRU on every interaction
        self.memory = nn.Parameter(torch.zeros(num_nodes, hidden_dim), requires_grad=False)
        self.gru = nn.GRUCell(in_dim + time_dim, hidden_dim)

        # Time encoder (Cosine-Sinusoid as in TGN paper)
        self.time_w = nn.Linear(1, time_dim)

        # Embedding network: graph attention over current neighbours
        self.feat_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(hidden_dim * 2, hidden_dim, heads=2, concat=False, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(
                TransformerConv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=dropout)
            )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.memory)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.GRUCell)):
                module.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def reset_memory(self) -> None:
        with torch.no_grad():
            self.memory.zero_()

    @staticmethod
    def _time_features(t: torch.Tensor) -> torch.Tensor:
        return t.float().unsqueeze(-1)

    def update_memory(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
    ) -> None:
        """Update the per-node memory using temporal interactions."""
        if edge_index.numel() == 0:
            return
        order = torch.argsort(edge_time)
        ei = edge_index[:, order]
        et = edge_time[order]

        time_feat = F.relu(self.time_w(self._time_features(et)))
        # Each interaction: dst-node receives msg = x_src ⊕ time_feat
        msgs = torch.cat([x[ei[0]], time_feat], dim=1)
        # Group by destination (sequential GRU per destination)
        new_memory = self.memory.clone()
        for k in range(ei.size(1)):
            dst = int(ei[1, k])
            new_memory[dst] = self.gru(msgs[k : k + 1], new_memory[dst : dst + 1]).squeeze(0)
        with torch.no_grad():
            self.memory.copy_(new_memory)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_time: torch.Tensor | None = None,
        update_memory: bool = False,
        return_embedding: bool = False,
        **_: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if update_memory and edge_time is not None:
            self.update_memory(x, edge_index, edge_time)

        feat = F.relu(self.feat_proj(x))
        emb = torch.cat([feat, self.memory[: x.size(0)]], dim=1)
        for conv in self.convs:
            emb = F.relu(conv(emb, edge_index))
            emb = self.dropout(emb)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits
