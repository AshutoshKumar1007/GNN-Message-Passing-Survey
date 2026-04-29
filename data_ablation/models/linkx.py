"""LINKX (Lim et al. 2021) — disentangles feature and adjacency MLPs.

Architecture: H = sigma( W( [MLP_X(X) || MLP_A(A)] ) + MLP_X(X) + MLP_A(A) )

We avoid the dense `nn.Linear(num_nodes, hidden)` from the original paper by
treating the adjacency branch as a learnable embedding lookup followed by a
neighbour-mean aggregation, which keeps the cost linear in |E|.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _aggregate_neighbours(
    edge_index: torch.Tensor,
    weights: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Mean-aggregate per-node weights over incoming edges (column-indexed)."""
    src, dst = edge_index
    out = torch.zeros((num_nodes, weights.size(1)), device=weights.device)
    out.index_add_(0, dst, weights[src])
    deg = torch.zeros(num_nodes, device=weights.device)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    deg = deg.clamp(min=1.0).unsqueeze(1)
    return out / deg


class LINKX(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_nodes: int,
        num_layers: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes

        # Feature branch
        self.mlp_x = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Adjacency branch (learnable per-node embedding + mean-aggregation)
        self.adj_embed = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.adj_embed)
        self.mlp_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Combined MLP with residual mixing of the two branches
        layers: list[nn.Module] = [nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(max(num_layers - 1, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp_w = nn.Sequential(*layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        nn.init.xavier_uniform_(self.adj_embed)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        return_embedding: bool = False,
        **_: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hx = self.mlp_x(x)
        ha_raw = _aggregate_neighbours(edge_index, self.adj_embed, self.num_nodes)
        ha = self.mlp_a(ha_raw)

        combined = torch.cat([hx, ha], dim=1)
        h = self.mlp_w(combined)
        h = h + hx + ha          # residual mixing as per LINKX paper
        h = F.relu(h)
        h = self.dropout(h)

        logits = self.classifier(h)
        if return_embedding:
            return logits, h
        return logits
