"""LINKX (Lim et al., NeurIPS 2021) — simple but strong heterophily baseline.

Architecture:
  h_A = MLP_A(A_row)     # embed adjacency row (so the model 'sees' who you connect to)
  h_X = MLP_X(X)         # embed features
  z   = MLP_f( [h_A, h_X, h_A + h_X] )
  out = MLP_out(z)

Works surprisingly well on many heterophilic graphs (outperforms GCN/GAT) and is
cheap — the dominant cost is embedding A which is a linear pass over sparse rows.

NOTE: For scalability on large graphs, A_row is implemented via a sparse matmul
with a learnable embedding matrix (no dense N x N materialization).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        if num_layers == 1:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
            layers.append(nn.Linear(hidden, out_dim))
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LINKX(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int,
                 num_layers: int = 1, inner_mlp_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout

        # Feature-side MLP
        self.mlp_x = _MLPBlock(in_dim, hidden, hidden, num_layers=inner_mlp_layers + 1,
                               dropout=dropout)
        # Adjacency-side: implemented as a learnable per-node embedding table
        # which, after aggregation via A, gives MLP_A(A_row) behavior at O(E) cost.
        # We initialize lazily on first forward since we need N.
        self._adj_emb: nn.Embedding | None = None
        self.mlp_a_out = _MLPBlock(hidden, hidden, hidden, num_layers=inner_mlp_layers,
                                   dropout=dropout)

        # Fusion
        self.fuse = _MLPBlock(3 * hidden, hidden, hidden, num_layers=inner_mlp_layers + 1,
                              dropout=dropout)

        # Classifier
        self.out = nn.Linear(hidden, num_classes)

    def _adj_row_embed(self, edge_index: torch.Tensor, num_nodes: int,
                       device: torch.device) -> torch.Tensor:
        """Compute MLP_A(A_row) for every node via a sparse aggregate over a learnable table."""
        if self._adj_emb is None or self._adj_emb.num_embeddings != num_nodes:
            self._adj_emb = nn.Embedding(num_nodes, self.hidden).to(device)
            nn.init.xavier_uniform_(self._adj_emb.weight)
        # Scatter-mean of per-neighbor embeddings to the source node.
        # This implements E[h_{dst}] over A_row, which is a first-order approximation
        # of MLP_A(A_row) that scales linearly in |E| and never materializes A.
        from torch_geometric.utils import scatter
        src, dst = edge_index
        neigh_emb = self._adj_emb(dst)
        h_a = scatter(neigh_emb, src, dim=0, dim_size=num_nodes, reduce="mean")
        return self.mlp_a_out(h_a)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        n = x.size(0)
        h_x = self.mlp_x(x)
        h_a = self._adj_row_embed(edge_index, n, x.device)
        z = torch.cat([h_a, h_x, h_a + h_x], dim=1)
        z = self.fuse(z)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.out(z)
