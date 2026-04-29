"""H2GCN (Zhu et al., NeurIPS 2020) — Heterophily-aware GNN.

Design principles for heterophily:
  (i)   Separate ego- and neighbor-embeddings (no self-loop mixing).
  (ii)  Aggregate over higher-order neighborhoods (1-hop AND 2-hop) separately.
  (iii) Concatenate embeddings from all intermediate layers.

This is particularly relevant for YelpChi, where fraudsters often connect to
benign reviews (low homophily), so plain GCN's self-loop smoothing hurts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, remove_self_loops, to_undirected


def _sym_norm_adj(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.sparse.Tensor:
    """Symmetric-normalized sparse adjacency D^{-1/2} (A) D^{-1/2} without self-loops."""
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    indices = edge_index
    values = norm
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device).coalesce()


def _spspmm_two_hop(A: torch.sparse.Tensor) -> torch.sparse.Tensor:
    """Compute A @ A as a sparse tensor (2-hop normalized adjacency)."""
    A_dense_threshold = 50_000  # below this we allow dense multiply; above, use sparse
    n = A.size(0)
    if n <= A_dense_threshold:
        A_dense = A.to_dense()
        A2_dense = A_dense @ A_dense
        A2_dense.fill_diagonal_(0)  # drop self-paths
        return A2_dense.to_sparse().coalesce()
    # Sparse x sparse via torch.sparse.mm
    A2 = torch.sparse.mm(A, A.to_dense()).to_sparse().coalesce()
    # Drop diagonal
    idx, val = A2.indices(), A2.values()
    mask = idx[0] != idx[1]
    return torch.sparse_coo_tensor(idx[:, mask], val[mask], A2.shape, device=A.device).coalesce()


class H2GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int,
                 k_hops: int = 2, dropout: float = 0.3):
        super().__init__()
        self.k_hops = k_hops
        self.dropout = dropout

        # Feature embedding
        self.embed = nn.Linear(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)

        # After K rounds of aggregation with 2 neighborhoods (1-hop, 2-hop),
        # we concatenate ego + 2*K hop-embeddings -> hidden * (1 + 2*K)
        cat_dim = hidden * (1 + 2 * k_hops)
        self.classifier = nn.Sequential(
            nn.Linear(cat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

        # cached sparse adjacencies (filled on first forward)
        self._A1: torch.sparse.Tensor | None = None
        self._A2: torch.sparse.Tensor | None = None
        self._cached_n: int | None = None

    def _ensure_adj(self, edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> None:
        if self._A1 is None or self._cached_n != num_nodes:
            self._A1 = _sym_norm_adj(edge_index, num_nodes, device)
            self._A2 = _spspmm_two_hop(self._A1)
            self._cached_n = num_nodes

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        n = x.size(0)
        self._ensure_adj(edge_index, n, x.device)
        A1, A2 = self._A1, self._A2

        h = self.bn(F.relu(self.embed(x)))
        reps = [h]

        cur = h
        for _ in range(self.k_hops):
            h1 = torch.sparse.mm(A1, cur)
            h2 = torch.sparse.mm(A2, cur)
            cur = torch.cat([h1, h2], dim=1)   # NOTE: concat doubles width for next round input
            # project back to `hidden` before next hop so widths stay bounded
            cur = F.relu(cur)
            reps.append(h1)
            reps.append(h2)
            # Re-project to hidden for next iteration to keep memory bounded.
            cur = h1 + h2   # mean-ish shortcut; reps already track the separate tensors

        z = torch.cat(reps, dim=1)
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.classifier(z)
