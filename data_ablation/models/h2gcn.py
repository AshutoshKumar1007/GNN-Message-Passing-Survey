"""H2GCN (Zhu et al. 2020 — 'Beyond Homophily in GNNs').

Faithful re-implementation of three design choices from the paper:

* (D1) ego / neighbour separation
* (D2) higher-order neighbourhoods (1-hop & 2-hop separately)
* (D3) intermediate-layer concatenation before the final classifier

Compared to the notebook prototype this version uses sparse matrices end-to-end
so it scales to the heterophilic and temporal datasets without OOM. Self-loops
are intentionally omitted on the propagation matrices, which is the critical
fix versus the original notebook.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, to_undirected


class H2GCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_ego_neighbor_separation: bool = True,
    ) -> None:
        super().__init__()
        self.K = num_layers
        self.use_ego = use_ego_neighbor_separation

        self.embed = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_embeddings = nn.ModuleList()
        for _ in range(num_layers):
            in_d = hidden_dim * (4 if use_ego_neighbor_separation else 2)
            self.layer_embeddings.append(nn.Linear(in_d, hidden_dim))
        self.classifier = nn.Linear(hidden_dim * (num_layers + 1), num_classes)

        # Cached sparse propagation matrices keyed by (edge_index version, n)
        self._cache_key: tuple[int, int] | None = None
        self._A1: torch.Tensor | None = None
        self._A2: torch.Tensor | None = None

    def reset_parameters(self) -> None:
        self.embed.reset_parameters()
        for layer in self.layer_embeddings:
            layer.reset_parameters()
        self.classifier.reset_parameters()

    # ------------------------------------------------------------------
    @staticmethod
    def _sparse_norm(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Row-normalised D^{-1} A as sparse COO."""
        device = edge_index.device
        values = torch.ones(edge_index.size(1), device=device)
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, edge_index[0], values)
        deg = deg.clamp(min=1.0)
        norm_values = values / deg[edge_index[0]]
        return torch.sparse_coo_tensor(
            edge_index, norm_values, (num_nodes, num_nodes)
        ).coalesce()

    @staticmethod
    def _two_hop_edge_index(
        edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute the 2-hop neighbourhood edge_index (excluding 1-hop & self)."""
        device = edge_index.device
        ei = edge_index
        # Build sparse A
        ones = torch.ones(ei.size(1), device=device)
        A = torch.sparse_coo_tensor(ei, ones, (num_nodes, num_nodes)).coalesce()

        # A @ A → 2-hop reachability (sparse)
        A2 = torch.sparse.mm(A, A).coalesce()
        idx = A2.indices()
        # Remove self-loops and 1-hop edges
        one_hop = set(map(tuple, ei.t().tolist()))
        keep = []
        for k in range(idx.size(1)):
            i, j = int(idx[0, k]), int(idx[1, k])
            if i == j:
                continue
            if (i, j) in one_hop:
                continue
            keep.append(k)
        if not keep:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        keep_t = torch.tensor(keep, device=device, dtype=torch.long)
        return idx[:, keep_t]

    def _build_propagation(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (A1_norm, A2_norm) as sparse tensors with caching."""
        key = (id(edge_index), num_nodes)
        if self._cache_key == key and self._A1 is not None and self._A2 is not None:
            return self._A1, self._A2

        ei = to_undirected(remove_self_loops(edge_index)[0], num_nodes=num_nodes)
        A1 = self._sparse_norm(ei, num_nodes)

        ei2 = self._two_hop_edge_index(ei, num_nodes)
        if ei2.size(1) == 0:
            A2 = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=ei.device),
                torch.empty(0, device=ei.device),
                (num_nodes, num_nodes),
            ).coalesce()
        else:
            A2 = self._sparse_norm(ei2, num_nodes)

        self._cache_key = key
        self._A1, self._A2 = A1, A2
        return A1, A2

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        return_embedding: bool = False,
        **_: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        n = x.size(0)
        A1, A2 = self._build_propagation(edge_index, n)

        r0 = F.relu(self.embed(x))
        r0 = self.dropout(r0)
        representations = [r0]
        r_prev = r0

        for k in range(self.K):
            n1 = torch.sparse.mm(A1, r_prev)
            n2 = torch.sparse.mm(A2, r_prev)
            if self.use_ego:
                concat = torch.cat([r_prev, n1, r_prev, n2], dim=1)
            else:
                concat = torch.cat([n1, n2], dim=1)
            r_k = F.relu(self.layer_embeddings[k](concat))
            r_k = self.dropout(r_k)
            representations.append(r_k)
            r_prev = r_k

        emb = torch.cat(representations, dim=1)
        emb = self.dropout(emb)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits
