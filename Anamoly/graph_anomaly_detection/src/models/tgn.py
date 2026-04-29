"""Temporal Graph Network (Rossi et al. 2020).

Components:
  * Memory       : per-node vector s_i(t) carrying compressed history
  * Message fn   : m(s_src, s_dst, dt, edge_feat) -> message
  * Memory updater: GRU(m, s)  -> s'
  * Embedding    : temporal graph attention over recent neighbors produces z_i(t)
  * Classifier   : z_i(t) -> anomaly logit

Built on PyG's `torch_geometric.nn.models.tgn.TGNMemory` + a graph-attention
embedding module. PyG already ships a mature TGNMemory, which saves us from
reimplementing the raw-message store / last-update tracking; the rest
(embedding + classifier) lives here so you can see and tweak it.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    MeanAggregator,
    TGNMemory,
)


class TimeEncoder(nn.Module):
    """Cosine-based time encoding from TGAT/TGN.

    phi(dt) = cos(w * dt + b), learned w, b.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [E]
        return torch.cos(self.w(t.float().unsqueeze(-1)))


class GraphAttentionEmbedding(nn.Module):
    """Single-layer temporal graph-attention embedding over sampled neighbors.

    For each target node, takes its current memory + raw features, attends over
    its sampled neighbors using (neighbor memory + neighbor features + time-encoded dt)
    as keys/values, and returns a fused embedding z.
    """

    def __init__(self, in_dim: int, out_dim: int, msg_dim: int, time_enc: TimeEncoder,
                 heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.dim
        self.conv = TransformerConv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
        )

    def forward(self, x: torch.Tensor, last_update: torch.Tensor,
                edge_index: torch.Tensor, t: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        # Relative time between the event and the last update of each dst node
        rel_t = t - last_update[edge_index[0]]
        rel_t_enc = self.time_enc(rel_t)
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class TGN(nn.Module):
    """Full TGN model adapted to node classification.

    Forward usage:
        tgn.memory.reset_state()             # at the start of each epoch
        for batch in temporal_loader:        # TemporalDataLoader over sorted events
            n_id, edge_index, e_id = neighbor_loader(batch.n_id)
            z = tgn(n_id, edge_index, e_id, batch.t)          # node embeddings at time t
            logits = tgn.classifier(z[target_nodes])
            loss = F.cross_entropy(logits, labels)
            loss.backward(); opt.step()
            tgn.memory.update_state(src, dst, t, msg)         # write events to memory
    """

    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        node_feat_dim: int,
        memory_dim: int = 100,
        time_dim: int = 100,
        embedding_dim: int = 100,
        num_classes: int = 2,
        message_module: str = "identity",
        aggregator: str = "last",
        dropout: float = 0.1,
    ):
        super().__init__()

        # -- Message module ----------------------------------------------------
        if message_module == "identity":
            msg_mod = IdentityMessage(raw_msg_dim=raw_msg_dim,
                                      memory_dim=memory_dim,
                                      time_dim=time_dim)
            msg_out_dim = msg_mod.out_channels
        elif message_module == "mlp":
            # Simple MLP message function
            msg_out_dim = memory_dim
            class MLPMessage(nn.Module):
                def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int,
                             out_dim: int):
                    super().__init__()
                    self.out_channels = out_dim
                    in_dim = 2 * memory_dim + raw_msg_dim + time_dim
                    self.net = nn.Sequential(
                        nn.Linear(in_dim, out_dim), nn.ReLU(),
                        nn.Linear(out_dim, out_dim),
                    )

                def forward(self, z_src, z_dst, raw_msg, t_enc):
                    return self.net(torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1))

            msg_mod = MLPMessage(raw_msg_dim, memory_dim, time_dim, msg_out_dim)
        else:
            raise ValueError(f"Unknown message_module={message_module}")

        # -- Aggregator --------------------------------------------------------
        if aggregator == "last":
            agg_mod = LastAggregator()
        elif aggregator == "mean":
            agg_mod = MeanAggregator()
        else:
            raise ValueError(f"Unknown aggregator={aggregator}")

        # -- Memory module -----------------------------------------------------
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=msg_mod,
            aggregator_module=agg_mod,
        )

        # -- Embedding module --------------------------------------------------
        self.time_enc = TimeEncoder(time_dim)
        in_dim = memory_dim + node_feat_dim
        self.embedding = GraphAttentionEmbedding(
            in_dim=in_dim,
            out_dim=embedding_dim,
            msg_dim=raw_msg_dim,
            time_enc=self.time_enc,
            heads=2,
            dropout=dropout,
        )

        # -- Classifier --------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim

    def compute_embedding(
        self,
        n_id: torch.Tensor,
        edge_index_block: torch.Tensor,
        e_id_block: torch.Tensor,
        t_targets: torch.Tensor,
        node_feat: torch.Tensor,
        edge_raw_msg: torch.Tensor,
        edge_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute node embeddings z(t) for the nodes in `n_id`.

        Parameters
        ----------
        n_id : [K] unique node IDs involved in this batch (targets + sampled neighbors)
        edge_index_block : [2, M] local edge_index into n_id
        e_id_block : [M] global edge IDs, used to gather raw messages
        t_targets : [K] the timestamp at which each node's embedding is queried
        node_feat : [N, F] all node features (we index with n_id)
        edge_raw_msg : [E_total, msg_dim] all edges' raw messages
        edge_t : [E_total] all edges' timestamps
        """
        # Pull current memory + last update for these nodes.
        memory, last_update = self.memory(n_id)
        x = torch.cat([memory, node_feat[n_id]], dim=-1)

        # Gather edge attrs for this block.
        msg_block = edge_raw_msg[e_id_block]
        t_block = edge_t[e_id_block]

        z = self.embedding(x, last_update, edge_index_block, t_block, msg_block)
        return z

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)

    def reset_state(self) -> None:
        self.memory.reset_state()

    def detach_memory(self) -> None:
        """Detach memory state between epochs to avoid backprop through time across epochs."""
        self.memory.detach()
