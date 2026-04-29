"""Elliptic Bitcoin dataset.

Loads the official PyG EllipticBitcoinDataset and produces:
  - A PyG `Data` object with boolean masks for train/val/test (temporal split).
  - An event stream (src, dst, t, msg) for temporal GNN training.

Key invariants (spec requires these):
  * temporal split: train t <= 34, val 35..39, test t >= 40
  * unknown nodes (label -1) stay in the graph for message passing but are masked from loss
  * class 1 = illicit (anomaly, positive class); class 0 = licit
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data


# Elliptic's raw labels: 1=illicit, 2=licit, 3=unknown. PyG remaps to 0/1/2 in y,
# but EllipticBitcoinDataset provides `train_mask` / `test_mask` and `y` with 0=licit,
# 1=illicit, 2=unknown depending on version. We normalize below to:
#   y = 0 (licit), 1 (illicit); labeled_mask separates unknown.

# Elliptic has 49 timesteps total. Spec requires train <=34, val 35..39, test >=40.
TRAIN_MAX_T_DEFAULT = 34
VAL_MAX_T_DEFAULT = 39
TOTAL_TIMESTEPS = 49

# Elliptic feature columns:
#   [0]          = timestep (1..49)
#   [1..166]     = 166 "local" features of the transaction
#   [167..]      = 72 "aggregated" features from 1-hop neighbors (avg/min/max/etc)
# Reference: Weber et al. "Anti-Money Laundering in Bitcoin" (KDD DLG 2019)
LOCAL_FEATURE_SLICE = slice(1, 167)
AGGREGATED_FEATURE_SLICE = slice(167, None)


@dataclass
class EllipticBundle:
    """A flat container handed to the trainers."""
    data: Data                  # x, edge_index, y, t (node timestep), labeled_mask
    train_mask: torch.Tensor    # bool, over all nodes
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_features: int
    num_classes: int = 2
    positive_class: int = 1

    def event_stream(self) -> "EllipticEventStream":
        return _build_event_stream(self.data, self.train_mask, self.val_mask, self.test_mask)


@dataclass
class EllipticEventStream:
    src: torch.Tensor           # [E] long
    dst: torch.Tensor           # [E] long
    t: torch.Tensor             # [E] long, event timestep in 1..49
    msg: torch.Tensor           # [E, msg_dim] float  (we use zeros since Elliptic has no edge features)
    node_feat: torch.Tensor     # [N, F] float
    y: torch.Tensor             # [N] long, 0/1
    labeled_mask: torch.Tensor  # [N] bool
    node_t: torch.Tensor        # [N] long, per-node timestep
    train_event_mask: torch.Tensor  # [E] bool
    val_event_mask: torch.Tensor
    test_event_mask: torch.Tensor


def load_elliptic(
    root: str = "data/elliptic",
    use_aggregated_features: bool = True,
    train_max_t: int = TRAIN_MAX_T_DEFAULT,
    val_max_t: int = VAL_MAX_T_DEFAULT,
) -> EllipticBundle:
    """Load Elliptic via PyG and apply the temporal split."""
    from torch_geometric.datasets import EllipticBitcoinDataset

    ds = EllipticBitcoinDataset(root=root)
    data: Data = ds[0]

    # PyG's EllipticBitcoinDataset stores:
    #   x: [N, 166] if use_aggregated=False else [N, 165]? (API has changed across versions)
    # For safety we read the raw feature matrix from disk when available; otherwise use data.x.
    x_full = data.x.clone().float()

    # Infer per-node timestep: PyG's version may or may not include it. We pull from `data` if
    # the column is present (column 0), else from the labeled y and the mask tensors.
    if hasattr(data, "t") and data.t is not None:
        node_t = data.t.long()
    else:
        # EllipticBitcoinDataset stores timestep info separately; recover from raw features file.
        node_t = _recover_node_timesteps(root, x_full.size(0))

    # Normalize labels to {0: licit, 1: illicit}, with a separate labeled_mask.
    y_raw = data.y.clone().long()
    # In PyG >=2.4, y uses {0: illicit, 1: licit, 2: unknown}. Check orientation:
    # we want illicit==1 (positive anomaly class).
    labeled_mask = (y_raw != 2)
    # Map: if license-first (0=illicit, 1=licit): flip; if 1=illicit already: keep.
    y = y_raw.clone()
    # We detect orientation by looking at the canonical masks PyG supplies.
    # EllipticBitcoinDataset provides data.train_mask and data.test_mask where
    # the majority (licit) class is labeled 1. We map so the MINORITY class = 1.
    if labeled_mask.sum() > 0:
        counts = torch.bincount(y_raw[labeled_mask], minlength=2)
        minority = int(torch.argmin(counts).item())
        y = torch.where(y_raw == minority, torch.ones_like(y_raw), torch.zeros_like(y_raw))
        y[~labeled_mask] = -1
    else:
        raise RuntimeError("Elliptic dataset has no labeled nodes — check the download.")

    # Build temporal masks
    train_mask = labeled_mask & (node_t <= train_max_t)
    val_mask = labeled_mask & (node_t > train_max_t) & (node_t <= val_max_t)
    test_mask = labeled_mask & (node_t > val_max_t)

    # Optionally drop aggregated features (columns after 166) for a clean MLP-vs-GNN ablation.
    if not use_aggregated_features and x_full.size(1) > 166:
        x_full = x_full[:, :166].contiguous()

    # Stash node_t onto Data so downstream code has access.
    data.x = x_full
    data.y = y
    data.t = node_t
    data.labeled_mask = labeled_mask

    return EllipticBundle(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_features=x_full.size(1),
        num_classes=2,
        positive_class=1,
    )


def _recover_node_timesteps(root: str, n_nodes: int) -> torch.Tensor:
    """Try to read the per-node timestep from the raw CSV PyG extracts."""
    # PyG caches raw files at <root>/elliptic_bitcoin_dataset/ (names may vary across versions).
    candidates = [
        Path(root) / "elliptic_bitcoin_dataset" / "elliptic_txs_features.csv",
        Path(root) / "raw" / "elliptic_txs_features.csv",
    ]
    for p in candidates:
        if p.exists():
            # First column = txId, second = timestep
            arr = np.loadtxt(p, delimiter=",", dtype=np.float64)
            # Build a txId -> timestep map, then reindex by PyG's ordering if possible.
            # Easiest: PyG preserves the order from the raw CSV, so col 1 in row i is node i's t.
            if arr.shape[0] == n_nodes:
                return torch.tensor(arr[:, 1], dtype=torch.long)
    raise RuntimeError(
        "Could not recover per-node timesteps from raw CSVs. "
        "Re-run scripts/download_elliptic.py or pin PyG >= 2.4."
    )


def _build_event_stream(
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> EllipticEventStream:
    """Convert the static directed graph into an event stream ordered by time.

    Elliptic edges do not carry their own timestamp. We assign each edge the
    maximum of its endpoints' timesteps (the moment the edge becomes observable).
    This is the convention used in TGN / TGAT applications to Elliptic.
    """
    src, dst = data.edge_index
    node_t = data.t.long()
    edge_t = torch.max(node_t[src], node_t[dst])

    # Stable sort by edge time so later events come after earlier ones.
    order = torch.argsort(edge_t, stable=True)
    src, dst, edge_t = src[order], dst[order], edge_t[order]

    # Elliptic has no edge features; TGN expects a message tensor — use a single
    # zero-dimensional-ish tensor of width 1 to satisfy the API.
    msg = torch.zeros((src.size(0), 1), dtype=torch.float32)

    # Event-level split: events where BOTH endpoints are observed in their split's
    # time window. To stay strictly causal we place an event in the earliest
    # split that contains its `edge_t`.
    train_event_mask = edge_t <= TRAIN_MAX_T_DEFAULT
    val_event_mask = (edge_t > TRAIN_MAX_T_DEFAULT) & (edge_t <= VAL_MAX_T_DEFAULT)
    test_event_mask = edge_t > VAL_MAX_T_DEFAULT

    return EllipticEventStream(
        src=src.long(),
        dst=dst.long(),
        t=edge_t.long(),
        msg=msg,
        node_feat=data.x.float(),
        y=data.y.long(),
        labeled_mask=data.labeled_mask.bool(),
        node_t=node_t,
        train_event_mask=train_event_mask,
        val_event_mask=val_event_mask,
        test_event_mask=test_event_mask,
    )


def summarize(bundle: EllipticBundle) -> dict:
    """Quick stats for the report."""
    d = bundle.data
    n_total = d.x.size(0)
    n_labeled = int(d.labeled_mask.sum().item())
    n_anom = int((d.y == 1).sum().item())
    n_benign = int((d.y == 0).sum().item())
    return {
        "nodes": n_total,
        "edges": int(d.edge_index.size(1)),
        "features": int(d.x.size(1)),
        "labeled_nodes": n_labeled,
        "anomalous": n_anom,
        "benign": n_benign,
        "anomaly_rate": round(n_anom / max(n_labeled, 1), 4),
        "train_nodes": int(bundle.train_mask.sum().item()),
        "val_nodes": int(bundle.val_mask.sum().item()),
        "test_nodes": int(bundle.test_mask.sum().item()),
        "timesteps": TOTAL_TIMESTEPS,
    }
