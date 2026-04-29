"""YelpChi dataset (review fraud detection).

Loads the canonical YelpChi .mat file (Dou et al., CARE-GNN 2020) and produces a
homogeneous PyG `Data` object with an inductive 40/20/40 stratified split.

The .mat file contains:
  - net_rur, net_rsr, net_rtr : sparse adjacency matrices for three relations
      R-U-R : reviews by the same user
      R-S-R : reviews on the same business with the same star rating
      R-T-R : reviews on the same business in the same month
  - features                  : (N, 32) node features
  - label                     : (N,) binary labels  (1 = fraudulent)

Download from CARE-GNN repo:
  https://github.com/YingtongDou/CARE-GNN/raw/master/data/YelpChi.zip
(scripts/download_yelp.py handles this.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


@dataclass
class YelpBundle:
    data: Data                  # x, edge_index, y
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_features: int
    num_classes: int = 2
    positive_class: int = 1


def load_yelp(
    root: str = "data/yelp",
    relations_mode: str = "union",
    train_ratio: float = 0.4,
    val_ratio: float = 0.2,
    stratified: bool = True,
    seed: int = 42,
) -> YelpBundle:
    """Load YelpChi and produce a PyG Data + masks."""
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise ImportError("scipy is required to load YelpChi.mat") from e

    root = Path(root)
    mat_path = root / "YelpChi.mat"
    if not mat_path.exists():
        raise FileNotFoundError(
            f"{mat_path} not found. Run `python scripts/download_yelp.py` first."
        )

    raw = loadmat(str(mat_path))

    features = np.asarray(raw["features"].todense() if hasattr(raw["features"], "todense") else raw["features"]).astype(np.float32)
    labels = np.asarray(raw["label"]).flatten().astype(np.int64)
    n = features.shape[0]

    relations = {
        "rur": raw.get("net_rur"),
        "rsr": raw.get("net_rsr"),
        "rtr": raw.get("net_rtr"),
    }
    # sanity
    for name, mat in relations.items():
        if mat is None:
            raise KeyError(f"YelpChi.mat missing expected relation: net_{name}")

    # Build edge_index according to `relations_mode`.
    if relations_mode == "union":
        A = relations["rur"] + relations["rsr"] + relations["rtr"]
    elif relations_mode == "rur_only":
        A = relations["rur"]
    elif relations_mode == "rsr_only":
        A = relations["rsr"]
    elif relations_mode == "rtr_only":
        A = relations["rtr"]
    else:
        raise ValueError(f"Unknown relations_mode={relations_mode}")

    A = A.tocoo()
    src = torch.from_numpy(A.row).long()
    dst = torch.from_numpy(A.col).long()
    # drop self-loops; they carry no information and inflate degree
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = torch.stack([src, dst], dim=0).contiguous()

    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)

    # Inductive split: stratified on labels. (Following PC-GNN / CARE-GNN convention.)
    idx = np.arange(n)
    if stratified:
        train_idx, temp_idx, y_train, y_temp = train_test_split(
            idx, labels, test_size=1 - train_ratio, stratify=labels, random_state=seed
        )
        # Split temp into val and test with correct proportions
        test_ratio = 1.0 - train_ratio - val_ratio
        val_frac = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, y_temp, test_size=1 - val_frac, stratify=y_temp, random_state=seed
        )
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

    def _to_mask(indices: np.ndarray) -> torch.Tensor:
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.from_numpy(np.asarray(indices)).long()] = True
        return m

    train_mask = _to_mask(train_idx)
    val_mask = _to_mask(val_idx)
    test_mask = _to_mask(test_idx)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.labeled_mask = torch.ones(n, dtype=torch.bool)  # every YelpChi node is labeled

    return YelpBundle(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_features=x.size(1),
        num_classes=2,
        positive_class=1,
    )


def summarize(bundle: YelpBundle) -> dict:
    d = bundle.data
    y = d.y.numpy()
    return {
        "nodes": int(d.x.size(0)),
        "edges": int(d.edge_index.size(1)),
        "features": int(d.x.size(1)),
        "anomalous": int((y == 1).sum()),
        "benign": int((y == 0).sum()),
        "anomaly_rate": round(float((y == 1).mean()), 4),
        "train_nodes": int(bundle.train_mask.sum().item()),
        "val_nodes": int(bundle.val_mask.sum().item()),
        "test_nodes": int(bundle.test_mask.sum().item()),
    }
