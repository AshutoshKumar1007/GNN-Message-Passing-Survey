"""Unified dataset loader returning a normalised `DatasetBundle`.

The bundle hides PyG-vs-custom differences from the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import torch
from torch_geometric.data import Data

from .homophily import (
    class_insensitive_homophily,
    edge_homophily,
    node_homophily,
)


class DatasetCategory(str, Enum):
    HOMOPHILIC = "homophilic"
    HETEROPHILIC = "heterophilic"
    TEMPORAL = "temporal"


@dataclass
class DatasetBundle:
    """Common dataset interface used by training / analysis."""

    name: str
    category: DatasetCategory
    data: Data                     # PyG Data with .x, .edge_index, .y, masks
    num_classes: int
    num_features: int
    num_splits: int = 1
    edge_attr: torch.Tensor | None = None      # for TGN: timestamps
    extra: dict[str, Any] | None = None

    def homophily_summary(self) -> dict[str, float]:
        labels = self.data.y.view(-1)
        ei = self.data.edge_index
        return {
            "edge_homophily": edge_homophily(ei, labels),
            "node_homophily": node_homophily(ei, labels),
            "class_insensitive_homophily": class_insensitive_homophily(ei, labels),
        }


# ---------------------------------------------------------------------------
# Loader implementations
# ---------------------------------------------------------------------------


def _planetoid(name: str, root: str) -> DatasetBundle:
    from torch_geometric.datasets import Planetoid

    ds = Planetoid(root=f"{root}/{name}", name=name.capitalize())
    return DatasetBundle(
        name=name,
        category=DatasetCategory.HOMOPHILIC,
        data=ds[0],
        num_classes=ds.num_classes,
        num_features=ds.num_features,
        num_splits=1,
    )


def _webkb(name: str, root: str) -> DatasetBundle:
    from torch_geometric.datasets import WebKB

    ds = WebKB(root=f"{root}/{name}", name=name.capitalize())
    data = ds[0]
    splits = data.train_mask.shape[1] if data.train_mask.dim() > 1 else 1
    return DatasetBundle(
        name=name,
        category=DatasetCategory.HETEROPHILIC,
        data=data,
        num_classes=ds.num_classes,
        num_features=ds.num_features,
        num_splits=splits,
    )


def _wikipedia(name: str, root: str) -> DatasetBundle:
    from torch_geometric.datasets import WikipediaNetwork

    ds = WikipediaNetwork(root=f"{root}/{name}", name=name)
    data = ds[0]
    splits = data.train_mask.shape[1] if data.train_mask.dim() > 1 else 1
    return DatasetBundle(
        name=name,
        category=DatasetCategory.HETEROPHILIC,
        data=data,
        num_classes=ds.num_classes,
        num_features=ds.num_features,
        num_splits=splits,
    )


def _actor(name: str, root: str) -> DatasetBundle:
    from torch_geometric.datasets import Actor

    ds = Actor(root=f"{root}/actor")
    data = ds[0]
    splits = data.train_mask.shape[1] if data.train_mask.dim() > 1 else 1
    return DatasetBundle(
        name=name,
        category=DatasetCategory.HETEROPHILIC,
        data=data,
        num_classes=ds.num_classes,
        num_features=ds.num_features,
        num_splits=splits,
    )


def _amazon(name: str, root: str) -> DatasetBundle:
    """Amazon Computers / Photo (homophilic baselines without canonical splits)."""
    from torch_geometric.datasets import Amazon

    sub = name.split("_", 1)[1].capitalize()
    ds = Amazon(root=f"{root}/amazon-{sub}", name=sub)
    data = ds[0]
    if not hasattr(data, "train_mask") or data.train_mask is None:
        _attach_random_split(data, num_classes=ds.num_classes, train_per_class=20, val=500, test=1000)
    return DatasetBundle(
        name=name,
        category=DatasetCategory.HOMOPHILIC,
        data=data,
        num_classes=ds.num_classes,
        num_features=ds.num_features,
        num_splits=1,
    )


def _elliptic(name: str, root: str) -> DatasetBundle:
    """Elliptic bitcoin temporal graph for illicit-account classification."""
    from torch_geometric.datasets import EllipticBitcoinDataset

    ds = EllipticBitcoinDataset(root=f"{root}/elliptic")
    data = ds[0]
    return DatasetBundle(
        name=name,
        category=DatasetCategory.TEMPORAL,
        data=data,
        num_classes=ds.num_classes,
        num_features=ds.num_features,
        num_splits=1,
        edge_attr=getattr(data, "edge_attr", None),
        extra={"node_time": getattr(data, "node_time", None)},
    )


def _yelp(name: str, root: str) -> DatasetBundle:
    """YelpChi anomaly detection (heterophilic, near-temporal review graph)."""
    from torch_geometric.datasets import Yelp

    ds = Yelp(root=f"{root}/yelp")
    data = ds[0]
    if data.y.dim() > 1 and data.y.shape[1] > 1:
        # Yelp ships multi-label; collapse to argmax for node classification
        data.y = data.y.argmax(dim=1)
    num_classes = int(data.y.max().item()) + 1
    if not hasattr(data, "train_mask") or data.train_mask is None:
        _attach_random_split(data, num_classes=num_classes, train_per_class=200, val=2000, test=5000)
    return DatasetBundle(
        name=name,
        category=DatasetCategory.TEMPORAL,
        data=data,
        num_classes=num_classes,
        num_features=data.num_features,
        num_splits=1,
    )


def _attach_random_split(
    data: Data,
    *,
    num_classes: int,
    train_per_class: int,
    val: int,
    test: int,
) -> None:
    """Attach reproducible train / val / test masks to a PyG Data object."""
    n = data.num_nodes
    g = torch.Generator().manual_seed(0)

    train_mask = torch.zeros(n, dtype=torch.bool)
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        perm = idx[torch.randperm(idx.size(0), generator=g)]
        train_mask[perm[:train_per_class]] = True

    remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
    perm = remaining[torch.randperm(remaining.size(0), generator=g)]
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[perm[:val]] = True
    test_mask[perm[val:val + test]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, tuple[DatasetCategory, Callable[[str, str], DatasetBundle]]] = {
    # Homophilic
    "cora": (DatasetCategory.HOMOPHILIC, _planetoid),
    "citeseer": (DatasetCategory.HOMOPHILIC, _planetoid),
    "pubmed": (DatasetCategory.HOMOPHILIC, _planetoid),
    "amazon_computers": (DatasetCategory.HOMOPHILIC, _amazon),
    "amazon_photo": (DatasetCategory.HOMOPHILIC, _amazon),
    # Heterophilic
    "texas": (DatasetCategory.HETEROPHILIC, _webkb),
    "cornell": (DatasetCategory.HETEROPHILIC, _webkb),
    "wisconsin": (DatasetCategory.HETEROPHILIC, _webkb),
    "chameleon": (DatasetCategory.HETEROPHILIC, _wikipedia),
    "squirrel": (DatasetCategory.HETEROPHILIC, _wikipedia),
    "actor": (DatasetCategory.HETEROPHILIC, _actor),
    # Temporal / dynamic
    "elliptic": (DatasetCategory.TEMPORAL, _elliptic),
    "yelp": (DatasetCategory.TEMPORAL, _yelp),
}


def is_temporal(name: str) -> bool:
    return DATASET_REGISTRY[name][0] == DatasetCategory.TEMPORAL


def load_dataset(name: str, root: str | Path = "data") -> DatasetBundle:
    """Load a dataset by short name, returning a normalised `DatasetBundle`."""
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Known: {sorted(DATASET_REGISTRY.keys())}"
        )
    _, loader = DATASET_REGISTRY[key]
    root = str(root)
    Path(root).mkdir(parents=True, exist_ok=True)
    return loader(key, root)
