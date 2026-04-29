"""Dataset loaders covering homophilic, heterophilic, and temporal regimes."""

from .registry import (
    DATASET_REGISTRY,
    DatasetBundle,
    DatasetCategory,
    is_temporal,
    load_dataset,
)
from .homophily import edge_homophily, node_homophily, class_insensitive_homophily

__all__ = [
    "DATASET_REGISTRY",
    "DatasetBundle",
    "DatasetCategory",
    "is_temporal",
    "load_dataset",
    "edge_homophily",
    "node_homophily",
    "class_insensitive_homophily",
]
