"""Training loops: full-batch, neighbour-sampled, and temporal."""

from .feature_transform import apply_feature_setting
from .full_batch import train_full_batch
from .sampled import train_with_neighbor_sampling
from .temporal import train_temporal
from .splits import resolve_split, ensure_masks

__all__ = [
    "apply_feature_setting",
    "train_full_batch",
    "train_with_neighbor_sampling",
    "train_temporal",
    "resolve_split",
    "ensure_masks",
]
