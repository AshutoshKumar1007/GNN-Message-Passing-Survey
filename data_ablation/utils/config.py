"""Frozen dataclasses describing experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FeatureSetting(str, Enum):
    """The four feature ablation settings required by the research plan."""

    FULL = "full"
    NONE = "none"          # Structure-only (replace X with identity / zeros)
    RANDOM = "random"      # Same dim as X but i.i.d. Gaussian
    TOPK = "topk"          # Selected via variance / correlation / model-based


class FeatureSelector(str, Enum):
    VARIANCE = "variance"
    CORRELATION = "correlation"
    RANDOM_FOREST = "random_forest"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 500
    lr: float = 0.01
    weight_decay: float = 5e-4
    patience: int = 100
    batch_size: int = 100           # 0 = full-batch
    num_neighbors: tuple = (5,5)


@dataclass(frozen=True)
class RunConfig:
    """Single (model  dataset  feature_setting  seed) run."""

    model: str
    dataset: str
    feature_setting: FeatureSetting = FeatureSetting.FULL
    feature_selector: FeatureSelector | None = None
    topk: int | None = None
    seed: int = 42
    hidden_dim: int = 64
    dropout: float = 0.5
    num_layers: int = 2
    train: TrainConfig = field(default_factory=TrainConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    def short_id(self) -> str:
        parts = [self.dataset, self.model, self.feature_setting.value]
        if self.feature_setting == FeatureSetting.TOPK:
            parts.append(self.feature_selector.value if self.feature_selector else "novar")
            parts.append(f"k{self.topk}")
        parts.append(f"s{self.seed}")
        return "_".join(parts)
