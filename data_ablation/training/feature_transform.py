"""Feature ablations: full / none / random / top-k.

The transforms operate on a copy of the input tensor so the underlying
`Data` object stays immutable - critical when the same dataset is reused
across many experiment runs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from utils.config import FeatureSelector, FeatureSetting


def _row_normalise(x: torch.Tensor) -> torch.Tensor:
    rs = x.sum(dim=1, keepdim=True).clamp(min=1.0)
    return x / rs


def _identity_features(num_nodes: int, dim: int = 64, *, seed: int = 0) -> torch.Tensor:
    """Replace X with a small random projection of the identity matrix.

    A pure one-hot identity has dim = num_nodes which OOMs Yelp / Elliptic.
    A fixed Gaussian projection preserves structural information while
    keeping dim = `dim`.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_nodes, dim, generator=g)


def _random_features(num_nodes: int, dim: int, *, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_nodes, dim, generator=g)


# ---------------------------------------------------------------------------
# Feature selection helpers
# ---------------------------------------------------------------------------


def select_topk(
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    k: int,
    method: FeatureSelector,
    seed: int = 0,
) -> torch.Tensor:
    """Return indices of the top-k features picked by `method`.

    Selection only uses the training subset to avoid label leakage.
    """
    if k <= 0 or k >= x.size(1):
        return torch.arange(x.size(1))

    x_train = x[train_mask].cpu().numpy()
    y_train = y[train_mask].cpu().numpy()

    if method == FeatureSelector.VARIANCE:
        scores = x_train.var(axis=0)
    elif method == FeatureSelector.CORRELATION:
        scores = _correlation_scores(x_train, y_train)
    elif method == FeatureSelector.RANDOM_FOREST:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            n_jobs=-1,
            random_state=seed,
        )
        rf.fit(x_train, y_train)
        scores = rf.feature_importances_
    else:
        raise ValueError(f"Unknown feature selector {method}")

    top_idx = np.argsort(-scores)[:k]
    return torch.tensor(np.sort(top_idx), dtype=torch.long)


def _correlation_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean absolute Pearson correlation across one-vs-rest binarisations."""
    classes = np.unique(y)
    scores = np.zeros(x.shape[1])
    for c in classes:
        target = (y == c).astype(float)
        if target.std() == 0:
            continue
        target_centered = target - target.mean()
        x_centered = x - x.mean(axis=0, keepdims=True)
        denom = (
            np.sqrt((x_centered ** 2).sum(axis=0))
            * np.sqrt((target_centered ** 2).sum())
            + 1e-12
        )
        corr = (x_centered * target_centered[:, None]).sum(axis=0) / denom
        scores += np.abs(corr)
    return scores / max(len(classes), 1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_feature_setting(
    x: torch.Tensor,
    *,
    setting: FeatureSetting,
    y: torch.Tensor | None = None,
    train_mask: torch.Tensor | None = None,
    selector: FeatureSelector | None = None,
    topk: int | None = None,
    seed: int = 0,
    drop_features: Sequence[int] | None = None,
    no_feature_dim: int = 64,
) -> torch.Tensor:
    """Return a (possibly new) feature tensor for the given ablation setting."""
    x_clone = x.clone()
    if drop_features is not None:
        keep = [i for i in range(x_clone.size(1)) if i not in set(drop_features)]
        x_clone = x_clone[:, keep]

    if setting == FeatureSetting.FULL:
        return _row_normalise(x_clone) if x_clone.dtype.is_floating_point else x_clone

    if setting == FeatureSetting.NONE:
        return _identity_features(x_clone.size(0), no_feature_dim, seed=seed)

    if setting == FeatureSetting.RANDOM:
        return _random_features(x_clone.size(0), x_clone.size(1), seed=seed)

    if setting == FeatureSetting.TOPK:
        if y is None or train_mask is None or selector is None or topk is None:
            raise ValueError("TOPK requires labels, train_mask, selector, and topk")
        idx = select_topk(x_clone, y, train_mask, k=topk, method=selector, seed=seed)
        sub = x_clone[:, idx]
        return _row_normalise(sub) if sub.dtype.is_floating_point else sub

    raise ValueError(f"Unsupported feature setting {setting}")
