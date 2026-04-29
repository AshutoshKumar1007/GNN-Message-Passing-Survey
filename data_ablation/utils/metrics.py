"""Classification metrics that work for binary, multi-class, and imbalanced graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricBundle:
    """Bundle of metrics produced by `compute_metrics`."""

    accuracy: float
    f1_macro: float
    f1_micro: float
    precision_macro: float
    recall_macro: float
    auc: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_micro": self.f1_micro,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "auc": self.auc,
        }


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_auc(probs: np.ndarray, labels: np.ndarray, classes: Iterable[int]) -> float:
    """Return ROC-AUC; falls back to NaN when only one class present."""
    classes = sorted(set(classes))
    try:
        if len(classes) == 2:
            # Binary: use probability of the positive class
            if probs.ndim == 2 and probs.shape[1] == 2:
                pos = probs[:, 1]
            else:
                pos = probs.squeeze()
            return float(roc_auc_score(labels, pos))
        return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def compute_metrics(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> MetricBundle:
    """Compute the full metric bundle from raw logits."""
    logits_np = _to_numpy(logits)
    labels_np = _to_numpy(labels).astype(int)

    if logits_np.ndim == 1:
        # Already class predictions
        preds = logits_np.astype(int)
        probs = np.zeros((len(preds), int(preds.max()) + 1))
        probs[np.arange(len(preds)), preds] = 1.0
    else:
        probs = _softmax(logits_np)
        preds = probs.argmax(axis=1)

    classes = np.unique(labels_np).tolist()

    return MetricBundle(
        accuracy=float(accuracy_score(labels_np, preds)),
        f1_macro=float(f1_score(labels_np, preds, average="macro", zero_division=0)),
        f1_micro=float(f1_score(labels_np, preds, average="micro", zero_division=0)),
        precision_macro=float(
            precision_score(labels_np, preds, average="macro", zero_division=0)
        ),
        recall_macro=float(recall_score(labels_np, preds, average="macro", zero_division=0)),
        auc=_safe_auc(probs, labels_np, classes),
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)
