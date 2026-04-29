"""Common utilities: seeding, metrics, early stopping, logging, config loading."""

from __future__ import annotations

import csv
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all RNGs for reproducibility. Deterministic algos enabled on best-effort basis."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Not all PyG ops are deterministic; this is best-effort.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_device(pref: str = "cuda") -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    roc_auc: float
    f1_anomaly: float
    precision_anomaly: float
    recall_anomaly: float
    macro_f1: float
    threshold: float
    confusion: list[list[int]]
    report_text: str

    def as_row(self, model: str, dataset: str, split: str) -> dict[str, Any]:
        return {
            "model": model,
            "dataset": dataset,
            "split": split,
            "roc_auc": round(self.roc_auc, 4),
            "f1_anomaly": round(self.f1_anomaly, 4),
            "precision": round(self.precision_anomaly, 4),
            "recall": round(self.recall_anomaly, 4),
            "macro_f1": round(self.macro_f1, 4),
            "threshold": round(self.threshold, 4),
        }


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: Optional[float] = None,
    search_threshold: bool = False,
    pos_label: int = 1,
) -> EvalResult:
    """Compute the full evaluation bundle.

    If `search_threshold`, sweeps thresholds on y_score to maximize F1 for the positive class.
    Otherwise uses `threshold` (default 0.5).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # ROC-AUC uses scores, not thresholded preds.
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    if search_threshold:
        # Sweep over the unique sorted scores (cheap for <100k eval points).
        # Use percentiles if the eval set is very large.
        if len(y_score) > 50_000:  # ==================== issue !
            cand = np.quantile(y_score, np.linspace(0.01, 0.99, 199))
        else:
            cand = np.unique(y_score)
            if len(cand) > 500:
                cand = np.quantile(y_score, np.linspace(0.01, 0.99, 199))
        best_f1, best_t = -1.0, 0.5
        for t in cand:
            pred = (y_score >= t).astype(int)
            f1 = f1_score(y_true, pred, pos_label=pos_label, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        threshold = best_t
    elif threshold is None:
        threshold = 0.5

    y_pred = (y_score >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[pos_label], zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred, digits=4, zero_division=0, target_names=["benign", "anomaly"]
    )
    return EvalResult(
        roc_auc=float(auc),
        f1_anomaly=float(f1[0]),
        precision_anomaly=float(prec[0]),
        recall_anomaly=float(rec[0]),
        macro_f1=float(macro_f1),
        threshold=float(threshold),
        confusion=cm,
        report_text=report,
    )


# ---------------------------------------------------------------------------
# Early stopping (tracks best val F1-anomaly; restores best state)
# ---------------------------------------------------------------------------

@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 1e-4
    mode: str = "max"       # we track val F1-anomaly -> maximize
    best: float = field(init=False, default=-float("inf"))
    counter: int = field(init=False, default=0)
    best_state: Optional[dict] = field(init=False, default=None)
    should_stop: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.mode == "min":
            self.best = float("inf")

    def step(self, value: float, model: torch.nn.Module) -> bool:
        """Returns True if this is a new best. Updates internal state."""
        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False

    def restore(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Class-weight helper
# ---------------------------------------------------------------------------

def class_weights_from_labels(y: np.ndarray, scheme: str | float = "balanced") -> torch.Tensor:
    """Compute per-class weights for CrossEntropyLoss."""
    y = np.asarray(y)
    if isinstance(scheme, (int, float)):
        # interpret as pos_weight for binary; weight the positive class
        return torch.tensor([1.0, float(scheme)], dtype=torch.float32)
    if scheme == "balanced":
        classes, counts = np.unique(y, return_counts=True)
        total = counts.sum()
        w = total / (len(classes) * counts.astype(float))
        out = torch.ones(int(classes.max()) + 1, dtype=torch.float32)
        for c, wc in zip(classes, w):
            out[int(c)] = float(wc)
        return out
    raise ValueError(f"Unknown class_weight scheme: {scheme}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, runs_dir: str = "runs", log_to_file: bool = True) -> logging.Logger:
    os.makedirs(runs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_to_file:
        ts = time.strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(runs_dir, f"{name}_{ts}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
 

# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------

RESULT_COLUMNS = [
    "model", "dataset", "split",
    "roc_auc", "f1_anomaly", "precision", "recall", "macro_f1", "threshold",
]


def append_result(csv_path: str, row: dict[str, Any]) -> None:
    """Append one row to results.csv, writing header if file doesn't exist."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})


# ---------------------------------------------------------------------------
# Homophily ratio (edge-level label agreement) — reported in the analysis
# ---------------------------------------------------------------------------

def edge_homophily(edge_index: torch.Tensor, y: torch.Tensor, labeled_mask: torch.Tensor) -> float:
    """Fraction of edges (u,v) where both endpoints are labeled and y[u] == y[v].

    Only edges with BOTH endpoints labeled are counted. This matches the
    standard homophily measure used in the heterophily literature (Zhu et al. 2020).
    """
    src, dst = edge_index
    mask = labeled_mask[src] & labeled_mask[dst]
    if mask.sum() == 0:
        return float("nan")
    same = (y[src[mask]] == y[dst[mask]]).float().mean().item()
    return float(same)
