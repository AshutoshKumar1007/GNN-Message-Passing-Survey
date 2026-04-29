"""Plotting utilities. All plots write to disk and never call `plt.show()`."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve


# ---------------------------------------------------------------------------
def plot_embedding_tsne(
    embeddings: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    *,
    out_path: Path | str,
    title: str = "t-SNE embedding",
    seed: int = 42,
    max_points: int = 5000,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    emb = embeddings.detach().cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.asarray(embeddings)
    lbl = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)

    if len(emb) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(emb), max_points, replace=False)
        emb, lbl = emb[idx], lbl[idx]

    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed)
    coords = tsne.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=lbl, cmap="tab20", s=10, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_feature_setting_comparison(
    df: pd.DataFrame,
    *,
    out_path: Path | str,
    metric: str = "test_accuracy",
) -> None:
    """Grouped bar chart of mean test-metric per (dataset, feature_setting)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = (
        df.groupby(["dataset", "feature_setting"])[metric]
        .mean()
        .unstack("feature_setting")
    )
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(pivot)), 5))
    pivot.plot.bar(ax=ax, edgecolor="black")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xlabel("dataset")
    ax.set_title(f"Feature setting comparison — {metric}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_homophily_vs_performance(
    df: pd.DataFrame,
    *,
    out_path: Path | str,
    metric: str = "test_accuracy",
    homophily_metric: str = "homophily_edge_homophily",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sub = df[df["feature_setting"] == "full"].copy()
    grouped = sub.groupby(["dataset", "model"]).agg(
        {metric: "mean", homophily_metric: "mean"}
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, df_m in grouped.groupby("model"):
        ax.scatter(
            df_m[homophily_metric],
            df_m[metric],
            label=model_name,
            s=80,
            alpha=0.8,
        )
        for _, row in df_m.iterrows():
            ax.annotate(
                row["dataset"],
                (row[homophily_metric], row[metric]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xlabel(homophily_metric)
    ax.set_ylabel(metric)
    ax.set_title("Homophily vs performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_pr_curves(
    runs: Iterable[tuple[str, torch.Tensor, torch.Tensor]],
    *,
    out_path: Path | str,
) -> None:
    """`runs` is an iterable of (label, logits, labels). Multi-class collapses
    to one-vs-rest macro-averaged PR."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, logits, y in runs:
        logits_np = logits.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        n_classes = logits_np.shape[1]
        precisions, recalls = [], []
        for c in range(n_classes):
            target = (y_np == c).astype(int)
            if target.sum() == 0:
                continue
            score = logits_np[:, c]
            p, r, _ = precision_recall_curve(target, score)
            precisions.append(p)
            recalls.append(r)
        if not precisions:
            continue
        max_len = max(len(p) for p in precisions)
        p_avg = np.mean(
            [np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(p)), p) for p in precisions],
            axis=0,
        )
        r_avg = np.linspace(0, 1, max_len)
        ax.plot(r_avg, p_avg, label=name, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Macro-averaged Precision–Recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_feature_ablation_curves(
    df: pd.DataFrame,
    *,
    out_path: Path | str,
    metric: str = "test_accuracy",
) -> None:
    """Plots metric vs k for every (dataset, model, selector) combination."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sub = df[df["feature_setting"] == "topk"].copy()
    if sub.empty:
        return
    sub["topk"] = sub["topk"].astype(int)
    grouped = sub.groupby(["dataset", "model", "feature_selector", "topk"])[metric].mean().reset_index()

    datasets = grouped["dataset"].unique()
    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=1,
        figsize=(7, 3.5 * len(datasets)),
        sharex=True,
    )
    if len(datasets) == 1:
        axes = [axes]
    for ax, dset in zip(axes, datasets):
        for (model_name, sel), g in grouped[grouped["dataset"] == dset].groupby(
            ["model", "feature_selector"]
        ):
            ax.plot(g["topk"], g[metric], marker="o", label=f"{model_name}/{sel}")
        ax.set_title(dset)
        ax.set_ylabel(metric)
        ax.legend(fontsize=7, loc="best")
    axes[-1].set_xlabel("top-k features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
