"""Unified trainer for all static / feature-only models.

Usage
-----
    python -m src.train_static --dataset elliptic --model gcn \
        --config configs/elliptic.yaml

Supported models: mlp, gcn, sage, gat, h2gcn, linkx
Supported datasets: elliptic, yelp

Design notes
------------
* Mini-batched with `NeighborLoader` for 4GB-VRAM budgets.
* Strict temporal split for Elliptic (train t<=34, val 35..39, test >=40).
* Class imbalance handled via class-weighted cross-entropy.
* Eval: AUC, F1-anomaly, P/R, macro-F1, confusion matrix, full classification_report.
* Threshold searched on val set, locked for test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from .data.elliptic import load_elliptic, summarize as elliptic_summary
from .data.yelp import load_yelp, summarize as yelp_summary
from .models import STATIC_MODELS
from .utils import (
    EarlyStopping,
    append_result,
    class_weights_from_labels,
    compute_metrics,
    edge_homophily,
    get_logger,
    load_config,
    pick_device,
    set_seed,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(name: str, cfg: dict):
    if name == "elliptic":
        b = load_elliptic(
            root=cfg["dataset"]["root"],
            use_aggregated_features=cfg["dataset"].get("use_aggregated_features", True),
            train_max_t=cfg["dataset"]["temporal_split"]["train_max_t"],
            val_max_t=cfg["dataset"]["temporal_split"]["val_max_t"],
        )
        return b, elliptic_summary(b)
    if name == "yelp":
        b = load_yelp(
            root=cfg["dataset"]["root"],
            relations_mode=cfg["dataset"].get("relations_mode", "union"),
            train_ratio=cfg["dataset"].get("train_ratio", 0.4),
            val_ratio=cfg["dataset"].get("val_ratio", 0.2),
            stratified=cfg["dataset"].get("stratified", True),
            seed=cfg.get("seed", 42),
        )
        return b, yelp_summary(b)
    raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Loader construction
# ---------------------------------------------------------------------------

def make_loader(data: Data, node_mask: torch.Tensor, cfg: dict, shuffle: bool) -> NeighborLoader:
    return NeighborLoader(
        data,
        num_neighbors=cfg["loader"]["num_neighbors"],
        batch_size=cfg["loader"]["batch_size"],
        input_nodes=node_mask,
        shuffle=shuffle,
        num_workers=cfg["loader"].get("num_workers", 0),
    )


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(model_name: str, in_dim: int, num_classes: int, cfg: dict) -> torch.nn.Module:
    if model_name not in STATIC_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    Model = STATIC_MODELS[model_name]
    mcfg = cfg["models"][model_name]

    common = dict(in_dim=in_dim, num_classes=num_classes, hidden=mcfg["hidden"],
                  dropout=mcfg.get("dropout", 0.3))

    if model_name == "mlp":
        return Model(**common, num_layers=mcfg.get("num_layers", 3))
    if model_name == "gcn":
        return Model(**common, num_layers=mcfg.get("num_layers", 2))
    if model_name == "sage":
        return Model(**common, num_layers=mcfg.get("num_layers", 2),
                     aggr=mcfg.get("aggr", "mean"))
    if model_name == "gat":
        return Model(**common, num_layers=mcfg.get("num_layers", 2),
                     heads=mcfg.get("heads", 4))
    if model_name == "h2gcn":
        return Model(**common, k_hops=mcfg.get("k_hops", 2))
    if model_name == "linkx":
        return Model(**common, num_layers=mcfg.get("num_layers", 1),
                     inner_mlp_layers=mcfg.get("inner_mlp_layers", 1))
    raise ValueError(f"Unhandled model: {model_name}")


# ---------------------------------------------------------------------------
# Training / eval loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, class_weight, device, grad_clip=None) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        # NeighborLoader places the "seed" nodes first: [:batch_size].
        seed = batch.batch_size
        logits = out[:seed]
        y = batch.y[:seed].long()
        # Mask out any unknown labels (Elliptic-only concern, but safe for Yelp too).
        valid = y >= 0
        if valid.sum() == 0:
            continue
        loss = F.cross_entropy(logits[valid], y[valid], weight=class_weight)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * int(valid.sum().item())
        total_n += int(valid.sum().item())
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device, positive_class: int = 1):
    model.eval()
    ys, ss = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        seed = batch.batch_size
        logits = out[:seed]
        y = batch.y[:seed].long()
        valid = y >= 0
        if valid.sum() == 0:
            continue
        prob = F.softmax(logits[valid], dim=-1)[:, positive_class]
        ys.append(y[valid].cpu().numpy())
        ss.append(prob.cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.array([])
    y_score = np.concatenate(ss) if ss else np.array([])
    return y_true, y_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["elliptic", "yelp"], required=True)
    parser.add_argument("--model", choices=list(STATIC_MODELS.keys()), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override cfg.train.epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = pick_device(cfg.get("device", "cuda"))

    log = get_logger(f"{args.dataset}_{args.model}", cfg["logging"]["runs_dir"])
    log.info(f"device={device}  model={args.model}  dataset={args.dataset}")

    bundle, stats = load_dataset(args.dataset, cfg)
    log.info(f"dataset stats: {stats}")

    # Homophily diagnostic (single line in the report)
    h = edge_homophily(bundle.data.edge_index, bundle.data.y.cpu(),
                       bundle.data.labeled_mask.cpu() if hasattr(bundle.data, "labeled_mask")
                       else torch.ones(bundle.data.x.size(0), dtype=torch.bool))
    log.info(f"edge homophily (labeled endpoints): {h:.4f}")

    data = bundle.data
    # If using aggregated features for Elliptic, we still want to fit everything in VRAM.
    # NeighborLoader handles this — we stream per-batch.

    train_loader = make_loader(data, bundle.train_mask, cfg, shuffle=True)
    val_loader = make_loader(data, bundle.val_mask, cfg, shuffle=False)
    test_loader = make_loader(data, bundle.test_mask, cfg, shuffle=False)

    # Class weights from TRAIN labels only (no leakage)
    train_y = data.y[bundle.train_mask].cpu().numpy()
    cw = class_weights_from_labels(train_y, cfg["train"].get("class_weight", "balanced")).to(device)
    log.info(f"class_weight: {cw.tolist()}")

    model = build_model(args.model, in_dim=bundle.num_features,
                        num_classes=bundle.num_classes, cfg=cfg).to(device)
    log.info(f"model:\n{model}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=cfg["train"].get("scheduler_factor", 0.5),
        patience=cfg["train"].get("scheduler_patience", 4),
    )
    stopper = EarlyStopping(
        patience=cfg["train"]["patience"],
        mode="max",
    )

    num_epochs = args.epochs or cfg["train"]["epochs"]
    for ep in range(1, num_epochs + 1):
        tl = train_epoch(model, train_loader, optimizer, cw, device,
                         grad_clip=cfg["train"].get("grad_clip"))
        y_val, s_val = evaluate(model, val_loader, device, bundle.positive_class)
        val_metrics = compute_metrics(
            y_val, s_val,
            search_threshold=cfg["train"].get("threshold_search", True),
            pos_label=bundle.positive_class,
        )
        scheduler.step(val_metrics.f1_anomaly)
        is_best = stopper.step(val_metrics.f1_anomaly, model)
        log.info(
            f"ep {ep:03d}  loss {tl:.4f}  val AUC {val_metrics.roc_auc:.4f}  "
            f"val F1-anom {val_metrics.f1_anomaly:.4f}  thr {val_metrics.threshold:.3f}  "
            f"{'*' if is_best else ''}"
        )
        if stopper.should_stop:
            log.info(f"early stop @ epoch {ep}; best val F1-anom = {stopper.best:.4f}")
            break

    stopper.restore(model)

    # Final eval at val-best threshold
    y_val, s_val = evaluate(model, val_loader, device, bundle.positive_class)
    val_metrics = compute_metrics(
        y_val, s_val,
        search_threshold=cfg["train"].get("threshold_search", True),
        pos_label=bundle.positive_class,
    )
    locked_thr = val_metrics.threshold

    y_test, s_test = evaluate(model, test_loader, device, bundle.positive_class)
    test_metrics = compute_metrics(
        y_test, s_test,
        threshold=locked_thr,
        search_threshold=False,
        pos_label=bundle.positive_class,
    )
    log.info("=" * 70)
    log.info(f"FINAL  dataset={args.dataset}  model={args.model}")
    log.info(f"val  : AUC {val_metrics.roc_auc:.4f}  F1-anom {val_metrics.f1_anomaly:.4f}  "
             f"P {val_metrics.precision_anomaly:.4f}  R {val_metrics.recall_anomaly:.4f}")
    log.info(f"test : AUC {test_metrics.roc_auc:.4f}  F1-anom {test_metrics.f1_anomaly:.4f}  "
             f"P {test_metrics.precision_anomaly:.4f}  R {test_metrics.recall_anomaly:.4f}")
    log.info(f"confusion (test):\n{np.array(test_metrics.confusion)}")
    log.info(f"classification report (test):\n{test_metrics.report_text}")

    append_result(cfg["logging"]["results_csv"],
                  test_metrics.as_row(args.model, args.dataset, "test"))
    append_result(cfg["logging"]["results_csv"],
                  val_metrics.as_row(args.model, args.dataset, "val"))


if __name__ == "__main__":
    main()
