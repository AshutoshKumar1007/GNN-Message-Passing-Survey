"""Unified experiment runner CLI.

Usage examples
--------------

# Run the full grid (all models × datasets × feature settings × seeds 0-2)
python run_experiments.py --all

# Run a single (model × dataset) cell
python run_experiments.py --models gcn h2gcn --datasets cora texas --seeds 0 1

# Just feature ablation curves on the heterophilic suite
python run_experiments.py \
    --models h2gcn linkx \
    --datasets texas wisconsin actor \
    --feature-settings topk \
    --topk-values 16 64 256 \
    --selectors variance correlation random_forest

# Generate analysis from an existing CSV without re-running
python run_experiments.py --analyse-only --csv reports/results/grid.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from analysis.aggregate import (
    aggregate_runs,
    feature_setting_report,
    stability_report,
)
from analysis.report import generate_report
from analysis.visualize import (
    plot_embedding_tsne,
    plot_feature_ablation_curves,
    plot_feature_setting_comparison,
    plot_homophily_vs_performance,
)
from datasets.registry import DATASET_REGISTRY, DatasetCategory
from experiments.ablation import build_ablation_grid, run_ablation_grid
from experiments.runner import ExperimentRunner
from utils.config import FeatureSelector, FeatureSetting, RunConfig, TrainConfig
from utils.logging_utils import get_logger


DEFAULT_MODELS = ["gcn", "graphsage", "gat", "h2gcn", "linkx", "tgn"]
DEFAULT_HOMOPHILIC = ["cora", "citeseer", "pubmed"]
DEFAULT_HETEROPHILIC = ["texas", "wisconsin", "cornell", "chameleon", "squirrel", "actor"]
DEFAULT_TEMPORAL = ["elliptic"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument(
        
        "--feature-settings",
        nargs="+",
        choices=[s.value for s in FeatureSetting],
        default=None,
    )
    p.add_argument(
        "--selectors",
        nargs="+",
        choices=[s.value for s in FeatureSelector],
        default=None,
    )
    p.add_argument("--topk-values", nargs="+", type=int, default=None)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=0, help="0 → full-batch")
    p.add_argument("--data-root", default="data")
    p.add_argument("--output-root", default="reports/results")
    p.add_argument("--csv", default="reports/results/grid.csv")
    p.add_argument("--device", default=None)
    p.add_argument("--all", action="store_true", help="Run the full default grid")
    p.add_argument("--analyse-only", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args(argv)


def cfg_factory(args: argparse.Namespace):
    train = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        batch_size=args.batch_size,
    )

    def factory(model: str, dataset: str) -> RunConfig:
        return RunConfig(
            model=model,
            dataset=dataset,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            num_layers=args.num_layers,
            train=train,
            seed=0,
        )

    return factory


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logger = get_logger("cli", "reports/results/logs")

    if args.analyse_only:
        return _analyse(args, logger)

    if args.all:
        models = DEFAULT_MODELS
        datasets = DEFAULT_HOMOPHILIC + DEFAULT_HETEROPHILIC + DEFAULT_TEMPORAL
        feature_settings = list(FeatureSetting)
        selectors = list(FeatureSelector)
        topk_values = [16, 64, 256]
    else:
        models = args.models or DEFAULT_MODELS
        datasets = args.datasets or DEFAULT_HOMOPHILIC
        feature_settings = (
            [FeatureSetting(s) for s in args.feature_settings]
            if args.feature_settings
            else [FeatureSetting.FULL]
        )
        selectors = (
            [FeatureSelector(s) for s in args.selectors] if args.selectors else None
        )
        topk_values = args.topk_values

    cfgs = build_ablation_grid(
        models=models,
        datasets=datasets,
        seeds=args.seeds,
        feature_settings=feature_settings,
        selectors=selectors,
        topk_values=topk_values,
        base_cfg_factory=cfg_factory(args),
    )
    logger.info("Total configs: %d", len(cfgs))

    runner = ExperimentRunner(
        data_root=args.data_root,
        output_root=args.output_root,
        device=args.device,
    )
    df = run_ablation_grid(
        runner,
        cfgs,
        csv_path=args.csv,
        skip_existing=args.skip_existing,
    )
    logger.info("Wrote %d rows to %s", len(df), args.csv)

    _analyse(args, logger)
    return 0


def _analyse(args: argparse.Namespace, logger) -> int:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        return 1
    df = pd.read_csv(csv_path)

    plot_dir = Path("reports/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_feature_setting_comparison(df, out_path=plot_dir / "feature_setting_comparison.png")
    plot_homophily_vs_performance(df, out_path=plot_dir / "homophily_vs_performance.png")
    plot_feature_ablation_curves(df, out_path=plot_dir / "feature_ablation_curves.png")

    aggregate_runs(df).to_csv(plot_dir / "aggregated.csv", index=False)
    stability_report(df).to_csv(plot_dir / "stability.csv", index=False)
    feature_setting_report(df).to_csv(plot_dir / "feature_setting_delta.csv", index=False)

    text = generate_report(csv_path, out_path="reports/analysis.txt")
    logger.info("Wrote analysis report:\n%s", text[:600])

    _emit_tsne(df, args, logger)
    return 0


def _emit_tsne(df: pd.DataFrame, args: argparse.Namespace, logger) -> None:
    """Pull saved embeddings and emit one t-SNE plot per (model, dataset)."""
    tsne_dir = Path("reports/plots/tsne")
    tsne_dir.mkdir(parents=True, exist_ok=True)

    for (dataset, model), _ in df[df["feature_setting"] == "full"].groupby(
        ["dataset", "model"]
    ):
        run_dir = Path(args.output_root) / dataset / model
        if not run_dir.exists():
            continue
        candidates = sorted(run_dir.glob("*_full_s*.pt"))
        if not candidates:
            continue
        payload = torch.load(candidates[0], map_location="cpu")
        emb = payload.get("embeddings")
        if emb is None:
            continue
        # Align labels via dataset reload (cheap; cached on disk)
        from datasets.registry import load_dataset

        bundle = load_dataset(dataset, root=args.data_root)
        labels = bundle.data.y[: emb.size(0)]
        plot_embedding_tsne(
            emb,
            labels,
            out_path=tsne_dir / f"{dataset}__{model}.png",
            title=f"t-SNE — {model} on {dataset}",
        )
        logger.info("Wrote t-SNE for %s/%s", dataset, model)


if __name__ == "__main__":
    raise SystemExit(main())
