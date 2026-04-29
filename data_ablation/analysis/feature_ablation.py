"""Leave-one-out feature ablation: drop each feature and measure performance drop.

Used as a complement to the four-way feature_setting study. Operates per
(dataset, model) and writes the ranked drop list to disk.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from training.feature_transform import apply_feature_setting
from utils.config import FeatureSetting, RunConfig

from experiments.runner import ExperimentRunner


def run_feature_drop_ablation(
    runner: ExperimentRunner,
    base_cfg: RunConfig,
    *,
    feature_indices: Sequence[int] | None = None,
    out_csv: Path | str | None = None,
) -> pd.DataFrame:
    """Drop each feature in `feature_indices` (default: all) and report metrics.

    Beware: O(num_features * train_time). Pass `feature_indices` to a small
    subset (e.g. top-32 by variance) for tractable runs.
    """
    # Baseline run
    baseline = runner.run(base_cfg)
    baseline_acc = baseline.test_metrics["accuracy"]

    # Materialise feature indices
    bundle = runner_data(runner, base_cfg.dataset)
    if feature_indices is None:
        feature_indices = list(range(bundle.num_features))

    rows: list[dict] = [
        {
            "dropped": -1,
            "test_accuracy": baseline_acc,
            "delta_vs_baseline": 0.0,
        }
    ]

    for fi in feature_indices:
        cfg = replace(base_cfg, feature_setting=FeatureSetting.FULL)
        # `cfg.extra` is reserved for model kwargs; drop_features is applied via
        # the temporary patch below so it never leaks into model construction.
        rec = _run_with_drop(runner, cfg, drop=[fi])
        rows.append(
            {
                "dropped": fi,
                "test_accuracy": rec.test_metrics["accuracy"],
                "delta_vs_baseline": rec.test_metrics["accuracy"] - baseline_acc,
            }
        )

    df = pd.DataFrame(rows).sort_values("delta_vs_baseline")
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def runner_data(runner: ExperimentRunner, dataset_name: str):
    from datasets.registry import load_dataset

    return load_dataset(dataset_name, root=runner.data_root)


def _run_with_drop(
    runner: ExperimentRunner, cfg: RunConfig, *, drop: Sequence[int]
):
    """Patches the in-process feature transform with `drop_features`.

    `experiments.runner` imports `apply_feature_setting` by name, so we have
    to rebind the symbol on that module (not on `training.feature_transform`).
    """
    import experiments.runner as runner_mod

    original = runner_mod.apply_feature_setting

    def patched(x: torch.Tensor, **kw):
        return original(x, **{**kw, "drop_features": drop})

    runner_mod.apply_feature_setting = patched
    try:
        return runner.run(cfg)
    finally:
        runner_mod.apply_feature_setting = original
