"""Feature ablation grid: full / none / random / top-k × selectors × seeds."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from utils.config import FeatureSelector, FeatureSetting, RunConfig

from .runner import ExperimentRunner, RunRecord


def build_ablation_grid(
    *,
    models: Sequence[str],
    datasets: Sequence[str],
    seeds: Sequence[int],
    feature_settings: Sequence[FeatureSetting] | None = None,
    selectors: Sequence[FeatureSelector] | None = None,
    topk_values: Sequence[int] | None = None,
    base_cfg_factory=None,
) -> list[RunConfig]:
    """Cartesian product of all knobs into a flat list of `RunConfig`s."""

    if feature_settings is None:
        feature_settings = list(FeatureSetting)
    if selectors is None:
        selectors = list(FeatureSelector)
    if topk_values is None:
        topk_values = [16, 64, 256]

    cfgs: list[RunConfig] = []
    for model, dataset, seed in product(models, datasets, seeds):
        base = (
            base_cfg_factory(model, dataset)
            if base_cfg_factory is not None
            else RunConfig(model=model, dataset=dataset, seed=seed)
        )
        base = replace(base, model=model, dataset=dataset, seed=seed)

        for setting in feature_settings:
            if setting != FeatureSetting.TOPK:
                cfgs.append(replace(base, feature_setting=setting))
                continue
            for selector, k in product(selectors, topk_values):
                cfgs.append(
                    replace(
                        base,
                        feature_setting=FeatureSetting.TOPK,
                        feature_selector=selector,
                        topk=k,
                    )
                )
    return cfgs


def run_ablation_grid(
    runner: ExperimentRunner,
    cfgs: Iterable[RunConfig],
    *,
    csv_path: Path | str | None = None,
    skip_existing: bool = False,
) -> pd.DataFrame:
    """Execute every config and return a DataFrame of flattened records."""
    rows: list[dict] = []
    csv_path = Path(csv_path) if csv_path else None

    if csv_path and skip_existing and csv_path.exists():
        prior = pd.read_csv(csv_path)
        rows.extend(prior.to_dict("records"))
        already = {
            (r["dataset"], r["model"], r["feature_setting"], r["feature_selector"], r["topk"], r["seed"])
            for r in rows
        }
    else:
        already = set()

    for cfg in cfgs:
        sig = (
            cfg.dataset,
            cfg.model,
            cfg.feature_setting.value,
            cfg.feature_selector.value if cfg.feature_selector else None,
            cfg.topk,
            cfg.seed,
        )
        if sig in already:
            continue
        record = runner.run(cfg)
        rows.append(record.flatten())

        if csv_path:
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    if csv_path:
        df.to_csv(csv_path, index=False)
    return df
