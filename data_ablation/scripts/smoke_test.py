"""Five-minute end-to-end smoke test.

Runs GCN + H2GCN on Cora (homophilic) and Texas (heterophilic) for two seeds
across the four feature settings, then writes a mini analysis report. If this
passes you can confidently scale to `--all`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.report import generate_report
from analysis.visualize import plot_feature_setting_comparison
from experiments.ablation import build_ablation_grid, run_ablation_grid
from experiments.runner import ExperimentRunner
from utils.config import FeatureSelector, FeatureSetting, RunConfig, TrainConfig


def main() -> None:
    train = TrainConfig(epochs=200, patience=40)

    def factory(model: str, dataset: str) -> RunConfig:
        return RunConfig(
            model=model,
            dataset=dataset,
            hidden_dim=64,
            dropout=0.5,
            num_layers=2,
            train=train,
        )

    cfgs = build_ablation_grid(
        models=["gcn", "h2gcn"],
        datasets=["cora", "texas"],
        seeds=[0, 1],
        feature_settings=[FeatureSetting.FULL, FeatureSetting.NONE,
                          FeatureSetting.RANDOM, FeatureSetting.TOPK],
        selectors=[FeatureSelector.VARIANCE],
        topk_values=[64],
        base_cfg_factory=factory,
    )

    runner = ExperimentRunner(output_root="reports/results_smoke")
    df = run_ablation_grid(runner, cfgs, csv_path="reports/results_smoke/grid.csv")
    print(df.head())

    plot_feature_setting_comparison(
        df, out_path="reports/plots_smoke/feature_setting.png"
    )
    print(generate_report("reports/results_smoke/grid.csv",
                          out_path="reports/analysis_smoke.txt"))


if __name__ == "__main__":
    main()
