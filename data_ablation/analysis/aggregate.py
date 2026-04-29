"""Cross-run aggregation helpers (mean ± std, pivots, comparison tables)."""

from __future__ import annotations

import pandas as pd


def aggregate_runs(df: pd.DataFrame, metric: str = "test_accuracy") -> pd.DataFrame:
    """Return mean / std / count grouped by (dataset, model, feature_setting)."""
    grouped = df.groupby(
        ["dataset", "model", "feature_setting", "feature_selector", "topk"],
        dropna=False,
    )[metric]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
    return summary


def stability_report(df: pd.DataFrame, metric: str = "test_accuracy") -> pd.DataFrame:
    """Coefficient-of-variation across seeds: lower = more stable."""
    summary = aggregate_runs(df, metric=metric)
    cv_col = f"{metric}_cv"
    summary[cv_col] = (
        summary[f"{metric}_std"]
        / summary[f"{metric}_mean"].replace(0, float("nan"))
    )
    summary = summary.sort_values([cv_col], ascending=True)
    return summary


def feature_setting_report(
    df: pd.DataFrame, metric: str = "test_accuracy"
) -> pd.DataFrame:
    """Pivot summarising delta between FULL and other feature settings."""
    summary = aggregate_runs(df, metric=metric)
    base = summary[summary["feature_setting"] == "full"][
        ["dataset", "model", f"{metric}_mean"]
    ].rename(columns={f"{metric}_mean": "full_mean"})
    merged = summary.merge(base, on=["dataset", "model"], how="left")
    merged[f"delta_vs_full_{metric}"] = (
        merged[f"{metric}_mean"] - merged["full_mean"]
    )
    return merged
