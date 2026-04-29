"""Free-form text report: pulls insights out of the aggregated CSV and writes
`reports/analysis.txt` for the paper write-up."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from datasets.registry import DATASET_REGISTRY, DatasetCategory


def _category(dataset: str) -> str:
    if dataset not in DATASET_REGISTRY:
        return "unknown"
    return DATASET_REGISTRY[dataset][0].value


def generate_report(
    results_csv: Path | str,
    out_path: Path | str = "reports/analysis.txt",
    *,
    metric: str = "test_accuracy",
) -> str:
    df = pd.read_csv(results_csv)
    df["category"] = df["dataset"].apply(_category)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("GNN Research Report")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"# Total runs : {len(df)}")
    lines.append(f"# Datasets   : {df['dataset'].nunique()}")
    lines.append(f"# Models     : {df['model'].nunique()}")
    lines.append(f"# Seeds      : {df['seed'].nunique()}")
    lines.append("")

    # ---- (A) Homophily vs performance ----
    lines.append("─" * 72)
    lines.append("(A) Homophily vs performance — full features only")
    lines.append("─" * 72)
    full = df[df["feature_setting"] == "full"]
    if not full.empty:
        agg = full.groupby(["dataset", "model"]).agg(
            {metric: "mean", "homophily_edge_homophily": "mean"}
        ).reset_index()
        for model_name, df_m in agg.groupby("model"):
            corr = df_m[metric].corr(df_m["homophily_edge_homophily"])
            lines.append(f"  {model_name:<10s} corr(metric, homophily) = {corr:+.3f}")
    lines.append("")

    # ---- (B) Temporal vs static ----
    lines.append("─" * 72)
    lines.append("(B) Temporal vs static models on temporal datasets")
    lines.append("─" * 72)
    temporal = df[df["category"] == "temporal"]
    if not temporal.empty:
        for dataset, df_d in temporal.groupby("dataset"):
            lines.append(f"  {dataset}:")
            for model_name, df_m in df_d.groupby("model"):
                vals = df_m[metric].values
                lines.append(
                    f"    {model_name:<12s} {vals.mean():.4f} ± {vals.std(ddof=0):.4f}  (n={len(vals)})"
                )
    lines.append("")

    # ---- (C) Feature setting analysis ----
    lines.append("─" * 72)
    lines.append("(C) Feature importance: full vs none vs random vs top-k")
    lines.append("─" * 72)
    settings = df.groupby(["category", "feature_setting"])[metric].mean().unstack()
    if not settings.empty:
        lines.append(settings.to_string(float_format=lambda x: f"{x:.4f}"))
    lines.append("")

    # Per-dataset feature-vs-structure split
    lines.append("Per-dataset Δ(none − full) — negative = features needed:")
    pivot = df.groupby(["dataset", "feature_setting"])[metric].mean().unstack()
    if "none" in pivot.columns and "full" in pivot.columns:
        pivot["delta_none_minus_full"] = pivot["none"] - pivot["full"]
        for ds, val in pivot["delta_none_minus_full"].sort_values().items():
            verdict = "FEATURES dominate" if val < -0.05 else (
                "STRUCTURE suffices" if abs(val) <= 0.05 else "STRUCTURE > features"
            )
            lines.append(f"    {ds:<22s} Δ={val:+.4f}   → {verdict}")
    lines.append("")

    # ---- (D) Stability ----
    lines.append("─" * 72)
    lines.append("(D) Stability across seeds (CV = std / mean)")
    lines.append("─" * 72)
    stability = (
        df.groupby(["dataset", "model", "feature_setting"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stability["cv"] = stability["std"] / stability["mean"].replace(0, np.nan)
    most_stable = stability.sort_values("cv").head(10)
    least_stable = stability.sort_values("cv", ascending=False).head(10)
    lines.append("Most stable (lowest CV):")
    lines.append(most_stable.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")
    lines.append("Least stable (highest CV):")
    lines.append(least_stable.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")

    # ---- (E) Best (model × dataset × feature_setting) ----
    lines.append("─" * 72)
    lines.append("(E) Best configurations per dataset")
    lines.append("─" * 72)
    best = (
        df.groupby(["dataset", "model", "feature_setting"])[metric]
        .mean()
        .reset_index()
        .sort_values(metric, ascending=False)
    )
    for ds, df_ds in best.groupby("dataset"):
        top = df_ds.head(3)
        lines.append(f"  {ds}:")
        for _, row in top.iterrows():
            lines.append(
                f"    {row['model']:<10s} {row['feature_setting']:<8s} → {row[metric]:.4f}"
            )
    lines.append("")

    text = "\n".join(lines)
    out_path.write_text(text)
    return text
