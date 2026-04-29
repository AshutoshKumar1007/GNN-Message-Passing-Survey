"""Aggregation, plotting, and report generation."""

from .aggregate import aggregate_runs, stability_report, feature_setting_report
from .visualize import (
    plot_embedding_tsne,
    plot_feature_setting_comparison,
    plot_homophily_vs_performance,
    plot_pr_curves,
    plot_feature_ablation_curves,
)
from .report import generate_report
from .feature_ablation import run_feature_drop_ablation

__all__ = [
    "aggregate_runs",
    "stability_report",
    "feature_setting_report",
    "plot_embedding_tsne",
    "plot_feature_setting_comparison",
    "plot_homophily_vs_performance",
    "plot_pr_curves",
    "plot_feature_ablation_curves",
    "generate_report",
    "run_feature_drop_ablation",
]
