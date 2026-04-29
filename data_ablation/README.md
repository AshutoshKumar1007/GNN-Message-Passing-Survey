# GNN Research Pipeline — Refactored

A clean, modular, research-grade re-implementation of the original
`mini_project/` codebase. The pipeline answers four core research questions:

1. **Do node features matter more than graph structure?** Controlled comparison
   of *full / none / random / top-k* feature settings with three different
   selection methods (variance, label correlation, RandomForest importance).
2. **How do GCN, GraphSAGE, GAT, H2GCN, LINKX, and TGN compare** under a single,
   faithful training protocol?
3. **Homophilic vs heterophilic vs temporal** behaviour of every model, plotted
   against measured edge / node / class-insensitive homophily.
4. **Scalable training** via PyG `NeighborLoader` for large graphs, kept
   *automatically* equivalent to full-batch training on small ones.

---

## Layout

```
refactored/
├── models/           GCN, GraphSAGE, GAT, H2GCN, LINKX, TGN (PyG-based)
│   └── registry.py   build_model(name, …)
├── datasets/         Unified loader → DatasetBundle (cat = homo/hetero/temporal)
│   └── homophily.py  edge / node / class-insensitive homophily metrics
├── training/         Three loops: full_batch, sampled, temporal
│   └── feature_transform.py   FULL / NONE / RANDOM / TOPK ablations
├── experiments/      Runner + ablation-grid orchestration
├── analysis/         Aggregation, plots, t-SNE, text report
├── utils/            seed, logging, metrics, config dataclasses
├── reports/          plots/ + analysis.txt + per-run JSONs (created on run)
├── scripts/smoke_test.py
├── run_experiments.py   ← CLI entry point
└── requirements.txt
```

Every module has a one-purpose `__init__.py` re-exporting public API only.

---

## Quick start

```bash
# install (use the wheels that match your CUDA/torch version for torch-scatter etc.)
pip install -r requirements.txt

# 5-minute smoke test (Cora + Texas, 2 seeds, 4 feature settings)
python scripts/smoke_test.py

# small custom run
python run_experiments.py \
    --models gcn h2gcn linkx \
    --datasets cora texas chameleon \
    --feature-settings full none random topk \
    --selectors variance correlation random_forest \
    --topk-values 16 64 256 \
    --seeds 0 1 2

# full grid (all models × all datasets × all settings × 3 seeds)
python run_experiments.py --all

# regenerate plots and report from an existing CSV
python run_experiments.py --analyse-only --csv reports/results/grid.csv
```

Outputs:

- `reports/results/grid.csv` — flat CSV `dataset,model,feature_setting,…,test_*`.
- `reports/results/<dataset>/<model>/<run-id>.{config.json,metrics.json,pt}`.
- `reports/plots/{feature_setting_comparison,homophily_vs_performance,feature_ablation_curves}.png`.
- `reports/plots/tsne/<dataset>__<model>.png` per (dataset, model).
- `reports/analysis.txt` — narrative report (homophily corr, temporal vs static,
  feature dominance, stability, best configs).

---

## Feature ablation matrix

Per `(model, dataset, seed)` the runner produces:

| setting     | x replacement                                      |
|-------------|----------------------------------------------------|
| `full`      | original X (row-normalised if dense floats)        |
| `none`      | fixed Gaussian projection of identity (dim=64)     |
| `random`    | i.i.d. Gaussian, same shape as X                   |
| `topk`      | only top-k features picked on the *training* set   |

For `topk`, three selectors are run independently:

| selector       | score                                                              |
|----------------|--------------------------------------------------------------------|
| `variance`     | per-feature variance on the train slice                            |
| `correlation`  | mean abs Pearson(feature, one-vs-rest binarised label)             |
| `random_forest`| `RandomForestClassifier.feature_importances_`                      |

Plus a leave-one-out feature drop ablation in `analysis/feature_ablation.py`
that ranks individual features by the test-accuracy drop they cause.

---