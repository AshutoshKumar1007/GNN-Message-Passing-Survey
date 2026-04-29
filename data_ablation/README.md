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

## What was improved & why

| # | Issue in the original repo                                                            | Fix                                                                                 |
|---|---------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| 1 | Dense `n×n` adjacency in every model (OOM on Yelp / Elliptic)                         | All propagation operators use sparse PyG primitives or sparse SpMM                  |
| 2 | GraphSAGE hand-rolled and unstable                                                    | Replaced with PyG `SAGEConv`; standardised LR + dropout                             |
| 3 | Each model duplicated GCN code                                                        | Single `models/` directory + factory                                                |
| 4 | Hyper-parameters and seeds scattered across notebooks                                 | `RunConfig` dataclass, deterministic `set_seed(...)`                                |
| 5 | Mixed PyG / OGB / mat loaders with different return types                             | Unified `DatasetBundle` exposes `data, num_classes, num_features, num_splits, …`    |
| 6 | No quantitative analysis of "features vs structure"                                   | 4-way feature ablation × 3 selectors × N seeds, plus leave-one-out drop            |
| 7 | t-SNE only run interactively in notebooks                                             | Saved automatically per (model, dataset) in `reports/plots/tsne/`                   |
| 8 | Reports were `.txt` files written by hand                                             | `analysis/report.py` produces a structured `reports/analysis.txt`                   |
| 9 | No batching — large graphs simply crashed                                             | Auto-switch to `NeighborLoader` when `num_nodes > batch_threshold`                  |
| 10| Temporal model was a plain GCN labelled "TGN"                                         | Real TGN-style memory + temporal attention with snapshot training loop              |
| 11| `print` statements and ad-hoc seeding                                                 | `logging`, deterministic cuBLAS, structured JSON config dumps                       |
| 12| Test metrics = accuracy only                                                          | Accuracy + F1 (macro/micro) + Precision + Recall + ROC-AUC (binary or OVR macro)    |

---

## Insights you should expect to see

These come for free once the grid finishes:

- **Texas / Wisconsin / Actor** (heterophilic) — the `none` setting beats or
  matches `full` for vanilla GCN; H2GCN and LINKX recover the gap because their
  designs decouple feature and structure.
- **Cora / CiteSeer / PubMed** (homophilic) — `none` collapses to ~30 % accuracy
  while `full` retains 80 %+; structure alone is *not* enough.
- **Elliptic** — TGN beats static GNNs because illicit-account labels concentrate
  in specific time windows; static models trained without `edge_time` underfit
  the temporal pattern.
- **Top-k selectors** — RandomForest > correlation > variance on the homophilic
  suite; correlation often catches up on heterophilic graphs because RF
  overfits to noisy local features.
- **Stability** — H2GCN tends to have the lowest CV across seeds on
  heterophilic graphs; vanilla GCN has the highest CV on Texas / Wisconsin.

The exact numbers depend on the seeds and hardware; the grid produces the
mean ± std needed for a publication table.

---

## Running on a single machine without internet

`datasets/registry.py` only downloads the first time. Cached data goes to
`data/<dataset>/`. To pre-populate, run the smoke test once on a connected box
and then copy the `data/` directory.

---

## Testing the implementation correctness

The pipeline produces three sanity checks worth eyeballing before reporting
numbers:

1. `homophily_*` columns in `grid.csv` should match the published
   homophily ratios for Cora (~0.81) / Texas (~0.11) / Squirrel (~0.22).
2. `feature_setting=none + dataset=cora` should drop GCN to ~0.3 accuracy.
3. `feature_setting=full + dataset=texas + model=gcn` should *under*-perform
   `model=h2gcn`/`linkx` — that's the heterophily story.

If any of those break, the regression is in the data loader or the model,
not in the analysis.
