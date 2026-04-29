# Temporal vs Static GNNs for Anomaly Detection

Research-quality comparative study of **Temporal Graph Networks (TGN)** against strong
**static GNN** and **feature-only** baselines on two anomaly-detection datasets:

- **Elliptic Bitcoin** — temporal transaction graph (illicit vs. licit)
- **YelpChi** — static review graph (fraudulent vs. benign reviews)

Designed to run on a single GPU with **~4GB VRAM** using neighbor sampling and
temporal mini-batching. No full-graph training on the large datasets.

---

## 1. Project layout

```
graph_anomaly_detection/
├── README.md                 <- this file
├── requirements.txt
├── run_all.sh                <- one-shot: runs every model x dataset, writes results.csv
├── configs/
│   ├── elliptic.yaml
│   └── yelp.yaml
├── scripts/
│   ├── download_elliptic.py  <- uses PyG's EllipticBitcoinDataset (auto-download)
│   └── download_yelp.py      <- fetches YelpChi .mat (CARE-GNN format)
├── src/
│   ├── utils.py              <- seeding, metrics, EarlyStopping, logging
│   ├── data/
│   │   ├── elliptic.py       <- temporal split + event stream (u,v,t)
│   │   └── yelp.py           <- YelpChi -> PyG Data (homogenized from multi-relation)
│   ├── models/
│   │   ├── mlp.py            <- feature-only MLP
│   │   ├── gcn.py            <- GCNConv x2
│   │   ├── sage.py           <- SAGEConv x2
│   │   ├── gat.py            <- GATv2Conv x2
│   │   ├── h2gcn.py          <- H2GCN (heterophily-aware)
│   │   ├── linkx.py          <- LINKX (heterophily-aware)
│   │   └── tgn.py            <- TGN: memory + message + GRU updater + GAT embedding
│   ├── train_static.py       <- mini-batched training for all static/MLP models
│   ├── train_tgn.py          <- streaming temporal training for TGN
│   └── run_experiments.py    <- orchestrator, fills reports/results.csv
└── reports/
    ├── results.csv           <- filled by run_experiments.py
    └── final_report.md       <- structured analysis (you fill numbers after running)
```

---

## 2. Environment

Target: Python 3.10+, CUDA 11.8 / 12.1 with ~4GB GPU (GTX 1650 / T500 / Colab T4 all fine).

```bash
python -m venv .venv && source .venv/bin/activate

# Install PyTorch for YOUR CUDA version first (see https://pytorch.org)
# Example for CUDA 12.1:
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Then the rest
pip install -r requirements.txt
```

YelpChi also supports DGL's `FraudYelpDataset` as a fallback loader if you install DGL;
the default path uses the `.mat` file from CARE-GNN so no DGL is required.

---

## 3. Download datasets

```bash
python scripts/download_elliptic.py   # ~500MB, extracts to data/elliptic/
python scripts/download_yelp.py       # ~100MB, writes data/yelp/YelpChi.mat
```

Elliptic uses `torch_geometric.datasets.EllipticBitcoinDataset` (auto-download, cached).
YelpChi is the canonical CARE-GNN release (Dou et al. 2020).

---

## 4. Run a single experiment

```bash
# Static models on Elliptic (temporal split, not random)
python -m src.train_static --dataset elliptic --model gcn    --config configs/elliptic.yaml
python -m src.train_static --dataset elliptic --model sage   --config configs/elliptic.yaml
python -m src.train_static --dataset elliptic --model gat    --config configs/elliptic.yaml
python -m src.train_static --dataset elliptic --model h2gcn  --config configs/elliptic.yaml
python -m src.train_static --dataset elliptic --model linkx  --config configs/elliptic.yaml
python -m src.train_static --dataset elliptic --model mlp    --config configs/elliptic.yaml

# Temporal TGN on Elliptic
python -m src.train_tgn    --dataset elliptic --config configs/elliptic.yaml

# All of the above but on YelpChi (static split)
python -m src.train_static --dataset yelp --model gcn  --config configs/yelp.yaml
# ... etc
```

Each run prints the full `classification_report`, confusion matrix, and appends one row
to `reports/results.csv`.

---

## 5. Run everything & build the results table

```bash
bash run_all.sh                        # ~1-3 hours depending on GPU
python -m src.run_experiments --collect  # regenerates reports/results.csv from logs
```

---

## 6. What the report answers

See [reports/final_report.md](reports/final_report.md) for the structured analysis:

1. Dataset characteristics — class imbalance, sparsity, homophily ratio
2. Model comparison — AUC / F1-anom / Precision / Recall
3. Why some models win/lose
4. Temporal vs. static modeling
5. Homophilic vs. heterophily-aware models
6. Elliptic vs. YelpChi differences
7. **MLP vs. GNN** — does graph structure actually help?

---

## 7. Reproducibility

- All seeds fixed via `src.utils.set_seed(42)`
- `torch.use_deterministic_algorithms(True)` where possible
- Hyperparameters live in `configs/*.yaml` — no magic numbers in code
- Full training logs written under `runs/<dataset>_<model>_<timestamp>.log`

---

## 8. Known gotchas (read before debugging)

- **Elliptic temporal split is mandatory.** Never shuffle. Train t ≤ 34, val 35–39, test ≥ 40.
  The first 166 features are local, the rest are aggregated from neighbors — avoid using the
  aggregated features for MLP vs. GNN comparisons if you want a clean ablation
  (set `use_aggregated_features: false` in `configs/elliptic.yaml`).
- **Unknown nodes (class -1) must remain in the graph** for message passing but be
  **excluded from the loss**. Masks are handled in `src/data/elliptic.py`.
- **YelpChi has three relations (R-U-R, R-S-R, R-T-R).** The homogenized loader unions
  them. For a proper multi-relation comparison, swap in a heterogeneous model (not covered
  here by design — the spec asks for static-GNN baselines).
- **TGN on Elliptic** uses edge timestamps derived from node timestamps
  (`t_edge = max(t_u, t_v)`) since Elliptic's edge list has no native timestamp column.
  This is the standard convention in the Elliptic/TGN literature.

