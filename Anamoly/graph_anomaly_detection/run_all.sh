#!/usr/bin/env bash
# One-shot runner: trains every (dataset, model) pair and aggregates results.csv.
# Usage:  bash run_all.sh
#
# Individual runs stream stdout; also tees into runs/*.log.
# Expected wall time on a 4GB GPU (GTX 1650 / T4):
#   - Elliptic: ~5-10 min/model (MLP fastest, TGN ~15-25 min)
#   - YelpChi : ~5-15 min/model

set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# --- Elliptic (temporal + static) --------------------------------------------
# for m in mlp gcn sage gat h2gcn linkx; do
#     python -m src.train_static --dataset elliptic --model "$m" --config configs/elliptic.yaml
# done
python -m src.train_tgn --dataset elliptic --config configs/elliptic.yaml

# --- YelpChi (static only) ---------------------------------------------------
# for m in mlp gcn sage gat h2gcn linkx; do
#     python -m src.train_static --dataset yelp --model "$m" --config configs/yelp.yaml
# done

# for m in h2gcn linkx; do
#     python -m src.train_static --dataset yelp --model "$m" --config configs/yelp.yaml
# done


# --- Aggregate ---------------------------------------------------------------
echo
echo "=========  reports/results.csv  =========="
cat reports/results.csv
