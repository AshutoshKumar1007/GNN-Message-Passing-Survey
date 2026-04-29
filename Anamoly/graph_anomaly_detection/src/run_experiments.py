"""Orchestrator: runs every (dataset, model) pair and/or rebuilds the results table.

Examples
--------
    # Run everything (calls train_static.py / train_tgn.py in subprocesses)
    python -m src.run_experiments --run

    # Rebuild reports/results.csv from existing logs only
    python -m src.run_experiments --collect

    # Run a subset
    python -m src.run_experiments --run --datasets elliptic --models mlp gcn tgn
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from .utils import append_result, load_config

STATIC_MODELS_ORDER = ["mlp", "gcn", "sage", "gat", "h2gcn", "linkx"]


def run_one(dataset: str, model: str, config: str) -> int:
    """Subprocess-invoke the right trainer. Returns exit code."""
    if model == "tgn":
        cmd = [sys.executable, "-m", "src.train_tgn", "--dataset", dataset, "--config", config]
    else:
        cmd = [sys.executable, "-m", "src.train_static",
               "--dataset", dataset, "--model", model, "--config", config]
    print(">>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    return proc.returncode


def collect_from_logs(runs_dir: str, results_csv: str) -> int:
    """Parse FINAL lines from each per-run log and re-emit reports/results.csv.

    This is useful if results.csv gets out-of-date or corrupt. We look for the
    single 'FINAL' block each training script writes and pull AUC/F1/P/R from it.
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        print(f"No runs dir at {runs_dir}; nothing to collect.")
        return 0

    pattern = re.compile(
        r"FINAL\s+dataset=(?P<dataset>\S+)\s+model=(?P<model>\S+).*?"
        r"test\s*:\s*AUC\s+(?P<auc>[0-9.]+)\s+F1-anom\s+(?P<f1>[0-9.]+)\s+"
        r"P\s+(?P<p>[0-9.]+)\s+R\s+(?P<r>[0-9.]+)",
        re.DOTALL,
    )
    n = 0
    # Delete any existing csv to rebuild cleanly
    Path(results_csv).unlink(missing_ok=True)
    for f in sorted(runs_path.glob("*.log")):
        txt = f.read_text(errors="ignore")
        m = pattern.search(txt)
        if not m:
            continue
        row = {
            "model": m["model"],
            "dataset": m["dataset"],
            "split": "test",
            "roc_auc": float(m["auc"]),
            "f1_anomaly": float(m["f1"]),
            "precision": float(m["p"]),
            "recall": float(m["r"]),
            "macro_f1": "",
            "threshold": "",
        }
        append_result(results_csv, row)
        n += 1
    print(f"Collected {n} FINAL records -> {results_csv}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true", help="Run every (dataset, model) experiment.")
    p.add_argument("--collect", action="store_true", help="Rebuild results.csv from logs.")
    p.add_argument("--datasets", nargs="+", default=["elliptic", "yelp"])
    p.add_argument("--models", nargs="+",
                   default=STATIC_MODELS_ORDER + ["tgn"])
    p.add_argument("--elliptic-config", default="configs/elliptic.yaml")
    p.add_argument("--yelp-config", default="configs/yelp.yaml")
    args = p.parse_args()

    if not (args.run or args.collect):
        p.error("Pass --run and/or --collect")

    if args.run:
        for ds in args.datasets:
            cfg_path = args.elliptic_config if ds == "elliptic" else args.yelp_config
            for m in args.models:
                if m == "tgn" and ds != "elliptic":
                    print(f"Skipping TGN on {ds} (no timestamps).")
                    continue
                rc = run_one(ds, m, cfg_path)
                if rc != 0:
                    print(f"!! {ds}/{m} failed with code {rc}")

    if args.collect:
        # Use either config — the `logging` block is the same structurally.
        cfg = load_config(args.elliptic_config)
        collect_from_logs(cfg["logging"]["runs_dir"], cfg["logging"]["results_csv"])


if __name__ == "__main__":
    main()
