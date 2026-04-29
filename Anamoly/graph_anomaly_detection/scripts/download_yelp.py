"""Download YelpChi.mat (CARE-GNN release).

Usage:
    python scripts/download_yelp.py
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Canonical mirror used by CARE-GNN / PC-GNN authors.
URL_CANDIDATES = [
    "https://github.com/YingtongDou/CARE-GNN/raw/master/data/YelpChi.zip",
    "https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/YelpChi.zip",
]


def main():
    out_dir = Path("data/yelp")
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "YelpChi.mat"
    if final.exists():
        print(f"already present: {final}")
        return

    zip_path = out_dir / "YelpChi.zip"
    for url in URL_CANDIDATES:
        try:
            print(f"downloading {url} ...")
            with urllib.request.urlopen(url, timeout=60) as r, open(zip_path, "wb") as f:
                shutil.copyfileobj(r, f)
            break
        except Exception as e:
            print(f"  failed: {e}")
    else:
        print("ERROR: could not download YelpChi.zip from any candidate URL.",
              file=sys.stderr)
        print("       Manually place YelpChi.mat at data/yelp/YelpChi.mat and re-run.",
              file=sys.stderr)
        sys.exit(1)

    print(f"extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    # The zip contains YelpChi.mat at the root; move it in case it's nested.
    for m in out_dir.rglob("YelpChi.mat"):
        if m != final:
            shutil.move(str(m), final)
        break
    zip_path.unlink(missing_ok=True)
    print(f"ok -> {final}")


if __name__ == "__main__":
    main()
