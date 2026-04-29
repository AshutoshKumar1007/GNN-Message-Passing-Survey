"""Download / cache the Elliptic Bitcoin dataset via PyG.

Usage:
    python scripts/download_elliptic.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main():
    try:
        from torch_geometric.datasets import EllipticBitcoinDataset
    except ImportError:
        print("ERROR: torch_geometric is not installed. See README.md.", file=sys.stderr)
        sys.exit(1)

    root = Path("data/elliptic")
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading / caching Elliptic into {root} ...")
    ds = EllipticBitcoinDataset(root=str(root))
    data = ds[0]
    print(f"ok. {data}")


if __name__ == "__main__":
    main()
