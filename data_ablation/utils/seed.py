"""Centralised RNG seeding for reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed every RNG in the stack.

    `deterministic=True` enforces cuBLAS / cuDNN deterministic kernels at the
    cost of throughput. Disable for very large runs where speed matters more
    than bit-for-bit reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUBLAS workspace must be configured for deterministic matmul
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            # Older PyTorch
            pass
    else:
        torch.backends.cudnn.benchmark = True
