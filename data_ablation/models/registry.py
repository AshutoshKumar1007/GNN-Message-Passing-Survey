"""Model factory: build a model from name + dataset metadata."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .gat import GAT
from .gcn import GCN
from .graphsage import GraphSAGE
from .h2gcn import H2GCN
from .linkx import LINKX
from .tgn import TGNNodeClassifier


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "gcn": GCN,
    "graphsage": GraphSAGE,
    "gat": GAT,
    "gatv2": GAT,
    "h2gcn": H2GCN,
    "linkx": LINKX,
    "tgn": TGNNodeClassifier,
}


def build_model(
    name: str,
    *,
    in_dim: int,
    hidden_dim: int,
    num_classes: int,
    num_nodes: int,
    num_layers: int = 2,
    dropout: float = 0.5,
    extra: dict[str, Any] | None = None,
) -> nn.Module:
    """Construct a model by name with sensible per-architecture defaults."""
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Known: {sorted(MODEL_REGISTRY.keys())}")
    extra = extra or {}

    common = dict(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
    )

    if key == "linkx":
        return LINKX(num_nodes=num_nodes, **common, **extra)
    if key == "tgn":
        return TGNNodeClassifier(num_nodes=num_nodes, **common, **extra)
    if key == "gatv2":
        return GAT(v2=True, **common, **extra)
    return MODEL_REGISTRY[key](**common, **extra)
