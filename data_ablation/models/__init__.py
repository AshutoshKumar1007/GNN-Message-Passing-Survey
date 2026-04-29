"""GNN model zoo. All models share a common `forward(x, edge_index, **kw)`
signature so the unified trainer can drive them interchangeably."""

from .gcn import GCN
from .graphsage import GraphSAGE
from .gat import GAT
from .h2gcn import H2GCN
from .linkx import LINKX
from .tgn import TGNNodeClassifier
from .registry import MODEL_REGISTRY, build_model

__all__ = [
    "GCN",
    "GraphSAGE",
    "GAT",
    "H2GCN",
    "LINKX",
    "TGNNodeClassifier",
    "MODEL_REGISTRY",
    "build_model",
]
