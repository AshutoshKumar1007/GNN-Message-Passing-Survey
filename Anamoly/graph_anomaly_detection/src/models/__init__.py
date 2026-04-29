from .mlp import MLP
from .gcn import GCN
from .sage import GraphSAGE
from .gat import GAT
from .h2gcn import H2GCN
from .linkx import LINKX

STATIC_MODELS = {
    "mlp": MLP,
    "gcn": GCN,
    "sage": GraphSAGE,
    "gat": GAT,
    "h2gcn": H2GCN,
    "linkx": LINKX,
}

__all__ = ["MLP", "GCN", "GraphSAGE", "GAT", "H2GCN", "LINKX", "STATIC_MODELS"]
