"""Mini-batch training using `NeighborLoader` for large graphs.

Used automatically by the runner when the dataset size exceeds the
`batch_threshold`. Loss / metrics are accumulated over the loader so the
reported numbers stay comparable with full-batch training on smaller graphs.
"""

from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from utils.metrics import compute_metrics

from .full_batch import TrainingResult


def _eval(model: nn.Module, loader: NeighborLoader, device: torch.device) -> tuple[float, dict]:
    model.eval()
    losses, all_logits, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            target_idx = slice(0, batch.batch_size)
            loss = F.cross_entropy(logits[target_idx], batch.y[target_idx])
            losses.append(loss.item())
            all_logits.append(logits[target_idx].detach().cpu())
            all_labels.append(batch.y[target_idx].detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels)
    return float(sum(losses) / max(len(losses), 1)), {
        "metrics": metrics,
        "logits": logits,
        "labels": labels,
    }


def train_with_neighbor_sampling(
    model: nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    *,
    epochs: int = 100,
    lr: float = 0.005,
    weight_decay: float = 5e-4,
    patience: int = 20,
    batch_size: int = 1024,
    num_neighbors: tuple[int, ...] = (15, 10),
    device: torch.device | None = None,
    log_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainingResult:
    """Mini-batch trainer for graphs that don't fit in memory."""

    device = device or next(model.parameters()).device

    train_loader = NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),
        batch_size=batch_size,
        input_nodes=train_mask,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),
        batch_size=batch_size,
        input_nodes=val_mask,
        shuffle=False,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),
        batch_size=batch_size,
        input_nodes=test_mask,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        ep_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            target_idx = slice(0, batch.batch_size)
            loss = F.cross_entropy(logits[target_idx], batch.y[target_idx])
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())

        train_loss = float(sum(ep_losses) / max(len(ep_losses), 1))
        val_loss, val_pack = _eval(model, val_loader, device)
        val_acc = float(val_pack["metrics"].accuracy)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if log_callback is not None and epoch % 5 == 0:
            log_callback(
                epoch,
                {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_pack = _eval(model, test_loader, device)
    val_loss, val_pack = _eval(model, val_loader, device)

    return TrainingResult(
        test_metrics=test_pack["metrics"],
        val_metrics=val_pack["metrics"],
        best_epoch=best_epoch,
        history=history,
        best_state=best_state or model.state_dict(),
        embeddings=None,            # collected separately if needed
        test_logits=test_pack["logits"],
    )
