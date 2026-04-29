"""Full-batch training loop with early stopping on validation loss."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from utils.metrics import MetricBundle, compute_metrics


@dataclass
class TrainingResult:
    test_metrics: MetricBundle
    val_metrics: MetricBundle
    best_epoch: int
    history: dict[str, list[float]]
    best_state: dict[str, torch.Tensor]
    embeddings: torch.Tensor | None = None
    test_logits: torch.Tensor | None = None


def _forward(model: nn.Module, data: Data, *, return_embedding: bool = False):
    return model(
        data.x,
        data.edge_index,
        return_embedding=return_embedding,
    )


def train_full_batch(
    model: nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    *,
    epochs: int = 500,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 100,
    log_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainingResult:
    """Run the canonical 'GNN training' loop and return rich diagnostics."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = _forward(model, data)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = _forward(model, data)
            val_loss = F.cross_entropy(val_logits[val_mask], data.y[val_mask]).item()
            val_acc = (
                (val_logits[val_mask].argmax(dim=1) == data.y[val_mask]).float().mean().item()
            )

        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        if log_callback is not None and epoch % 25 == 0:
            log_callback(
                epoch,
                {"train_loss": loss.item(), "val_loss": val_loss, "val_acc": val_acc},
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

    model.eval()
    with torch.no_grad():
        logits, embeddings = _forward(model, data, return_embedding=True)

    test_metrics = compute_metrics(logits[test_mask], data.y[test_mask])
    val_metrics = compute_metrics(logits[val_mask], data.y[val_mask])

    return TrainingResult(
        test_metrics=test_metrics,
        val_metrics=val_metrics,
        best_epoch=best_epoch,
        history=history,
        best_state=best_state or model.state_dict(),
        embeddings=embeddings.detach().cpu(),
        test_logits=logits[test_mask].detach().cpu(),
    )
