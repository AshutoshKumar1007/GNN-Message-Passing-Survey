"""Temporal training loop that streams snapshots through the model.

Snapshots are extracted from `edge_attr` if available (Elliptic) or by
splitting edges chronologically into a fixed number of bins. Within each
snapshot we update the model's memory (TGN-style) and evaluate on the
nodes whose ground-truth label belongs to that snapshot.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from utils.metrics import compute_metrics

from .full_batch import TrainingResult


def _split_snapshots(
    edge_index: torch.Tensor,
    edge_time: torch.Tensor | None,
    num_snapshots: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Bucket edges into chronologically-ordered snapshots."""
    if edge_time is None:
        # Fall back to equal-size chunks
        n_edges = edge_index.size(1)
        chunk = max(n_edges // num_snapshots, 1)
        snapshots = []
        for s in range(num_snapshots):
            start = s * chunk
            end = (s + 1) * chunk if s < num_snapshots - 1 else n_edges
            snap_ei = edge_index[:, start:end]
            snap_t = torch.arange(start, end, device=edge_index.device)
            snapshots.append((snap_ei, snap_t.float()))
        return snapshots

    edge_time = edge_time.float().view(-1)
    qs = torch.linspace(0, 1, num_snapshots + 1, device=edge_time.device)
    cuts = torch.quantile(edge_time, qs)
    snapshots = []
    for i in range(num_snapshots):
        lo, hi = cuts[i], cuts[i + 1]
        mask = (edge_time >= lo) & (edge_time <= hi if i == num_snapshots - 1 else edge_time < hi)
        snapshots.append((edge_index[:, mask], edge_time[mask]))
    return snapshots


def train_temporal(
    model: nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    *,
    epochs: int = 30,
    lr: float = 0.005,
    weight_decay: float = 5e-4,
    patience: int = 5,
    num_snapshots: int = 10,
    edge_time: torch.Tensor | None = None,
) -> TrainingResult:
    """Train a TGN-style model on chronologically ordered snapshots."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    snapshots = _split_snapshots(data.edge_index, edge_time, num_snapshots)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    for epoch in range(epochs):
        if hasattr(model, "reset_memory"):
            model.reset_memory()

        model.train()
        ep_losses = []
        # Sequentially absorb earlier snapshots into the memory; train on each.
        for snap_ei, snap_t in snapshots:
            optimizer.zero_grad()
            logits = model(
                data.x,
                snap_ei,
                edge_time=snap_t,
                update_memory=True,
            )
            mask = train_mask
            if mask.sum() == 0:
                continue
            loss = F.cross_entropy(logits[mask], data.y[mask])
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())

        train_loss = float(sum(ep_losses) / max(len(ep_losses), 1))

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(logits[val_mask], data.y[val_mask]).item()
            val_acc = (
                (logits[val_mask].argmax(dim=1) == data.y[val_mask]).float().mean().item()
            )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

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
        logits, embeddings = model(
            data.x, data.edge_index, return_embedding=True
        )
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
