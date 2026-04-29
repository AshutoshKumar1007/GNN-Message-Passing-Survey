"""TGN trainer — streaming temporal training on Elliptic's event stream.

Usage:
    python -m src.train_tgn --dataset elliptic --config configs/elliptic.yaml

Strictly temporal:
  * events with edge_t <= 34 -> train
  * 35..39 -> val
  * >=40   -> test
  * memory is reset at the start of each epoch and advanced only on TRAIN events;
    val/test events are consumed in order but labels are never fed back into
    the training gradient.

Node-classification setup:
  At each training batch of events, we sample recent neighbors for the
  involved nodes, compute their embeddings, predict on the LABELED subset
  of those nodes, and add a cross-entropy classification term to the loss.
  Unlabeled (class=-1 / 'unknown') nodes still produce messages + memory updates
  but do not contribute to the loss.
"""

from __future__ import annotations

import argparse
import yaml

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

from .data.elliptic import load_elliptic
from .models.tgn import TGN
from .utils import (
    EarlyStopping,
    append_result,
    class_weights_from_labels,
    compute_metrics,
    get_logger,
    load_config,
    pick_device,
    set_seed,
)


def _make_temporal_data(events, device) -> "TemporalData":
    from torch_geometric.data import TemporalData
    return TemporalData(
        src=events.src.to(device),
        dst=events.dst.to(device),
        t=events.t.to(device),
        msg=events.msg.to(device),
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["elliptic"], default="elliptic")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = pick_device(cfg.get("device", "cuda"))

    log = get_logger("elliptic_tgn", cfg["logging"]["runs_dir"])
    log.info('='*60)
    log.info("TGN CONFIG:\n" + yaml.dump(cfg["tgn"], sort_keys=False))
    log.info('='*60)
    log.info(f"device={device}  model=tgn  dataset={args.dataset}")

    # -- Data -----------------------------------------------------------------
    bundle = load_elliptic(
        root=cfg["dataset"]["root"],
        use_aggregated_features=cfg["dataset"].get("use_aggregated_features", True),
        train_max_t=cfg["dataset"]["temporal_split"]["train_max_t"],
        val_max_t=cfg["dataset"]["temporal_split"]["val_max_t"],
    )
    events = bundle.event_stream()
    n_nodes = events.node_feat.size(0)
    node_feat = events.node_feat.to(device)
    raw_msg_all = events.msg.to(device)
    edge_t_all = events.t.to(device)
    y = events.y.to(device)
    labeled_mask = events.labeled_mask.to(device)

    train_td = _make_temporal_data(
        type("E", (), {
            "src": events.src[events.train_event_mask],
            "dst": events.dst[events.train_event_mask],
            "t":   events.t[events.train_event_mask],
            "msg": events.msg[events.train_event_mask],
        })(),
        device,
    )
    val_td = _make_temporal_data(
        type("E", (), {
            "src": events.src[events.val_event_mask],
            "dst": events.dst[events.val_event_mask],
            "t":   events.t[events.val_event_mask],
            "msg": events.msg[events.val_event_mask],
        })(),
        device,
    )
    test_td = _make_temporal_data(
        type("E", (), {
            "src": events.src[events.test_event_mask],
            "dst": events.dst[events.test_event_mask],
            "t":   events.t[events.test_event_mask],
            "msg": events.msg[events.test_event_mask],
        })(),
        device,
    )

    bs = cfg["tgn"]["batch_size"]
    train_loader = TemporalDataLoader(train_td, batch_size=bs)
    val_loader = TemporalDataLoader(val_td, batch_size=bs)
    test_loader = TemporalDataLoader(test_td, batch_size=bs)

    # -- Model ----------------------------------------------------------------
    tgn = TGN(
        num_nodes=n_nodes,
        raw_msg_dim=events.msg.size(1),
        node_feat_dim=events.node_feat.size(1),
        memory_dim=cfg["tgn"]["memory_dim"],
        time_dim=cfg["tgn"]["time_dim"],
        embedding_dim=cfg["tgn"]["embedding_dim"],
        num_classes=bundle.num_classes,
        message_module=cfg["tgn"].get("message_module", "identity"),
        aggregator=cfg["tgn"].get("aggregator", "last"),
    ).to(device)
    log.info(f"tgn: memory_dim={cfg['tgn']['memory_dim']}  emb={cfg['tgn']['embedding_dim']}  "
             f"neighbor_size={cfg['tgn']['neighbor_size']}  bs={bs}")

    neighbor_loader = LastNeighborLoader(
        n_nodes, size=cfg["tgn"]["neighbor_size"], device=device,
    )

    # Class weights from labeled TRAIN nodes (those with t <= train_max_t AND labeled).
    train_node_mask = bundle.train_mask.cpu().numpy()
    train_labels = bundle.data.y[bundle.train_mask].cpu().numpy()
    cw = class_weights_from_labels(train_labels, cfg["train"].get("class_weight", "balanced")).to(device)
    log.info(f"class_weight: {cw.tolist()}")

    optimizer = torch.optim.Adam(tgn.parameters(), lr=cfg["tgn"]["lr"])
    stopper = EarlyStopping(patience=cfg["tgn"]["patience"], mode="max")

    num_epochs = args.epochs or cfg["tgn"]["epochs"]

    def run_split(loader, train: bool, update_memory: bool):
        """Iterate a split's event stream.

        Returns (mean_loss, y_true_np, y_score_np).
        """
        if train:
            tgn.train()
        else:
            tgn.eval()

        y_true, y_score = [], []
        total_loss, total_n = 0.0, 0

        for batch in loader:
            src = batch.src
            dst = batch.dst
            t = batch.t
            msg = batch.msg

            # Sample the union of sources & destinations + their recent neighbors
            n_id = torch.cat([src, dst]).unique()
            n_id_with_neigh, edge_index, e_id = neighbor_loader(n_id)

            # Embedding requires all involved nodes including neighbors
            if train:
                optimizer.zero_grad()

            z = tgn.compute_embedding(
                n_id=n_id_with_neigh,
                edge_index_block=edge_index,
                e_id_block=e_id,
                t_targets=t,
                node_feat=node_feat,
                edge_raw_msg=raw_msg_all,
                edge_t=edge_t_all,
            )

            # Build a map from global node id -> local index in z
            id2local = {int(nid.item()): i for i, nid in enumerate(n_id_with_neigh)}

            # Classify the target nodes (src + dst) if they are labeled
            targets = torch.cat([src, dst]).unique()
            local_idx = torch.tensor(
                [id2local[int(i.item())] for i in targets], device=device, dtype=torch.long
            )
            labels = y[targets]
            valid = (labels != -1)
            if valid.sum() > 0:
                logits = tgn.predict(z[local_idx[valid]])
                loss = F.cross_entropy(logits, labels[valid].long(), weight=cw)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(tgn.parameters(),
                                                   cfg["train"].get("grad_clip", 1.0))
                    optimizer.step()
                    tgn.memory.detach()
                with torch.no_grad():
                    prob = F.softmax(logits, dim=-1)[:, 1]
                total_loss += float(loss.item()) * int(valid.sum().item())
                total_n += int(valid.sum().item())
                y_true.append(labels[valid].detach().cpu().numpy())
                y_score.append(prob.detach().cpu().numpy())

            # Write events into memory + update neighbor cache
            if update_memory:
                tgn.memory.update_state(src, dst, t, msg)
                neighbor_loader.insert(src, dst)

        mean_loss = total_loss / max(total_n, 1)
        yt = np.concatenate(y_true) if y_true else np.array([])
        ys = np.concatenate(y_score) if y_score else np.array([])
        return mean_loss, yt, ys

    for ep in range(1, num_epochs + 1):
        tgn.reset_state()
        neighbor_loader.reset_state()

        tl, _, _ = run_split(train_loader, train=True, update_memory=True)

        # Validation: freeze params, but we still advance memory over val events
        # so embeddings use realistic history. No grad.
        with torch.no_grad():
            tgn.eval()
            tgn.detach_memory()
            _, y_val, s_val = run_split(val_loader, train=False, update_memory=True)

        if len(y_val) == 0:
            log.warning("val split had no labeled events; skipping metric.")
            continue
        val_metrics = compute_metrics(
            y_val, s_val,
            search_threshold=cfg["train"].get("threshold_search", True),
            pos_label=1,
        )
        is_best = stopper.step(val_metrics.f1_anomaly, tgn)
        log.info(
            f"ep {ep:03d}  train_loss {tl:.4f}  val AUC {val_metrics.roc_auc:.4f}  "
            f"val F1-anom {val_metrics.f1_anomaly:.4f}  thr {val_metrics.threshold:.3f}  "
            f"{'*' if is_best else ''}"
        )
        if stopper.should_stop:
            log.info(f"early stop @ epoch {ep}; best val F1-anom = {stopper.best:.4f}")
            break

    # Restore best params, replay memory from scratch on train+val, then test.
    stopper.restore(tgn)
    tgn.reset_state()
    neighbor_loader.reset_state()
    with torch.no_grad():
        tgn.eval()
        run_split(train_loader, train=False, update_memory=True)
        _, y_val, s_val = run_split(val_loader, train=False, update_memory=True)
        _, y_test, s_test = run_split(test_loader, train=False, update_memory=True)

    val_metrics = compute_metrics(
        y_val, s_val,
        search_threshold=cfg["train"].get("threshold_search", True),
        pos_label=1,
    )
    test_metrics = compute_metrics(
        y_test, s_test,
        threshold=val_metrics.threshold,
        search_threshold=False,
        pos_label=1,
    )
    log.info("=" * 70)
    log.info("FINAL  dataset=elliptic  model=tgn")
    log.info(f"val  : AUC {val_metrics.roc_auc:.4f}  F1-anom {val_metrics.f1_anomaly:.4f}")
    log.info(f"test : AUC {test_metrics.roc_auc:.4f}  F1-anom {test_metrics.f1_anomaly:.4f}  "
             f"P {test_metrics.precision_anomaly:.4f}  R {test_metrics.recall_anomaly:.4f}")
    log.info(f"confusion (test):\n{np.array(test_metrics.confusion)}")
    log.info(f"classification report (test):\n{test_metrics.report_text}")

    append_result(cfg["logging"]["results_csv"], test_metrics.as_row("tgn", args.dataset, "test"))
    append_result(cfg["logging"]["results_csv"], val_metrics.as_row("tgn", args.dataset, "val"))


if __name__ == "__main__":
    main()
