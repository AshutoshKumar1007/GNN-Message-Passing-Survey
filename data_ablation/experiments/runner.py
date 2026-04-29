"""ExperimentRunner: executes one (model, dataset, feature_setting, seed) run.

The runner deliberately keeps responsibilities small: it wires together the
loaders, feature transforms, and trainers, then writes a `RunRecord` to disk.
Cross-run aggregation (mean ± std, plots, etc.) lives in `analysis/`.
"""
from __future__ import annotations
import torch
import gc

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from datasets.registry import DatasetCategory, load_dataset
from models.registry import build_model
from training.feature_transform import apply_feature_setting
from training.full_batch import TrainingResult, train_full_batch
from training.sampled import train_with_neighbor_sampling
from training.splits import resolve_split
from training.temporal import train_temporal
from utils.config import FeatureSetting, RunConfig
from utils.logging_utils import get_logger
from utils.metrics import MetricBundle
from utils.seed import set_seed


@dataclass
class RunRecord:
    config: RunConfig
    test_metrics: dict[str, float]
    val_metrics: dict[str, float]
    best_epoch: int
    homophily: dict[str, float]
    extras: dict[str, Any] = field(default_factory=dict)

    def flatten(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "dataset": self.config.dataset,
            "model": self.config.model,
            "feature_setting": self.config.feature_setting.value,
            "feature_selector": (
                self.config.feature_selector.value if self.config.feature_selector else None
            ),
            "topk": self.config.topk,
            "seed": self.config.seed,
            "best_epoch": self.best_epoch,
            **{f"test_{k}": v for k, v in self.test_metrics.items()},
            **{f"val_{k}": v for k, v in self.val_metrics.items()},
            **{f"homophily_{k}": v for k, v in self.homophily.items()},
        }
        return row


class ExperimentRunner:
    def __init__(
        self,
        *,
        data_root: str | Path = "data",
        output_root: str | Path = "reports/results",
        device: str | None = None,
        batch_threshold: int = 30_000,
    ) -> None:
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_threshold = batch_threshold
        self.logger = get_logger("runner", self.output_root / "logs")

    # ------------------------------------------------------------------
    # def run(self, cfg: RunConfig, *, split_idx: int = 0) -> RunRecord:
    #     set_seed(cfg.seed)
    #     bundle = load_dataset(cfg.dataset, root=self.data_root)
    #     data = bundle.data.clone()

    #     train_mask, val_mask, test_mask = resolve_split(data, split_idx)

    #     # ---- Apply feature ablation BEFORE moving to device ----
    #     data.x = apply_feature_setting(
    #         data.x.float(),
    #         setting=cfg.feature_setting,
    #         y=data.y,
    #         train_mask=train_mask,
    #         selector=cfg.feature_selector,
    #         topk=cfg.topk,
    #         seed=cfg.seed,
    #     )
    #     data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    #     in_dim = data.x.size(1)
    #     num_classes = bundle.num_classes
    #     num_nodes = data.num_nodes

    #     model = build_model(
    #         cfg.model,
    #         in_dim=in_dim,
    #         hidden_dim=cfg.hidden_dim,
    #         num_classes=num_classes,
    #         num_nodes=num_nodes,
    #         num_layers=cfg.num_layers,
    #         dropout=cfg.dropout,
    #         extra=cfg.extra,
    #     ).to(self.device)

    #     data = data.to(self.device)

    #     result = self._dispatch(cfg, model, data, bundle.category)

    #     record = RunRecord(
    #         config=cfg,
    #         test_metrics=result.test_metrics.as_dict(),
    #         val_metrics=result.val_metrics.as_dict(),
    #         best_epoch=result.best_epoch,
    #         homophily=bundle.homophily_summary(),
    #         extras={
    #             "train_history": result.history,
    #             "split_idx": split_idx,
    #             "in_dim": in_dim,
    #             "num_nodes": num_nodes,
    #             "num_classes": num_classes,
    #         },
    #     )

    #     self._persist(cfg, record, result)
    #             # ---- CLEANUP GPU MEMORY ----
    #     try:
    #         del model
    #         del data
    #         del result
    #     except:
    #         pass

    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     return record
    def run(self, cfg: RunConfig, *, split_idx: int = 0) -> RunRecord:
        # ---- PRE-CLEAN (important when running multiple datasets) ----
        gc.collect()
        torch.cuda.empty_cache()

        set_seed(cfg.seed)

        bundle = load_dataset(cfg.dataset, root=self.data_root)

        # Avoid clone unless absolutely needed
        data = bundle.data  

        train_mask, val_mask, test_mask = resolve_split(data, split_idx)

        data.x = apply_feature_setting(
            data.x.float(),
            setting=cfg.feature_setting,
            y=data.y,
            train_mask=train_mask,
            selector=cfg.feature_selector,
            topk=cfg.topk,
            seed=cfg.seed,
        )

        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

        in_dim = data.x.size(1)
        num_classes = bundle.num_classes
        num_nodes = data.num_nodes

        model = build_model(
            cfg.model,
            in_dim=in_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=num_classes,
            num_nodes=num_nodes,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            extra=cfg.extra,
        ).to(self.device)

        data = data.to(self.device)

        try:
            result = self._dispatch(cfg, model, data, bundle.category)

            record = RunRecord(
                config=cfg,
                test_metrics=result.test_metrics.as_dict(),
                val_metrics=result.val_metrics.as_dict(),
                best_epoch=result.best_epoch,
                homophily=bundle.homophily_summary(),
                extras={
                    "train_history": result.history,
                    "split_idx": split_idx,
                    "in_dim": in_dim,
                    "num_nodes": num_nodes,
                    "num_classes": num_classes,
                },
            )

            self._persist(cfg, record, result)

        finally:
            # ---- GUARANTEED CLEANUP ----
            try:
                print("Deleted Model")
                del model
                del data
                if "result" in locals():
                    del result
            except:
                pass

            gc.collect()
            torch.cuda.empty_cache()
            print("freed cuda cache")

        return record

    # ------------------------------------------------------------------
    def _dispatch(
        self,
        cfg: RunConfig,
        model: torch.nn.Module,
        data: Any,
        category: DatasetCategory,
    ) -> TrainingResult:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

        is_tgn = cfg.model.lower() == "tgn"
        if is_tgn:
            # Only TGN accepts the edge_time / update_memory kwargs.
            # Static models on temporal datasets fall through to the standard
            # trainer below — that is the canonical "static-vs-temporal"
            # comparison.
            edge_time = getattr(data, "edge_attr", None)
            if edge_time is not None and edge_time.dim() > 1:
                edge_time = edge_time[:, 0]
            return train_temporal(
                model,
                data,
                train_mask,
                val_mask,
                test_mask,
                epochs=min(cfg.train.epochs, 30),
                lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                patience=cfg.train.patience // 5,
                edge_time=edge_time,
            )

        if data.num_nodes >= self.batch_threshold or cfg.train.batch_size > 0:
            batch = cfg.train.batch_size or 1024
            return train_with_neighbor_sampling(
                model,
                data,
                train_mask,
                val_mask,
                test_mask,
                epochs=min(cfg.train.epochs, 100),
                lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                patience=cfg.train.patience // 5,
                batch_size=batch,
                num_neighbors=cfg.train.num_neighbors,
                device=self.device,
            )

        return train_full_batch(
            model,
            data,
            train_mask,
            val_mask,
            test_mask,
            epochs=cfg.train.epochs,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            patience=cfg.train.patience,
        )

    # ------------------------------------------------------------------
    def _persist(self, cfg: RunConfig, record: RunRecord, result: TrainingResult) -> None:
        run_dir = self.output_root / cfg.dataset / cfg.model
        run_dir.mkdir(parents=True, exist_ok=True)
        base = cfg.short_id()

        config_path = run_dir / f"{base}.config.json"
        config_path.write_text(json.dumps(asdict(cfg), indent=2, default=str))

        metrics_path = run_dir / f"{base}.metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "test": record.test_metrics,
                    "val": record.val_metrics,
                    "best_epoch": record.best_epoch,
                    "homophily": record.homophily,
                },
                indent=2,
            )
        )

        if result.embeddings is not None:
            torch.save(
                {
                    "embeddings": result.embeddings,
                    "test_logits": result.test_logits,
                    "history": result.history,
                },
                run_dir / f"{base}.pt",
            )

        self.logger.info(
            "Run %s | acc=%.4f f1=%.4f auc=%.4f",
            base,
            record.test_metrics["accuracy"],
            record.test_metrics["f1_macro"],
            record.test_metrics["auc"],
        )
