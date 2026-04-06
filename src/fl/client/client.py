from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
from torch.utils.data import DataLoader

from src.fl.client.trainer import LocalTrainer
from src.fl.privacy.dp_engine import is_dp_enabled
from src.models.factory import get_loss_fn, get_model, get_optimizer
from src.models.utils import get_parameters, set_parameters
from src.utils.logging import get_logger

log = get_logger(__name__)

NDArrays = List[np.ndarray]
Scalar   = float | int | str | bool
Metrics  = Dict[str, Scalar]

class HealthClient(fl.client.NumPyClient):

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_cfg: SimpleNamespace,
        privacy_cfg: SimpleNamespace,
        fl_cfg: SimpleNamespace,
    ) -> None:
        self.client_id    = client_id
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.model_cfg    = model_cfg
        self.privacy_cfg  = privacy_cfg
        self.fl_cfg       = fl_cfg

        seq_len = self._infer_seq_len(train_loader)

        self.model   = get_model(model_cfg, seq_len=seq_len)
        self.loss_fn = get_loss_fn(model_cfg)

        self._dp_enabled = is_dp_enabled(privacy_cfg)
        self._trainer: LocalTrainer | None = None   

        log.debug(
            "HealthClient %d: model=%s dp=%s train=%d val=%d",
            client_id, self.model.__class__.__name__,
            self._dp_enabled, len(train_loader.dataset),  
            len(val_loader.dataset),  
        )

    def get_parameters(self, config: dict) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self,
        parameters: NDArrays,
        config: dict,
    ) -> Tuple[NDArrays, int, Metrics]:

        set_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", getattr(self.fl_cfg.local_training, "local_epochs", 2)))
        current_round = int(config.get("current_round", 0))
        num_rounds    = int(config.get("num_rounds", getattr(self.fl_cfg.rounds, "num_rounds", 20)))

        trainer = self._build_trainer(num_rounds=num_rounds)

        updated_weights, metrics = trainer.train(self.train_loader, epochs=local_epochs)

        metrics["client_id"]    = self.client_id
        metrics["current_round"] = current_round

        log.info(
            "Client %d | round %d | loss=%.4f | acc=%.4f | samples=%d%s",
            self.client_id, current_round,
            metrics.get("train_loss", 0),
            metrics.get("train_accuracy", 0),
            metrics.get("num_samples", 0),
            f" | ε={metrics['epsilon']:.4f}" if "epsilon" in metrics else "",
        )

        return updated_weights, metrics["num_samples"], metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict,
    ) -> Tuple[float, int, Metrics]:

        set_parameters(self.model, parameters)
        trainer = self._get_eval_trainer()
        metrics = trainer.evaluate(self.val_loader)

        log.debug(
            "Client %d eval: loss=%.4f  acc=%.4f  samples=%d",
            self.client_id,
            metrics["val_loss"],
            metrics["val_accuracy"],
            metrics["num_samples"],
        )

        return (
            metrics["val_loss"],
            metrics["num_samples"],
            {"val_accuracy": metrics["val_accuracy"], "client_id": self.client_id},
        )

    def _build_trainer(self, num_rounds: int) -> LocalTrainer:
        optimizer = get_optimizer(self.model, self.model_cfg)
        grad_clip = getattr(self.model_cfg.training, "gradient_clip_norm", 1.0)

        if self._dp_enabled:
            from src.fl.client.dp_trainer import DPLocalTrainer
            return DPLocalTrainer(
                model=self.model,
                optimizer=optimizer,
                loss_fn=self.loss_fn,
                privacy_cfg=self.privacy_cfg,
                loader=self.train_loader,
                num_rounds=num_rounds,
                device="cpu",
            )

        return LocalTrainer(
            model=self.model,
            optimizer=optimizer,
            loss_fn=self.loss_fn,
            device="cpu",
            grad_clip=grad_clip,
        )

    def _get_eval_trainer(self) -> LocalTrainer:
        import torch.optim as optim
        dummy_opt = optim.SGD(self.model.parameters(), lr=0.0)
        return LocalTrainer(
            model=self.model,
            optimizer=dummy_opt,
            loss_fn=self.loss_fn,
            device="cpu",
            grad_clip=0.0,
        )

    @staticmethod
    def _infer_seq_len(loader: DataLoader) -> int | None:
        try:
            x, _ = next(iter(loader))
            return x.shape[-1]   
        except Exception:
            return None