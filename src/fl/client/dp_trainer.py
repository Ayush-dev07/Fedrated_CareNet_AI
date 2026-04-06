from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.fl.client.trainer import LocalTrainer
from src.fl.privacy.accountant import RDPAccountant
from src.fl.privacy.dp_engine import setup_dp
from src.models.utils import get_parameters
from src.utils.logging import get_logger
from src.fl.privacy.budget import PrivacyBudget

log = get_logger(__name__)

class DPLocalTrainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        privacy_cfg,
        loader: DataLoader,
        num_rounds: int = 20,
        device: str = "cpu",
    ) -> None:
        self.loss_fn  = loss_fn
        self.device   = device

        self.dp_model, self.dp_optimizer, self.dp_loader, self.accountant = setup_dp(
            model=model,
            optimizer=optimizer,
            loader=loader,
            config=privacy_cfg,
            num_rounds=num_rounds,
        )

        self._inner = LocalTrainer(
            model=self.dp_model,
            optimizer=self.dp_optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip=0.0,    
        )

    def train(
        self,
        loader: DataLoader,
        epochs: int = 2,
    ) -> tuple[List[np.ndarray], dict]:

        updated_weights, metrics = self._inner.train(self.dp_loader, epochs=epochs)

        steps_this_round = self._steps_per_epoch(self.dp_loader) * epochs
        epsilon = self.accountant.step(num_steps=steps_this_round)

        metrics["epsilon"] = round(epsilon, 6)
        metrics["delta"]   = self.accountant.delta
        metrics["dp"]      = True
        log.info(
            "DP round complete: ε=%.4f  δ=%.1e  loss=%.4f  acc=%.4f",
            epsilon,
            self.accountant.delta,
            metrics["train_loss"],
            metrics["train_accuracy"],
        )

        if self.accountant.budget.is_exhausted:
            log.error(
                "DP budget exhausted (ε=%.4f >= target=%.4f). "
                "Results beyond this point have no DP guarantee.",
                epsilon, self.accountant.budget.target_epsilon,
            )

        return updated_weights, metrics

    def evaluate(self, loader: DataLoader) -> dict:
        return self._inner.evaluate(loader)

    @property
    def budget(self) -> "PrivacyBudget":
        return self.accountant.budget

    @staticmethod
    def _steps_per_epoch(loader: DataLoader) -> int:
        try:
            return len(loader)
        except TypeError:
            return 0