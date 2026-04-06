from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.utils import get_parameters
from src.utils.logging import get_logger

log = get_logger(__name__)

class LocalTrainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cpu",
        grad_clip: float = 1.0,
    ) -> None:
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.loss_fn   = loss_fn
        self.device    = device
        self.grad_clip = grad_clip

    def train(
        self,
        loader: DataLoader,
        epochs: int = 2,
    ) -> tuple[List[np.ndarray], dict]:

        self.model.train()

        total_loss     = 0.0
        total_correct  = 0
        total_samples  = 0
        total_batches  = 0

        for epoch in range(epochs):
            epoch_loss    = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss   = self.loss_fn(logits, y)
                loss.backward()

                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

                batch_size      = y.size(0)
                epoch_loss     += loss.item() * batch_size
                preds           = logits.argmax(dim=-1)
                epoch_correct  += (preds == y).sum().item()
                epoch_samples  += batch_size

            if epoch_samples > 0:
                log.debug(
                    "  epoch %d/%d  loss=%.4f  acc=%.4f  samples=%d",
                    epoch + 1, epochs,
                    epoch_loss / epoch_samples,
                    epoch_correct / epoch_samples,
                    epoch_samples,
                )

            total_loss    += epoch_loss
            total_correct += epoch_correct
            total_samples  = epoch_samples   # Use last epoch's sample count
            total_batches += max(1, epoch_samples)

        n = total_samples * epochs if total_samples > 0 else 1
        metrics = {
            "train_loss":     round(total_loss / n, 6),
            "train_accuracy": round(total_correct / n, 6),
            "num_samples":    total_samples,
            "num_epochs":     epochs,
        }

        updated_weights = get_parameters(self.model)
        return updated_weights, metrics

    def evaluate(
        self,
        loader: DataLoader,
    ) -> dict:

        self.model.eval()
        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss   = self.loss_fn(logits, y)

                batch_size     = y.size(0)
                total_loss    += loss.item() * batch_size
                preds          = logits.argmax(dim=-1)
                total_correct += (preds == y).sum().item()
                total_samples += batch_size

        n = max(total_samples, 1)
        return {
            "val_loss":     round(total_loss / n, 6),
            "val_accuracy": round(total_correct / n, 6),
            "num_samples":  total_samples,
        }