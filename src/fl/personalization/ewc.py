from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.utils import clone_model
from src.utils.logging import get_logger

log = get_logger(__name__)

class EWC:

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        lambda_ewc: float = 400.0,
        n_batches: int = 10,
        device: str = "cpu",
    ) -> None:
        self.lambda_ewc = lambda_ewc
        self.device     = device

        self._reference_params: Dict[str, torch.Tensor] = {
            name: param.detach().clone().to(device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        self._fisher: Dict[str, torch.Tensor] = self._compute_fisher(
            model, loader, n_batches
        )

        log.debug(
            "EWC registered: λ=%.1f  params=%d  fisher_mean=%.6f",
            lambda_ewc,
            sum(p.numel() for p in self._fisher.values()),
            sum(f.mean().item() for f in self._fisher.values()) / max(len(self._fisher), 1),
        )

    def penalty(self, model: nn.Module) -> torch.Tensor:

        loss = torch.tensor(0.0, device=self.device)

        for name, param in model.named_parameters():
            if name not in self._fisher:
                continue
            fisher   = self._fisher[name].to(self.device)
            ref      = self._reference_params[name].to(self.device)
            loss    += (fisher * (param - ref) ** 2).sum()

        return (self.lambda_ewc / 2.0) * loss

    def update_reference(
        self,
        new_model: nn.Module,
        new_loader: DataLoader,
        n_batches: int = 10,
    ) -> None:

        self._reference_params = {
            name: param.detach().clone().to(self.device)
            for name, param in new_model.named_parameters()
            if param.requires_grad
        }
        self._fisher = self._compute_fisher(new_model, new_loader, n_batches)
        log.debug("EWC reference updated.")

    def _compute_fisher(
        self,
        model: nn.Module,
        loader: DataLoader,
        n_batches: int,
    ) -> Dict[str, torch.Tensor]:
        
        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        model = model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        processed = 0

        for batch_idx, (x, y) in enumerate(loader):
            if n_batches > 0 and batch_idx >= n_batches:
                break

            x = x.to(self.device)
            y = y.to(self.device)

            model.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

            processed += 1

        if processed > 0:
            for name in fisher:
                fisher[name] /= processed

        return fisher
    
    def fine_tune_with_ewc(
        self,
        global_model: nn.Module,
        client_loader: DataLoader,
        n_steps: int = 10,
        learning_rate: float = 1e-3,
        loss_fn: nn.Module | None = None,
    ) -> tuple[nn.Module, Dict[str, float]]:

        model = clone_model(global_model).to(self.device)
        model.train()

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate)
        data_iter  = _infinite_loader(client_loader)

        total_task_loss = 0.0
        total_ewc_loss  = 0.0
        steps = 0

        for _ in range(n_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                break

            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            logits    = model(x)
            task_loss = loss_fn(logits, y)
            ewc_loss  = self.penalty(model)
            loss      = task_loss + ewc_loss

            loss.backward()
            optimizer.step()

            total_task_loss += task_loss.item()
            total_ewc_loss  += ewc_loss.item()
            steps += 1

        n = max(steps, 1)
        metrics = {
            "task_loss":     round(total_task_loss / n, 6),
            "ewc_loss":      round(total_ewc_loss / n, 6),
            "total_loss":    round((total_task_loss + total_ewc_loss) / n, 6),
            "steps":         steps,
            "lambda_ewc":    self.lambda_ewc,
        }

        log.debug(
            "EWC fine-tune: steps=%d  task_loss=%.4f  ewc_loss=%.4f",
            steps, metrics["task_loss"], metrics["ewc_loss"],
        )

        return model, metrics

def _infinite_loader(loader: DataLoader):
    while True:
        yield from loader