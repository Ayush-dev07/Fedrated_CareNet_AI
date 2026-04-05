from __future__ import annotations

from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.utils import clone_model
from src.utils.logging import get_logger

log = get_logger(__name__)

def local_fine_tune(
    global_model: nn.Module,
    client_loader: DataLoader,
    n_steps: int = 10,
    learning_rate: float = 1e-3,
    loss_fn: nn.Module | None = None,
    device: str = "cpu",
    freeze_layers: int = 0,
) -> tuple[nn.Module, Dict[str, float]]:

    model = clone_model(global_model).to(device)
    model.train()

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if freeze_layers > 0:
        _freeze_early_layers(model, freeze_layers)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    data_iter = _infinite_loader(client_loader)

    total_loss = 0.0
    completed_steps = 0

    for step in range(n_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            break

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        completed_steps += 1

    avg_loss = total_loss / max(completed_steps, 1)

    log.debug(
        "Fine-tune: steps=%d  avg_loss=%.4f  lr=%.5f  frozen_layers=%d",
        completed_steps, avg_loss, learning_rate, freeze_layers,
    )

    metrics = {
        "fine_tune_loss":    round(avg_loss, 6),
        "steps":             completed_steps,
        "lr":                learning_rate,
        "frozen_layers":     freeze_layers,
    }

    return model, metrics

def evaluate_personalisation_gain(
    global_model: nn.Module,
    personal_model: nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:

    global_acc  = _accuracy(global_model,  val_loader, device)
    personal_acc = _accuracy(personal_model, val_loader, device)

    gain = personal_acc - global_acc
    n    = len(val_loader.dataset)  

    log.info(
        "Personalisation: global_acc=%.4f  personal_acc=%.4f  gain=%+.4f  samples=%d",
        global_acc, personal_acc, gain, n,
    )

    return {
        "global_accuracy":   round(global_acc, 6),
        "personal_accuracy": round(personal_acc, 6),
        "accuracy_gain":     round(gain, 6),
        "num_val_samples":   n,
    }

def _accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds    = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / max(total, 1)


def _freeze_early_layers(model: nn.Module, n_layers: int) -> None:

    frozen = 0
    for name, param in model.named_parameters():
        if "lstm" in name and frozen < n_layers:
            param.requires_grad = False
            frozen += 1
        else:
            param.requires_grad = True

def _infinite_loader(loader: DataLoader):
    while True:
        yield from loader