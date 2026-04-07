from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.utils import set_parameters
from src.utils.logging import get_logger

log = get_logger(__name__)

EvaluateFn = Callable[[int, List[np.ndarray], Dict], Optional[Tuple[float, Dict[str, float]]]]

def evaluate_global(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    all_probs:  List[float] = []
    all_labels: List[int]   = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss   = criterion(logits, y)

            batch_size     = y.size(0)
            total_loss    += loss.item() * batch_size
            preds          = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += batch_size

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().tolist())

    n = max(total_samples, 1)
    metrics: Dict[str, float] = {
        "loss":     round(total_loss / n, 6),
        "accuracy": round(total_correct / n, 6),
    }

    try:
        from sklearn.metrics import roc_auc_score
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
            metrics["auc"] = round(float(auc), 6)
        else:
            metrics["auc"] = 0.5   
    except ImportError:
        pass   

    log.info(
        "Global eval: loss=%.4f  acc=%.4f  auc=%s  samples=%d",
        metrics["loss"],
        metrics["accuracy"],
        f"{metrics.get('auc', 'N/A'):.4f}" if "auc" in metrics else "N/A",
        total_samples,
    )

    return metrics

def make_evaluate_fn(
    model: nn.Module,
    val_loader: DataLoader,
    metrics_tracker=None,
    device: str = "cpu",
) -> EvaluateFn:
    def evaluate_fn(
        server_round: int,
        parameters: List[np.ndarray],
        config: dict,
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        
        set_parameters(model, parameters)

        metrics = evaluate_global(model, val_loader, device=device)

        if metrics_tracker is not None:
            metrics_tracker.log({
                "round":   server_round,
                "source":  "global_eval",
                **metrics,
            })

        return metrics["loss"], metrics

    return evaluate_fn