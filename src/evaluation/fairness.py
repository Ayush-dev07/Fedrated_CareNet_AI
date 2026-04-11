from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logging import get_logger

log = get_logger(__name__)

def equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, float]:

    unique_groups = np.unique(groups)
    group_tpr: Dict = {}
    group_fpr: Dict = {}
    group_counts: Dict = {}

    for g in unique_groups:
        mask     = groups == g
        yt       = y_true[mask]
        yp       = y_pred[mask]
        group_counts[str(g)] = int(mask.sum())

        tp = int(((yp == 1) & (yt == 1)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        group_tpr[str(g)] = round(tpr, 6)
        group_fpr[str(g)] = round(fpr, 6)

    tpr_vals = list(group_tpr.values())
    fpr_vals = list(group_fpr.values())

    tpr_disparity = round(float(max(tpr_vals) - min(tpr_vals)), 6) if len(tpr_vals) > 1 else 0.0
    fpr_disparity = round(float(max(fpr_vals) - min(fpr_vals)), 6) if len(fpr_vals) > 1 else 0.0
    eo_score      = round(tpr_disparity + fpr_disparity, 6)

    log.info(
        "Equalized Odds: tpr_disparity=%.4f  fpr_disparity=%.4f  eo_score=%.4f  groups=%s",
        tpr_disparity, fpr_disparity, eo_score, list(group_tpr.keys()),
    )

    return {
        "tpr_disparity":  tpr_disparity,
        "fpr_disparity":  fpr_disparity,
        "eo_score":       eo_score,
        "group_tpr":      group_tpr,
        "group_fpr":      group_fpr,
        "group_counts":   group_counts,
    }

def demographic_parity(
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, float]:

    unique_groups = np.unique(groups)
    group_pos_rate: Dict = {}
    group_counts: Dict   = {}

    for g in unique_groups:
        mask = groups == g
        group_counts[str(g)]   = int(mask.sum())
        group_pos_rate[str(g)] = round(float(y_pred[mask].mean()), 6)

    rates = list(group_pos_rate.values())
    dp_disparity = round(float(max(rates) - min(rates)), 6) if len(rates) > 1 else 0.0

    log.info(
        "Demographic Parity: disparity=%.4f  groups=%s",
        dp_disparity, {k: f"{v:.3f}" for k, v in group_pos_rate.items()},
    )

    return {
        "dp_disparity":   dp_disparity,
        "group_pos_rate": group_pos_rate,
        "group_counts":   group_counts,
    }

def evaluate_per_client_fairness(
    model: nn.Module,
    client_loaders: Dict[int, DataLoader],
    client_groups: Optional[Dict[int, str]] = None,
    device: str = "cpu",
) -> Dict[str, object]:

    model.eval()
    model.to(device)

    per_client_acc:      Dict[int, float] = {}
    per_client_pos_rate: Dict[int, float] = {}
    per_client_n:        Dict[int, int]   = {}

    all_y_true: List[int] = []
    all_y_pred: List[int] = []
    all_groups: List[str] = []

    for cid, loader in client_loaders.items():
        y_true_list: List[int] = []
        y_pred_list: List[int] = []

        with torch.no_grad():
            for x, y in loader:
                x     = x.to(device)
                preds = model(x).argmax(dim=-1).cpu().tolist()
                y_true_list.extend(y.tolist())
                y_pred_list.extend(preds)

        y_true_arr = np.array(y_true_list)
        y_pred_arr = np.array(y_pred_list)

        accuracy  = float((y_true_arr == y_pred_arr).mean())
        pos_rate  = float(y_pred_arr.mean())
        n_samples = len(y_true_arr)

        per_client_acc[cid]      = round(accuracy, 6)
        per_client_pos_rate[cid] = round(pos_rate, 6)
        per_client_n[cid]        = n_samples

        group_label = client_groups.get(cid, str(cid)) if client_groups else str(cid)
        all_y_true.extend(y_true_list)
        all_y_pred.extend(y_pred_list)
        all_groups.extend([group_label] * n_samples)

    accs   = list(per_client_acc.values())
    result = {
        "per_client_accuracy":  {str(k): v for k, v in per_client_acc.items()},
        "per_client_pos_rate":  {str(k): v for k, v in per_client_pos_rate.items()},
        "per_client_n_samples": {str(k): v for k, v in per_client_n.items()},
        "accuracy_std":         round(float(np.std(accs)), 6),
        "min_accuracy":         round(float(min(accs)), 6),
        "max_accuracy":         round(float(max(accs)), 6),
        "mean_accuracy":        round(float(np.mean(accs)), 6),
    }

    if client_groups is not None:
        y_true_np  = np.array(all_y_true)
        y_pred_np  = np.array(all_y_pred)
        groups_np  = np.array(all_groups)
        result["equalized_odds"]     = equalized_odds(y_true_np, y_pred_np, groups_np)
        result["demographic_parity"] = demographic_parity(y_pred_np, groups_np)

    log.info(
        "Per-client fairness: mean_acc=%.4f  std=%.4f  min=%.4f  max=%.4f  n_clients=%d",
        result["mean_accuracy"], result["accuracy_std"],
        result["min_accuracy"],  result["max_accuracy"],
        len(client_loaders),
    )

    return result

def plot_fairness(
    fairness_results: Dict,
    save_dir: str | Path = "results",
    filename: str = "fairness.png",
    show: bool = False,
) -> Path:

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename

    per_client_acc = fairness_results.get("per_client_accuracy", {})
    if not per_client_acc:
        log.warning("No per_client_accuracy data — skipping fairness plot.")
        return out_path

    client_ids = sorted(per_client_acc.keys(), key=lambda x: int(x))
    accs       = [per_client_acc[c] for c in client_ids]
    mean_acc   = fairness_results.get("mean_accuracy", np.mean(accs))

    fig, ax = plt.subplots(figsize=(max(8, len(client_ids) * 0.5), 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    bars = ax.bar(client_ids, accs, color="#00c4b4", alpha=0.8, width=0.7)
    ax.axhline(y=mean_acc, color="#e3b341", linestyle="--",
               linewidth=1.5, label=f"Mean={mean_acc:.3f}")

    std = fairness_results.get("accuracy_std", 0.0)
    ax.set_title(
        f"Per-Client Accuracy  (std={std:.4f})",
        color="#c9d1d9", fontsize=11,
    )
    ax.set_xlabel("Client ID", color="#6e7681")
    ax.set_ylabel("Accuracy", color="#6e7681")
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="#6e7681")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    ax.legend(facecolor="#161b22", labelcolor="#c9d1d9", framealpha=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    log.info("Fairness plot saved: %s", out_path)
    if show:
        plt.show()
    return out_path