from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src import fl

log = get_logger(__name__)

_COLORS = {
    "fedavg":       "#00c4b4",
    "fedprox":      "#7c5cbf",
    "trimmed_mean": "#f5a623",
    "dp_enabled":   "#ee4c2c",
    "centralized":  "#2d9cdb",
    "default":      "#c9d1d9",
}

def plot_convergence(
    history: "fl.server.History | None" = None,
    metrics_csv: str | Path | None = None,
    save_dir: str | Path = "results",
    filename: str = "convergence.png",
    title: str = "FL Convergence",
    show: bool = False,
) -> Path:

    if history is None and metrics_csv is None:
        raise ValueError("Provide either a Flower History object or a metrics_csv path.")

    if history is not None:
        rounds, losses, accuracies = _parse_flower_history(history)
    else:
        rounds, losses, accuracies = _parse_metrics_csv(metrics_csv)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")

    _style_ax(axes[0])
    _style_ax(axes[1])

    if losses:
        axes[0].plot(rounds[:len(losses)], losses, color=_COLORS["fedavg"],
                     linewidth=2, marker="o", markersize=4, label="Train Loss")
        axes[0].set_title("Loss per Round", color="#c9d1d9", fontsize=11)
        axes[0].set_xlabel("Round", color="#6e7681")
        axes[0].set_ylabel("Loss", color="#6e7681")
        axes[0].legend(facecolor="#161b22", labelcolor="#c9d1d9", framealpha=0.8)

    if accuracies:
        axes[1].plot(rounds[:len(accuracies)], accuracies,
                     color=_COLORS["fedprox"], linewidth=2,
                     marker="s", markersize=4, label="Accuracy")
        axes[1].set_title("Accuracy per Round", color="#c9d1d9", fontsize=11)
        axes[1].set_xlabel("Round", color="#6e7681")
        axes[1].set_ylabel("Accuracy", color="#6e7681")
        axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        axes[1].legend(facecolor="#161b22", labelcolor="#c9d1d9", framealpha=0.8)

    fig.suptitle(title, color="#c9d1d9", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    log.info("Convergence plot saved: %s", out_path)
    if show:
        plt.show()

    return out_path

def plot_multi_scenario_convergence(
    scenario_histories: Dict[str, "fl.server.History"],
    metric: str = "loss",
    save_dir: str | Path = "results",
    filename: str = "multi_scenario_convergence.png",
    show: bool = False,
) -> Path:

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    _style_ax(ax)

    for name, history in scenario_histories.items():
        rounds, losses, accuracies = _parse_flower_history(history)
        color  = _COLORS.get(name, _COLORS["default"])
        values = losses if metric == "loss" else accuracies

        if values:
            ax.plot(
                rounds[:len(values)], values,
                color=color, linewidth=2, marker="o", markersize=3,
                label=name,
            )

    ylabel = "Loss" if metric == "loss" else "Accuracy"
    ax.set_title(f"Multi-Scenario {ylabel} Comparison", color="#c9d1d9", fontsize=12)
    ax.set_xlabel("Round", color="#6e7681")
    ax.set_ylabel(ylabel, color="#6e7681")

    if metric == "accuracy":
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    ax.legend(facecolor="#161b22", labelcolor="#c9d1d9", framealpha=0.8,
              ncol=min(len(scenario_histories), 3))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    log.info("Multi-scenario plot saved: %s", out_path)
    if show:
        plt.show()

    return out_path

def plot_privacy_budget(
    privacy_csv: str | Path,
    target_epsilon: float,
    save_dir: str | Path = "results",
    filename: str = "privacy_budget.png",
    show: bool = False,
) -> Path:

    import pandas as pd

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename

    df = pd.read_csv(privacy_csv)

    if "round" not in df.columns or "epsilon" not in df.columns:
        log.warning("privacy.csv missing 'round' or 'epsilon' columns. Skipping plot.")
        return out_path

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0d1117")
    _style_ax(ax)

    ax.plot(df["round"], df["epsilon"], color=_COLORS["dp_enabled"],
            linewidth=2, marker="o", markersize=4, label="Cumulative ε")
    ax.axhline(y=target_epsilon, color="#e3b341", linestyle="--",
               linewidth=1.5, label=f"Target ε={target_epsilon}")
    ax.fill_between(df["round"], df["epsilon"], alpha=0.15, color=_COLORS["dp_enabled"])

    ax.set_title("Privacy Budget Depletion", color="#c9d1d9", fontsize=12)
    ax.set_xlabel("Round", color="#6e7681")
    ax.set_ylabel("Cumulative ε", color="#6e7681")
    ax.legend(facecolor="#161b22", labelcolor="#c9d1d9", framealpha=0.8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    log.info("Privacy budget plot saved: %s", out_path)
    if show:
        plt.show()

    return out_path

def _parse_flower_history(
    history: "fl.server.History",
) -> Tuple[List[int], List[float], List[float]]:
    rounds: List[int]   = []
    losses: List[float] = []
    accs:   List[float] = []

    if history.losses_distributed:
        for r, loss in sorted(history.losses_distributed):
            rounds.append(r)
            losses.append(loss)

    for metric_name, values in history.metrics_distributed.items():
        if "accuracy" in metric_name.lower():
            accs = [v for _, v in sorted(values)]
            break

    if not rounds:
        rounds = list(range(1, len(losses) + 1))

    return rounds, losses, accs

def _parse_metrics_csv(
    csv_path: str | Path,
) -> Tuple[List[int], List[float], List[float]]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    rounds = df["round"].tolist() if "round" in df.columns else list(range(len(df)))
    losses = df["train_loss"].tolist() if "train_loss" in df.columns else []
    accs   = df["train_accuracy"].tolist() if "train_accuracy" in df.columns else []

    return rounds, losses, accs

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#6e7681", labelsize=9)
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#30363d", linewidth=0.5, alpha=0.7)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("#6e7681")