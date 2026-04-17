from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import flwr as fl
import torch.nn as nn
from torch.utils.data import DataLoader

from src.fl.server.evaluator import make_evaluate_fn
from src.fl.server.secure_aggregation import setup_secure_aggregation
from src.fl.server.strategies.fedavg import CustomFedAvg
from src.fl.server.strategies.fedprox import CustomFedProx
from src.fl.server.strategies.trimmed_mean import TrimmedMean
from src.models.factory import get_model
from src.utils.logging import get_logger
from src.utils.metrics import MetricsTracker

log = get_logger(__name__)

_STRATEGY_REGISTRY = {
    "fedavg":       CustomFedAvg,
    "fedprox":      CustomFedProx,
    "trimmed_mean": TrimmedMean,
}

def build_strategy(
    fl_cfg: SimpleNamespace,
    model_cfg: SimpleNamespace,
    val_loader: Optional[DataLoader] = None,
    metrics_tracker: Optional[MetricsTracker] = None,
) -> fl.server.strategy.Strategy:
    strategy_name = getattr(fl_cfg, "strategy", "fedavg").lower()

    if strategy_name not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: '{strategy_name}'. "
            f"Available: {sorted(_STRATEGY_REGISTRY.keys())}"
        )

    rounds_cfg = fl_cfg.rounds

    base_kwargs = dict(
        fraction_fit=float(getattr(rounds_cfg, "fraction_fit", 0.5)),
        fraction_evaluate=float(getattr(rounds_cfg, "fraction_evaluate", 0.25)),
        min_fit_clients=int(getattr(rounds_cfg, "min_fit_clients", 2)),
        min_evaluate_clients=int(getattr(rounds_cfg, "min_evaluate_clients", 2)),
        min_available_clients=int(getattr(rounds_cfg, "min_available_clients", 5)),
        metrics_tracker=metrics_tracker,
    )

    # Setup secure aggregation if enabled
    secure_agg = setup_secure_aggregation(fl_cfg)
    if secure_agg is not None:
        base_kwargs["secure_aggregation"] = secure_agg
        log.info("Secure aggregation enabled in strategy")

    if val_loader is not None:
        global_model = get_model(model_cfg)
        evaluate_fn  = make_evaluate_fn(
            model=global_model,
            val_loader=val_loader,
            metrics_tracker=metrics_tracker,
        )
        base_kwargs["evaluate_fn"] = evaluate_fn
    else:
        base_kwargs["evaluate_fn"] = None

    if strategy_name == "fedprox":
        mu = getattr(getattr(fl_cfg.aggregation, "fedprox", None), "mu", 0.01)
        strategy = CustomFedProx(mu=mu, **base_kwargs)

    elif strategy_name == "trimmed_mean":
        trim = getattr(
            getattr(fl_cfg.aggregation, "trimmed_mean", None), "trim_fraction", 0.1
        )
        strategy = TrimmedMean(trim_fraction=trim, **base_kwargs)

    else:
        strategy = CustomFedAvg(**base_kwargs)

    sa_status = "enabled" if secure_agg is not None else "disabled"
    log.info(
        "Strategy: %s | fraction_fit=%.2f | min_fit=%d | eval=%s | secure_agg=%s",
        strategy_name,
        base_kwargs["fraction_fit"],
        base_kwargs["min_fit_clients"],
        "enabled" if val_loader is not None else "disabled",
        sa_status,
    )

    return strategy

def start_server(
    fl_cfg: SimpleNamespace,
    model_cfg: SimpleNamespace,
    val_loader: Optional[DataLoader] = None,
    metrics_tracker: Optional[MetricsTracker] = None,
    server_address: str = "0.0.0.0:8080",
) -> fl.server.History:
    strategy  = build_strategy(fl_cfg, model_cfg, val_loader, metrics_tracker)
    num_rounds = int(fl_cfg.rounds.num_rounds)

    log.info(
        "Starting FL server at %s | rounds=%d | strategy=%s",
        server_address, num_rounds, fl_cfg.strategy,
    )
    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    if metrics_tracker is not None:
        metrics_tracker.close()

    log.info("FL server finished. Rounds completed: %d", len(history.losses_distributed))
    return history