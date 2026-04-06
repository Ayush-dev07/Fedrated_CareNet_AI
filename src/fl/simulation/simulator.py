from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import flwr as fl

from src.fl.server.server import build_strategy
from src.fl.simulation.client_fn import make_client_fn_from_config
from src.fl.simulation.scenarios import SimulationConfig, get_scenario
from src.utils.logging import get_logger
from src.utils.metrics import MetricsTracker, PrivacyMetricsTracker
from src.utils.seed import set_seed

log = get_logger(__name__)

def run_simulation(
    sim_cfg: SimulationConfig | str,
    configs_dir: str = "configs",
    results_dir: str = "results",
    mlflow_enabled: bool | None = None,
) -> fl.server.History:
    
    if isinstance(sim_cfg, str):
        sim_cfg = get_scenario(sim_cfg, configs_dir=configs_dir)

    fl_cfg      = sim_cfg.fl_cfg
    model_cfg   = sim_cfg.model_cfg
    data_cfg    = sim_cfg.data_cfg
    privacy_cfg = sim_cfg.privacy_cfg

    seed = getattr(getattr(data_cfg, "partitioning", None), "seed", 42) if data_cfg else 42
    set_seed(seed)
    log.info("=== Simulation: %s ===", sim_cfg.summary())

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    metrics_tracker = MetricsTracker(results_path / "metrics.csv")
    privacy_tracker = PrivacyMetricsTracker(results_path / "privacy.csv") \
        if sim_cfg.dp_enabled else None

    mlflow_run = None
    _mlflow_enabled = mlflow_enabled if mlflow_enabled is not None else False
    try:
        if _mlflow_enabled:
            import mlflow
            mlflow.set_experiment(sim_cfg.scenario_name)
            mlflow_run = mlflow.start_run(run_name=sim_cfg.scenario_name)
            mlflow.log_params({
                "scenario":       sim_cfg.scenario_name,
                "num_clients":    sim_cfg.num_clients,
                "num_rounds":     sim_cfg.num_rounds,
                "strategy":       sim_cfg.strategy,
                "iid":            sim_cfg.iid,
                "dp_enabled":     sim_cfg.dp_enabled,
                "local_epochs":   sim_cfg.local_epochs,
            })
    except ImportError:
        log.debug("MLflow not available — skipping experiment tracking.")

    client_fn = make_client_fn_from_config(
        config=data_cfg,
        model_cfg=model_cfg,
        privacy_cfg=privacy_cfg,
        fl_cfg=fl_cfg,
    )

    strategy = build_strategy(
        fl_cfg=fl_cfg,
        model_cfg=model_cfg,
        val_loader=None,          
        metrics_tracker=metrics_tracker,
    )

    num_rounds  = sim_cfg.num_rounds
    num_clients = sim_cfg.num_clients

    log.info(
        "Starting Flower simulation: %d clients, %d rounds, strategy=%s",
        num_clients, num_rounds, sim_cfg.strategy,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"ignore_reinit_error": True, "num_cpus": num_clients},
    )

    
    _log_history(history, metrics_tracker, sim_cfg)

    metrics_tracker.close()
    if privacy_tracker is not None:
        privacy_tracker.close()

    if mlflow_run is not None:
        try:
            import mlflow
            final_losses = history.losses_distributed
            if final_losses:
                mlflow.log_metric("final_loss", final_losses[-1][1])
            mlflow.end_run()
        except Exception:
            pass

    log.info(
        "Simulation complete. Results written to: %s",
        results_path.resolve(),
    )
    return history

def _log_history(
    history: fl.server.History,
    metrics_tracker: MetricsTracker,
    sim_cfg: SimulationConfig,
) -> None:
    losses = dict(history.losses_distributed)

    for metric_name, values in history.metrics_distributed.items():
        for round_num, value in values:
            metrics_tracker.log({
                "round":     round_num,
                "source":    "distributed",
                "metric":    metric_name,
                "value":     value,
                "scenario":  sim_cfg.scenario_name,
                "strategy":  sim_cfg.strategy,
            })

    if losses:
        last_round, last_loss = max(losses.items())
        log.info(
            "Final round %d | loss=%.4f | scenario=%s",
            last_round, last_loss, sim_cfg.scenario_name,
        )

def run_quick_test(configs_dir: str = "configs") -> fl.server.History:

    log.info("Running quick smoke test (5 clients, 3 rounds)...")
    return run_simulation("quick_test", configs_dir=configs_dir, results_dir="results/quick_test")