"""Run FL simulation — in-process Flower VirtualClientEngine.

Usage:
    python scripts/run_sim.py
    python scripts/run_sim.py --scenario noniid_20
    python scripts/run_sim.py --scenario dp_enabled --rounds 10
    python scripts/run_sim.py --scenario fedprox --rounds 20 --results_dir results/fedprox_run
    python scripts/run_sim.py --list_scenarios

Available scenarios (from simulation.yaml):
    iid_20          20 clients, IID, FedAvg, no DP
    noniid_20       20 clients, non-IID (α=0.5), FedAvg, no DP
    noniid_extreme  20 clients, extreme non-IID (α=0.1)
    dp_enabled      20 clients, non-IID, FedAvg + DP-SGD (ε=3.0)
    fedprox         20 clients, non-IID, FedProx (μ=0.01)
    trimmed_mean    20 clients, non-IID, TrimmedMean defense
    quick_test      5 clients, 3 rounds — smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fl.simulation.scenarios import get_scenario, list_scenarios
from src.fl.simulation.simulator import run_simulation
from src.evaluation.convergence import plot_convergence
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a federated learning simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scenario",
        type=str,
        default="iid_20",
        help="Scenario name. Use --list_scenarios to see all options.",
    )
    p.add_argument("--rounds",      type=int,  default=None, help="Override num_rounds.")
    p.add_argument("--clients",     type=int,  default=None, help="Override num_clients.")
    p.add_argument("--local_epochs",type=int,  default=None, help="Override local_epochs.")
    p.add_argument("--seed",        type=int,  default=None, help="Override RNG seed.")
    p.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to write metrics.csv and plots.",
    )
    p.add_argument(
        "--configs_dir",
        type=str,
        default="configs",
        help="Path to YAML config directory.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Save convergence plot after simulation.",
    )
    p.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip convergence plot.",
    )
    p.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking (local mode).",
    )
    p.add_argument(
        "--list_scenarios",
        action="store_true",
        help="Print all available scenarios and exit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_scenarios:
        print("\nAvailable scenarios:\n")
        for name, desc in sorted(list_scenarios().items()):
            print(f"  {name:<20}  {desc}")
        print()
        sys.exit(0)

    # ── Load and optionally override scenario config ──────────────────────────
    sim_cfg = get_scenario(args.scenario, configs_dir=args.configs_dir)

    if args.rounds is not None:
        sim_cfg.num_rounds                      = args.rounds
        sim_cfg.fl_cfg.rounds.num_rounds        = args.rounds
        log.info("Override: num_rounds=%d", args.rounds)

    if args.clients is not None:
        sim_cfg.num_clients                           = args.clients
        sim_cfg.data_cfg.partitioning.num_clients     = args.clients
        log.info("Override: num_clients=%d", args.clients)

    if args.local_epochs is not None:
        sim_cfg.local_epochs                               = args.local_epochs
        sim_cfg.fl_cfg.local_training.local_epochs        = args.local_epochs
        log.info("Override: local_epochs=%d", args.local_epochs)

    if args.seed is not None:
        sim_cfg.seed                                  = args.seed
        sim_cfg.data_cfg.partitioning.seed            = args.seed
        set_seed(args.seed)
        log.info("Override: seed=%d", args.seed)

    # ── Run simulation ────────────────────────────────────────────────────────
    log.info("Starting simulation: %s", sim_cfg.summary())

    history = run_simulation(
        sim_cfg=sim_cfg,
        results_dir=args.results_dir,
        mlflow_enabled=args.mlflow,
    )

    # ── Post-simulation reporting ─────────────────────────────────────────────
    if history.losses_distributed:
        rounds_completed = len(history.losses_distributed)
        final_round, final_loss = max(history.losses_distributed, key=lambda x: x[0])
        log.info(
            "Simulation finished: %d rounds  final_loss=%.4f",
            rounds_completed, final_loss,
        )

    # ── Convergence plot ──────────────────────────────────────────────────────
    if args.plot and not args.no_plot:
        try:
            plot_path = plot_convergence(
                history=history,
                save_dir=args.results_dir,
                filename=f"{args.scenario}_convergence.png",
                title=f"FL Convergence — {args.scenario}",
            )
            log.info("Convergence plot saved: %s", plot_path)
        except Exception as e:
            log.warning("Could not save convergence plot: %s", e)

    log.info(
        "Results written to: %s",
        Path(args.results_dir).resolve(),
    )

if __name__ == "__main__":
    main()