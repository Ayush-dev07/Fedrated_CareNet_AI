from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.benchmark import compare_fl_vs_centralized
from src.fl.simulation.scenarios import get_scenario, list_scenarios
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)

_BENCHMARK_SCENARIOS = ["iid_20", "noniid_20", "fedprox"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark FL vs centralised training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scenario",       type=str, default="iid_20", help="Scenario to benchmark.")
    p.add_argument("--central_epochs", type=int, default=20,       help="Centralised training epochs.")
    p.add_argument("--results_dir",    type=str, default="results", help="Output directory.")
    p.add_argument("--configs_dir",    type=str, default="configs", help="YAML config directory.")
    p.add_argument("--seed",           type=int, default=42,        help="RNG seed.")
    p.add_argument(
        "--all_scenarios",
        action="store_true",
        help=f"Run benchmark for all core scenarios: {_BENCHMARK_SCENARIOS}.",
    )
    return p.parse_args()

def run_one(
    scenario: str,
    central_epochs: int,
    results_dir: str,
    configs_dir: str,
    seed: int,
) -> dict:
    set_seed(seed)
    sim_cfg = get_scenario(scenario, configs_dir=configs_dir)
    comparison = compare_fl_vs_centralized(
        sim_cfg=sim_cfg,
        results_dir=results_dir,
        centralized_epochs=central_epochs,
    )

    c = comparison["centralized"]
    f = comparison["federated"]
    log.info("── Benchmark: %s ──────────────────", scenario)
    log.info(
        "  Centralised : acc=%.4f  auc=%s  loss=%.4f  time=%.1fs  comm=%.2fMB",
        c["accuracy"],
        f"{c.get('auc', 0):.4f}" if "auc" in c else "N/A",
        c["loss"], c["train_time_s"], c.get("comm_cost_mb", 0),
    )
    log.info(
        "  Federated   : acc=%.4f  auc=%s  loss=%.4f  time=%.1fs  comm=%.2fMB",
        f["accuracy"],
        f"{f.get('auc', 0):.4f}" if "auc" in f else "N/A",
        f["loss"], f["train_time_s"], f.get("comm_cost_mb", 0),
    )
    log.info(
        "  Gap: accuracy=%+.4f  comm_ratio=%.2fx",
        comparison["accuracy_gap"], comparison["comm_cost_ratio"],
    )

    return comparison

def plot_benchmark(comparisons: list[dict], results_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scenarios  = [c["scenario"] for c in comparisons]
        central_accs = [c["centralized"]["accuracy"] for c in comparisons]
        fl_accs      = [c["federated"]["accuracy"]    for c in comparisons]
        x = range(len(scenarios))

        fig, ax = plt.subplots(figsize=(max(6, len(scenarios) * 2), 4))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        width = 0.35
        ax.bar([i - width/2 for i in x], central_accs, width,
               label="Centralised", color="#2d9cdb", alpha=0.85)
        ax.bar([i + width/2 for i in x], fl_accs, width,
               label="Federated",   color="#00c4b4", alpha=0.85)

        ax.set_xticks(list(x))
        ax.set_xticklabels(scenarios, color="#6e7681", fontsize=9)
        ax.set_ylabel("Accuracy", color="#6e7681")
        ax.set_ylim(0, 1.05)
        ax.set_title("FL vs Centralised Accuracy", color="#c9d1d9", fontsize=11)
        ax.tick_params(colors="#6e7681")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.legend(facecolor="#161b22", labelcolor="#c9d1d9", framealpha=0.8)

        out = Path(results_dir) / "benchmark.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Benchmark plot saved: %s", out)
    except Exception as e:
        log.warning("Could not save benchmark plot: %s", e)

def main() -> None:
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    scenarios = _BENCHMARK_SCENARIOS if args.all_scenarios else [args.scenario]
    comparisons = []

    for scenario in scenarios:
        log.info("=== Benchmarking: %s ===", scenario)
        try:
            result = run_one(
                scenario=scenario,
                central_epochs=args.central_epochs,
                results_dir=args.results_dir,
                configs_dir=args.configs_dir,
                seed=args.seed,
            )
            result["scenario"] = scenario
            comparisons.append(result)
        except Exception as e:
            log.error("Benchmark failed for scenario=%s: %s", scenario, e)

    if len(comparisons) > 1:
        plot_benchmark(comparisons, args.results_dir)

    log.info("Benchmark complete. Results → %s", Path(args.results_dir).resolve())

if __name__ == "__main__":
    main()