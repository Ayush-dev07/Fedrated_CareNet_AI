from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import get_dataloader
from src.evaluation.privacy_audit import compare_dp_vs_nodp_leakage, shadow_model_attack
from src.fl.simulation.scenarios import get_scenario
from src.fl.simulation.simulator import run_simulation
from src.models.factory import get_model
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.metrics import MetricsTracker
from src.utils.seed import set_seed

log = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run membership inference privacy audit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scenario",      type=str, default="noniid_20", help="FL scenario to audit.")
    p.add_argument("--client_id",     type=int, default=0,           help="Client whose data to use as members.")
    p.add_argument("--shadow_models", type=int, default=5,           help="Number of shadow models.")
    p.add_argument("--shadow_epochs", type=int, default=5,           help="Shadow model training epochs.")
    p.add_argument("--attack_epochs", type=int, default=10,          help="Attack classifier training epochs.")
    p.add_argument("--results_dir",   type=str, default="results",   help="Output directory.")
    p.add_argument("--configs_dir",   type=str, default="configs",   help="YAML config directory.")
    p.add_argument("--seed",          type=int, default=42,          help="RNG seed.")
    p.add_argument(
        "--compare_dp",
        action="store_true",
        help="Run audit on both DP and no-DP variants of the scenario.",
    )
    return p.parse_args()

def _build_model_and_loaders(sim_cfg, client_id: int, configs_dir: str):
    data_cfg  = sim_cfg.data_cfg
    model_cfg = sim_cfg.model_cfg

    partitions_dir = Path(getattr(data_cfg.dataset, "partitions_dir", "data/partitions"))
    if not (partitions_dir / f"client_{client_id}_windows.npy").exists():
        log.warning(
            "Partitioned data not found in %s\n"
            "Please run: python scripts/prepare_data.py --dataset synthetic\n"
            "Then try again.",
            partitions_dir.resolve(),
        )
        sys.exit(1)

    member_loader     = get_dataloader(client_id,     data_cfg, split="train")
    non_member_loader = get_dataloader(client_id + 1, data_cfg, split="train")   # different client

    seq_len = None
    try:
        x, _ = next(iter(member_loader))
        seq_len = x.shape[-1]
    except Exception:
        pass

    log.info("Running short simulation to get trained model for audit...")
    sim_cfg.num_rounds                   = 5   # quick — just need a trained model
    sim_cfg.fl_cfg.rounds.num_rounds     = 5

    _ = run_simulation(sim_cfg, results_dir="results/audit_sim_tmp")
    model = get_model(model_cfg, seq_len=seq_len)
    return model, member_loader, non_member_loader

def print_report(results: dict, label: str = "Audit") -> None:
    print(f"\n{'='*60}")
    print(f"  Privacy Audit Report — {label}")
    print(f"{'='*60}")
    print(f"  Attack AUC          : {results['attack_auc']:.4f}  (0.5=no leak, 1.0=full leak)")
    print(f"  Attack Accuracy     : {results['attack_accuracy']:.4f}")
    print(f"  Advantage (AUC-0.5) : {results['attack_advantage']:.4f}")
    print(f"  Member Confidence   : {results['member_confidence']:.4f}")
    print(f"  Non-member Conf.    : {results['nonmember_confidence']:.4f}")
    print(f"  Confidence Gap      : {results['confidence_gap']:.4f}")
    print(f"  ε empirical         : {results['epsilon_empirical']:.4f}")
    print(f"  Shadow models used  : {results['n_shadow_models']}")
    print(f"  Duration            : {results['duration_seconds']:.1f}s")

    advantage = results["attack_advantage"]
    if advantage < 0.02:
        verdict = "✓ LOW leakage — model appears well-protected."
    elif advantage < 0.10:
        verdict = "⚠ MODERATE leakage — some membership information exposed."
    else:
        verdict = "✗ HIGH leakage — significant privacy risk detected."

    print(f"\n  Verdict: {verdict}")
    print(f"{'='*60}\n")

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    tracker = MetricsTracker(Path(args.results_dir) / "privacy_audit.csv")

    if args.compare_dp:
        log.info("=== No-DP audit ===")
        nodp_cfg = get_scenario("noniid_20", configs_dir=args.configs_dir)
        nodp_model, member_loader, nonmember_loader = _build_model_and_loaders(
            nodp_cfg, args.client_id, args.configs_dir
        )
        nodp_results = shadow_model_attack(
            nodp_model, member_loader, nonmember_loader,
            n_shadow_models=args.shadow_models,
            shadow_train_epochs=args.shadow_epochs,
            attack_train_epochs=args.attack_epochs,
            seed=args.seed,
        )

        log.info("=== DP audit ===")
        dp_cfg = get_scenario("dp_enabled", configs_dir=args.configs_dir)
        dp_model, member_loader, nonmember_loader = _build_model_and_loaders(
            dp_cfg, args.client_id, args.configs_dir
        )
        dp_results = shadow_model_attack(
            dp_model, member_loader, nonmember_loader,
            n_shadow_models=args.shadow_models,
            shadow_train_epochs=args.shadow_epochs,
            attack_train_epochs=args.attack_epochs,
            seed=args.seed,
        )

        print_report(nodp_results, label="No-DP model")
        print_report(dp_results,   label="DP model (ε=3.0)")

        improvement = nodp_results["attack_auc"] - dp_results["attack_auc"]
        print(f"\n  DP Privacy Improvement (ΔAUC): {improvement:+.4f}")
        print(f"  (Positive = DP reduced attack success)\n")

        tracker.log({"run": "nodp", **nodp_results})
        tracker.log({"run": "dp",   **dp_results})
        tracker.log({"run": "comparison", "auc_improvement": round(improvement, 6)})

    else:
        log.info("=== Audit: %s ===", args.scenario)
        sim_cfg = get_scenario(args.scenario, configs_dir=args.configs_dir)
        model, member_loader, nonmember_loader = _build_model_and_loaders(
            sim_cfg, args.client_id, args.configs_dir
        )
        results = shadow_model_attack(
            model, member_loader, nonmember_loader,
            n_shadow_models=args.shadow_models,
            shadow_train_epochs=args.shadow_epochs,
            attack_train_epochs=args.attack_epochs,
            seed=args.seed,
        )
        print_report(results, label=f"Scenario: {args.scenario}")
        tracker.log({"run": args.scenario, **results})

    tracker.close()
    log.info("Audit results saved → %s", Path(args.results_dir).resolve())

if __name__ == "__main__":
    main()