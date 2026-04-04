from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root: python data/synthetic/generate.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.synthetic import generate_synthetic_signals
from src.utils.logging import get_logger

log = get_logger(__name__)

SAVE_DIR = Path(__file__).parent

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic physiological signals.")
    p.add_argument("--n_clients",    type=int,   default=20,   help="Number of virtual clients")
    p.add_argument("--n_samples",    type=int,   default=9600, help="Samples per client")
    p.add_argument("--anomaly_ratio",type=float, default=0.15, help="Fraction of anomaly samples")
    p.add_argument("--seed",         type=int,   default=42,   help="RNG seed")
    p.add_argument("--save_dir",     type=str,   default=str(SAVE_DIR), help="Output directory")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating synthetic signals: %d clients × %d samples", args.n_clients, args.n_samples)

    for modality in ("heart_rate", "spo2", "sleep"):
        log.info("  Modality: %s", modality)
        generate_synthetic_signals(
            n_clients=args.n_clients,
            n_samples=args.n_samples,
            modality=modality,
            anomaly_ratio=args.anomaly_ratio,
            seed=args.seed,
            save_dir=save_dir,
        )

    log.info("Done. Files saved to: %s", save_dir.resolve())
    log.info("Files:")
    for f in sorted(save_dir.glob("*.npy")):
        size_mb = f.stat().st_size / (1024 ** 2)
        log.info("  %-40s  %.2f MB", f.name, size_mb)

if __name__ == "__main__":
    main()