from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.partitioner import (
    dirichlet_partition,
    iid_partition,
    partition_windows_and_labels,
    save_partitions,
)
from src.data.preprocessing import preprocess_signal
from src.data.synthetic import generate_synthetic_signals
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare and partition data for FL simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "wesad", "ptbxl"],
        help="Dataset to prepare.",
    )
    p.add_argument("--n_clients",    type=int,   default=None,  help="Override num_clients from data.yaml.")
    p.add_argument("--n_samples",    type=int,   default=9600,  help="Samples per client (synthetic only).")
    p.add_argument("--seed",         type=int,   default=None,  help="Override RNG seed from data.yaml.")
    p.add_argument("--iid",          action="store_true",       help="Use IID partition (default: Dirichlet non-IID).")
    p.add_argument("--alpha",        type=float, default=None,  help="Dirichlet alpha (non-IID strength).")
    p.add_argument("--raw_dir",      type=str,   default="data/raw",          help="Raw data input directory.")
    p.add_argument("--processed_dir",type=str,   default="data/processed",    help="Processed .npy output directory.")
    p.add_argument("--partitions_dir",type=str,  default="data/partitions",   help="Per-client partition output directory.")
    p.add_argument("--synthetic_dir", type=str,  default="data/synthetic",    help="Synthetic data output directory.")
    p.add_argument("--configs_dir",  type=str,   default="configs",           help="Path to YAML config directory.")
    p.add_argument("--modality",     type=str,   default="heart_rate",
                   choices=["heart_rate", "spo2", "sleep"],
                   help="Signal modality to process.")
    return p.parse_args()

def prepare_synthetic(args: argparse.Namespace, data_cfg) -> tuple[np.ndarray, np.ndarray]:
    n_clients = args.n_clients or getattr(data_cfg.partitioning, "num_clients", 20)
    seed      = args.seed      or getattr(data_cfg.partitioning, "seed", 42)

    log.info("Generating synthetic data: %d clients × %d samples  modality=%s",
             n_clients, args.n_samples, args.modality)

    datasets = generate_synthetic_signals(
        n_clients=n_clients,
        n_samples=args.n_samples,
        modality=args.modality,
        anomaly_ratio=getattr(data_cfg.labels, "positive_class_ratio", 0.15),
        seed=seed,
        save_dir=args.synthetic_dir,
    )
    
    sampling_rate = getattr(data_cfg.signals, "sampling_rate", 64)
    window_size   = getattr(data_cfg.preprocessing, "window_size", 30) * sampling_rate
    stride        = getattr(data_cfg.preprocessing, "stride", 15)      * sampling_rate
    normalize_method = getattr(data_cfg.preprocessing, "normalization", "z_score")
    apply_bandpass   = (args.modality != "sleep")

    all_windows: list[np.ndarray] = []
    all_labels:  list[np.ndarray] = []

    for ds in datasets:
        windows, labels = preprocess_signal(
            signal=ds.windows,
            labels=ds.labels,
            apply_bandpass=apply_bandpass,
            normalize_method=normalize_method,
            window_size=window_size,
            stride=stride,
        )
        all_windows.append(windows)
        all_labels.append(labels)

    stacked_windows = np.vstack(all_windows)
    stacked_labels  = np.concatenate(all_labels)

    proc_dir = Path(args.processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    np.save(proc_dir / f"{args.modality}_windows.npy", stacked_windows)
    np.save(proc_dir / f"{args.modality}_labels.npy",  stacked_labels)
    log.info("Processed arrays saved: %s  shape=%s", proc_dir, stacked_windows.shape)

    return stacked_windows, stacked_labels

def prepare_wesad(args: argparse.Namespace, data_cfg) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "WESAD preprocessing not yet implemented.\n"
        "Download instructions: data/raw/README.md\n"
        "Use --dataset synthetic for now."
    )

def prepare_ptbxl(args: argparse.Namespace, data_cfg) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "PTB-XL preprocessing not yet implemented.\n"
        "Download instructions: data/raw/README.md\n"
        "Use --dataset synthetic for now."
    )

def partition_data(
    windows: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    data_cfg,
) -> None:
    n_clients = args.n_clients or getattr(data_cfg.partitioning, "num_clients", 20)
    seed      = args.seed      or getattr(data_cfg.partitioning, "seed", 42)
    alpha     = args.alpha     or getattr(data_cfg.partitioning, "dirichlet_alpha", 0.5)
    use_iid   = args.iid       or getattr(data_cfg.partitioning, "iid", False)

    n_windows = len(windows)
    log.info(
        "Partitioning %d windows across %d clients  strategy=%s  alpha=%s",
        n_windows, n_clients,
        "iid" if use_iid else f"dirichlet(α={alpha})",
        alpha if not use_iid else "N/A",
    )

    if use_iid:
        partitions = iid_partition(n_windows, n_clients, seed=seed)
    else:
        partitions = dirichlet_partition(
            labels=labels,
            n_clients=n_clients,
            alpha=alpha,
            min_samples=getattr(data_cfg.partitioning, "min_samples_per_client", 50),
            seed=seed,
        )

    partitions_dir = Path(args.partitions_dir)
    save_partitions(partitions, partitions_dir / "indices")
    partition_windows_and_labels(windows, labels, partitions, partitions_dir)

    log.info(
        "Partitions saved: %d clients → %s",
        n_clients, partitions_dir.resolve(),
    )

def main() -> None:
    args     = parse_args()
    data_cfg = load_config(Path(args.configs_dir) / "data.yaml")
    seed     = args.seed or getattr(data_cfg.partitioning, "seed", 42)
    set_seed(seed)

    log.info("=== prepare_data: dataset=%s  modality=%s ===", args.dataset, args.modality)

    if args.dataset == "synthetic":
        windows, labels = prepare_synthetic(args, data_cfg)
    elif args.dataset == "wesad":
        windows, labels = prepare_wesad(args, data_cfg)
    elif args.dataset == "ptbxl":
        windows, labels = prepare_ptbxl(args, data_cfg)
    else:
        log.error("Unknown dataset: %s", args.dataset)
        sys.exit(1)

    partition_data(windows, labels, args, data_cfg)
    log.info("=== Data preparation complete ===")

if __name__ == "__main__":
    main()