from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader, random_split

from src.data.dataset import HealthDataset
from src.data.preprocessing import preprocess_signal
from src.data.synthetic import generate_synthetic_signals
from src.utils.logging import get_logger

log = get_logger(__name__)

def get_dataloader(
    client_id: int,
    config: SimpleNamespace,
    split: str = "train",
    shuffle: bool | None = None,
) -> DataLoader:

    if shuffle is None:
        shuffle = (split == "train")

    dataset = _resolve_dataset(client_id, config)
    val_fraction = getattr(config.partitioning, "val_split", 0.2)
    seed = getattr(config.partitioning, "seed", 42)

    train_ds, val_ds = dataset.split(val_fraction=val_fraction, seed=seed)
    active_ds = train_ds if split == "train" else val_ds
    batch_size = getattr(config, "batch_size", 32)
    num_workers = 0   

    loader = DataLoader(
        active_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=(split == "train" and len(active_ds) > batch_size),
        pin_memory=False,
    )

    log.debug(
        "DataLoader: client=%d  split=%s  n_samples=%d  batch_size=%d  n_batches=%d",
        client_id, split, len(active_ds), batch_size, len(loader),
    )
    return loader

def get_dataloader_from_arrays(
    windows: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    client_id: Optional[int] = None,
) -> DataLoader:

    dataset = HealthDataset(windows, labels, client_id=client_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=shuffle and len(dataset) > batch_size,
        pin_memory=False,
    )

def get_all_dataloaders(
    config: SimpleNamespace,
    split: str = "train",
) -> dict[int, DataLoader]:

    n_clients = getattr(config.partitioning, "num_clients", 20)
    return {
        cid: get_dataloader(cid, config, split=split)
        for cid in range(n_clients)
    }

def _resolve_dataset(client_id: int, config: SimpleNamespace) -> HealthDataset:

    partitions_dir = Path(getattr(config.dataset, "partitions_dir", "data/partitions"))
    windows_path = partitions_dir / f"client_{client_id}_windows.npy"
    labels_path  = partitions_dir / f"client_{client_id}_labels.npy"

    if windows_path.exists() and labels_path.exists():
        log.debug("Loading client %d from disk: %s", client_id, partitions_dir)
        return HealthDataset.from_files(windows_path, labels_path, client_id=client_id)

    log.debug("Partition files not found — generating synthetic data for client %d", client_id)
    return _synthetic_dataset(client_id, config)

def _synthetic_dataset(client_id: int, config: SimpleNamespace) -> HealthDataset:

    n_samples = getattr(config, "n_samples_per_client", 9600)
    modality  = getattr(config.signals, "modalities", ["heart_rate"])[0]
    seed      = getattr(config.partitioning, "seed", 42)
    anomaly_ratio = getattr(config.labels, "positive_class_ratio", 0.15)

    from src.data.synthetic import _generate_heart_rate, _inject_anomalies
    client_rng = np.random.default_rng(seed + client_id * 1000)

    raw = _generate_heart_rate(n_samples, client_rng)
    signal, labels = _inject_anomalies(raw, client_rng, anomaly_ratio)

    window_size = getattr(config.preprocessing, "window_size", 30) * getattr(config.signals, "sampling_rate", 64)
    stride      = getattr(config.preprocessing, "stride", 15)      * getattr(config.signals, "sampling_rate", 64)

    windows, window_labels = preprocess_signal(
        signal, labels,
        apply_bandpass=True,
        normalize_method=getattr(config.preprocessing, "normalization", "z_score"),
        window_size=window_size,
        stride=stride,
    )

    return HealthDataset(windows, window_labels, client_id=client_id)