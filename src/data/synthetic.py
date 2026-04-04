from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)

class SyntheticDataset(NamedTuple):
    windows: np.ndarray      
    labels: np.ndarray      
    client_id: int
    modality: str

def _generate_heart_rate(
    n_samples: int,
    rng: np.random.Generator,
    base_hr: float = 70.0,
    std: float = 8.0,
) -> np.ndarray:
    t = np.linspace(0, 24 * np.pi, n_samples)
    circadian = 5.0 * np.sin(t / 12)                       
    ultradian = 2.0 * np.sin(t / 1.5)                      
    noise = rng.normal(0, std * 0.3, n_samples)
    signal = base_hr + circadian + ultradian + noise
    return signal.astype(np.float32)

def _generate_spo2(
    n_samples: int,
    rng: np.random.Generator,
    base_spo2: float = 97.5,
) -> np.ndarray:
    signal = rng.normal(base_spo2, 0.5, n_samples)
    n_drops = rng.integers(2, 8)
    for _ in range(n_drops):
        start = rng.integers(0, n_samples - 30)
        duration = rng.integers(10, 30)
        drop = rng.uniform(3.0, 8.0)
        signal[start:start + duration] -= drop
    signal = np.clip(signal, 70.0, 100.0)
    return signal.astype(np.float32)

def _generate_sleep_stages(
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    cycle = np.array([0, 1, 1, 2, 2, 2, 3, 3, 1, 1], dtype=np.int64)
    repeats = n_samples // len(cycle) + 1
    full = np.tile(cycle, repeats)[:n_samples]

    noise_mask = rng.random(n_samples) < 0.05
    full[noise_mask] = rng.integers(0, 4, noise_mask.sum())
    return full.astype(np.float32)

def _inject_anomalies(
    signal: np.ndarray,
    rng: np.random.Generator,
    anomaly_ratio: float = 0.15,
    anomaly_magnitude: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:

    n = len(signal)
    mask = np.zeros(n, dtype=np.float32)
    n_anomalies = int(n * anomaly_ratio)

    positions = rng.choice(n, size=n_anomalies, replace=False)
    anomaly_type = rng.integers(0, 2, size=n_anomalies)  

    signal = signal.copy()
    std = np.std(signal)

    for pos, atype in zip(positions, anomaly_type):
        if atype == 0:  # spike
            signal[pos] += rng.uniform(anomaly_magnitude, anomaly_magnitude * 2) * std
        else:            # dropout
            signal[pos] = signal[pos] * rng.uniform(0.1, 0.3)
        mask[pos] = 1.0

    return signal, mask

def generate_synthetic_signals(
    n_clients: int = 20,
    n_samples: int = 9600,           
    modality: str = "heart_rate",
    anomaly_ratio: float = 0.15,
    client_heterogeneity: float = 0.3,  
    seed: int = 42,
    save_dir: str | Path | None = None,
) -> list[SyntheticDataset]:

    set_seed(seed)
    global_rng = np.random.default_rng(seed)

    hr_offsets = global_rng.normal(0, client_heterogeneity * 10, n_clients)
    spo2_offsets = global_rng.normal(0, client_heterogeneity * 1.5, n_clients)

    datasets: list[SyntheticDataset] = []

    for cid in range(n_clients):
        client_rng = np.random.default_rng(seed + cid * 1000)

        if modality == "heart_rate":
            raw = _generate_heart_rate(
                n_samples, client_rng,
                base_hr=70.0 + hr_offsets[cid],
            )
        elif modality == "spo2":
            raw = _generate_spo2(
                n_samples, client_rng,
                base_spo2=97.5 + spo2_offsets[cid],
            )
        elif modality == "sleep":
            raw = _generate_sleep_stages(n_samples, client_rng)
        else:
            raise ValueError(f"Unknown modality: {modality}. Choose: heart_rate | spo2 | sleep")

        signal, anomaly_mask = _inject_anomalies(raw, client_rng, anomaly_ratio)
        datasets.append(SyntheticDataset(
            windows=signal,
            labels=anomaly_mask,
            client_id=cid,
            modality=modality,
        ))

    log.info(
        "Generated synthetic %s signals: %d clients × %d samples, anomaly_ratio=%.2f",
        modality, n_clients, n_samples, anomaly_ratio,
    )
    if save_dir is not None:
        _save_to_disk(datasets, Path(save_dir), modality)

    return datasets

def _save_to_disk(
    datasets: list[SyntheticDataset],
    save_dir: Path,
    modality: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    all_windows = np.stack([d.windows for d in datasets])   # (n_clients, n_samples)
    all_labels  = np.stack([d.labels  for d in datasets])   # (n_clients, n_samples)

    win_path = save_dir / f"{modality}.npy"
    lbl_path = save_dir / f"{modality}_labels.npy"
    np.save(win_path, all_windows)
    np.save(lbl_path, all_labels)
    log.info("Saved: %s  %s", win_path, lbl_path)