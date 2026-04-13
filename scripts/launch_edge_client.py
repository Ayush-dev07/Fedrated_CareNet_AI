from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import HealthDataset
from src.fl.client.client import HealthClient
from src.models.factory import get_loss_fn, get_model, get_optimizer
from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a real-time FL edge client.")
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--sensor", default="heart_rate")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--realtime-dir", default="data/realtime")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--fl-config", default="configs/fl.yaml")
    parser.add_argument("--privacy-config", default="configs/privacy.yaml")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser.parse_args()

def _load_snapshot(realtime_dir: Path, device_id: str, sensor: str) -> Tuple[np.ndarray, np.ndarray]:
    device_dir = realtime_dir / device_id / sensor
    windows_path = device_dir / f"{sensor}_windows.npy"
    labels_path = device_dir / f"{sensor}_labels.npy"
    windows = np.load(windows_path)
    if labels_path.exists():
        labels = np.load(labels_path)
    else:
        labels = np.zeros(len(windows), dtype=np.int64)
    return windows, labels

def _wait_for_snapshot(realtime_dir: Path, device_id: str, sensor: str, timeout_seconds: float, poll_seconds: float) -> Tuple[np.ndarray, np.ndarray]:
    start = time.time()
    while True:
        try:
            return _load_snapshot(realtime_dir, device_id, sensor)
        except FileNotFoundError:
            if time.time() - start > timeout_seconds:
                raise TimeoutError(
                    f"No realtime snapshot found for device={device_id}, sensor={sensor} within {timeout_seconds} seconds"
                )
            time.sleep(poll_seconds)

def _make_loaders(windows: np.ndarray, labels: np.ndarray, batch_size: int = 32, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    dataset = HealthDataset(windows=windows, labels=labels)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size, shuffle=False)

class LiveHealthClient(HealthClient):
    def __init__(
        self,
        *args,
        realtime_dir: Path,
        device_id: str,
        sensor: str,
        poll_seconds: float,
        data_cfg: SimpleNamespace,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.realtime_dir = realtime_dir
        self.device_id = device_id
        self.sensor = sensor
        self.poll_seconds = poll_seconds
        self.data_cfg = data_cfg

    def _refresh_loaders(self) -> None:
        windows, labels = _load_snapshot(self.realtime_dir, self.device_id, self.sensor)
        train_cfg = getattr(self.fl_cfg, "local_training", SimpleNamespace(batch_size=32))
        train_loader, val_loader = _make_loaders(
            windows,
            labels,
            batch_size=int(getattr(train_cfg, "batch_size", 32)),
            val_split=float(getattr(getattr(self.data_cfg, "partitioning", SimpleNamespace()), "val_split", 0.2)),
        )
        self.train_loader = train_loader
        self.val_loader = val_loader

    def fit(self, parameters, config):
        self._refresh_loaders()
        return super().fit(parameters, config)

    def evaluate(self, parameters, config):
        self._refresh_loaders()
        return super().evaluate(parameters, config)

if __name__ == "__main__":
    args = parse_args()
    realtime_dir = Path(args.realtime_dir)

    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    fl_cfg = load_config(args.fl_config)
    privacy_cfg = load_config(args.privacy_config)

    windows, labels = _wait_for_snapshot(
        realtime_dir=realtime_dir,
        device_id=args.device_id,
        sensor=args.sensor,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )

    train_loader, val_loader = _make_loaders(
        windows,
        labels,
        batch_size=int(getattr(fl_cfg.local_training, "batch_size", 32)),
        val_split=float(getattr(data_cfg.partitioning, "val_split", 0.2)),
    )

    client = LiveHealthClient(
        client_id=0,
        train_loader=train_loader,
        val_loader=val_loader,
        model_cfg=model_cfg,
        privacy_cfg=privacy_cfg,
        fl_cfg=fl_cfg,
        realtime_dir=realtime_dir,
        device_id=args.device_id,
        sensor=args.sensor,
        poll_seconds=args.poll_seconds,
        data_cfg=data_cfg,
    )

    import flwr as fl

    log.info("Connecting edge client for device=%s to %s", args.device_id, args.server_address)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
