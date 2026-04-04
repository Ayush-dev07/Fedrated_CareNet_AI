from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.logging import get_logger
log = get_logger(__name__)

class HealthDataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        client_id: Optional[int] = None,
        transform=None,
    ) -> None:

        if windows.ndim != 2:
            raise ValueError(f"windows must be 2-D (n_windows, window_size), got {windows.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1-D (n_windows,), got {labels.shape}")
        if len(windows) != len(labels):
            raise ValueError(
                f"windows/labels length mismatch: {len(windows)} vs {len(labels)}"
            )

        self.windows   = torch.from_numpy(windows.astype(np.float32))   # (N, W)
        self.labels    = torch.from_numpy(labels.astype(np.int64))       # (N,)
        self.client_id = client_id
        self.transform = transform

        log.debug(
            "HealthDataset: client=%s  n_windows=%d  window_size=%d  "
            "pos_rate=%.3f",
            client_id, len(self), self.window_size,
            self.labels.float().mean().item(),
        )

    @property
    def n_windows(self) -> int:
        return self.windows.shape[0]

    @property
    def window_size(self) -> int:
        return self.windows.shape[1]

    @property
    def positive_rate(self) -> float:
        return self.labels.float().mean().item()

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        x = self.windows[idx].unsqueeze(0)   
        y = self.labels[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    @classmethod
    def from_files(
        cls,
        windows_path: str | Path,
        labels_path: str | Path,
        client_id: Optional[int] = None,
        transform=None,
    ) -> "HealthDataset":

        windows_path = Path(windows_path)
        labels_path  = Path(labels_path)

        if not windows_path.exists():
            raise FileNotFoundError(f"Windows file not found: {windows_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        windows = np.load(windows_path)
        labels  = np.load(labels_path)
        log.info("Loaded dataset from %s + %s", windows_path.name, labels_path.name)

        return cls(windows, labels, client_id=client_id, transform=transform)

    @classmethod
    def from_client_partition(
        cls,
        partitions_dir: str | Path,
        client_id: int,
        transform=None,
    ) -> "HealthDataset":

        base = Path(partitions_dir)
        return cls.from_files(
            windows_path=base / f"client_{client_id}_windows.npy",
            labels_path=base  / f"client_{client_id}_labels.npy",
            client_id=client_id,
            transform=transform,
        )

    def split(self, val_fraction: float = 0.2, seed: int = 42) -> tuple["HealthDataset", "HealthDataset"]:
        n = len(self)
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val

        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
        train_idx = indices[:n_train]
        val_idx   = indices[n_train:]

        train_ds = HealthDataset(
            self.windows[train_idx].numpy(),
            self.labels[train_idx].numpy(),
            client_id=self.client_id,
        )
        val_ds = HealthDataset(
            self.windows[val_idx].numpy(),
            self.labels[val_idx].numpy(),
            client_id=self.client_id,
        )
        return train_ds, val_ds

    def __repr__(self) -> str:
        return (
            f"HealthDataset(client={self.client_id}, "
            f"n_windows={self.n_windows}, "
            f"window_size={self.window_size}, "
            f"pos_rate={self.positive_rate:.3f})"
        )