from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from src.data.preprocessing import preprocess_signal
from src.utils.logging import get_logger

log = get_logger(__name__)

@dataclass(slots=True)
class SensorPacket:

    device_id: str
    sensor: str = "heart_rate"
    samples: np.ndarray | list[float] = field(default_factory=list)
    labels: np.ndarray | list[float] | None = None
    sample_rate: float = 64.0
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RealtimeWindowProcessor:
    def __init__(
        self,
        *,
        window_size: int = 192,
        stride: int = 96,
        sampling_rate: float = 64.0,
        output_dir: str | Path = "data/realtime",
        apply_bandpass: bool = True,
        normalize_method: str = "z_score",
        low_hz: float = 0.5,
        high_hz: float = 4.0,
        filter_order: int = 4,
        label_aggregation: str = "max",
    ) -> None:
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.sampling_rate = float(sampling_rate)
        self.output_dir = Path(output_dir)
        self.apply_bandpass = bool(apply_bandpass)
        self.normalize_method = normalize_method
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.filter_order = int(filter_order)
        self.label_aggregation = label_aggregation

        self._signals: dict[tuple[str, str], list[float]] = {}
        self._labels: dict[tuple[str, str], list[float] | None] = {}
        self._lock = Lock()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self, packet: SensorPacket | dict[str, Any]) -> dict[str, Any]:
        packet = self._coerce_packet(packet)
        key = (packet.device_id, packet.sensor)

        with self._lock:
            buffer = self._signals.setdefault(key, [])
            buffer.extend(np.asarray(packet.samples, dtype=np.float32).tolist())

            if packet.labels is not None:
                label_buffer = self._labels.setdefault(key, [])
                label_buffer.extend(np.asarray(packet.labels, dtype=np.float32).tolist())
            else:
                self._labels.setdefault(key, None)

            result = self._materialize_snapshot(packet.device_id, packet.sensor)

        log.info(
            "Ingested packet device=%s sensor=%s samples=%d windows=%d",
            packet.device_id,
            packet.sensor,
            len(packet.samples),
            result["n_windows"],
        )
        return result

    def _materialize_snapshot(self, device_id: str, sensor: str) -> dict[str, Any]:
        key = (device_id, sensor)
        signal = np.asarray(self._signals.get(key, []), dtype=np.float32)
        labels_buffer = self._labels.get(key)
        labels = None if labels_buffer is None else np.asarray(labels_buffer, dtype=np.float32)

        if signal.size < self.window_size:
            return self._empty_snapshot(device_id, sensor, signal.size)

        windows, window_labels = preprocess_signal(
            signal,
            labels=labels,
            apply_bandpass=self.apply_bandpass,
            low_hz=self.low_hz,
            high_hz=self.high_hz,
            fs=self.sampling_rate,
            filter_order=self.filter_order,
            normalize_method=self.normalize_method,
            window_size=self.window_size,
            stride=self.stride,
            label_aggregation=self.label_aggregation,
        )

        self._save_snapshot(device_id, sensor, windows, window_labels)
        return {
            "device_id": device_id,
            "sensor": sensor,
            "n_samples": int(signal.size),
            "n_windows": int(len(windows)),
            "window_size": self.window_size,
            "stride": self.stride,
            "has_labels": window_labels is not None,
            "snapshot_dir": str(self._device_dir(device_id, sensor)),
        }

    def _empty_snapshot(self, device_id: str, sensor: str, n_samples: int) -> dict[str, Any]:
        device_dir = self._device_dir(device_id, sensor)
        device_dir.mkdir(parents=True, exist_ok=True)
        return {
            "device_id": device_id,
            "sensor": sensor,
            "n_samples": int(n_samples),
            "n_windows": 0,
            "window_size": self.window_size,
            "stride": self.stride,
            "has_labels": False,
            "snapshot_dir": str(device_dir),
        }

    def _save_snapshot(
        self,
        device_id: str,
        sensor: str,
        windows: np.ndarray,
        labels: np.ndarray | None,
    ) -> None:
        device_dir = self._device_dir(device_id, sensor)
        device_dir.mkdir(parents=True, exist_ok=True)
        np.save(device_dir / f"{sensor}_windows.npy", windows)
        if labels is not None:
            np.save(device_dir / f"{sensor}_labels.npy", labels)

    def _device_dir(self, device_id: str, sensor: str) -> Path:
        return self.output_dir / device_id / sensor

    @staticmethod
    def _coerce_packet(packet: SensorPacket | dict[str, Any]) -> SensorPacket:
        if isinstance(packet, SensorPacket):
            return packet
        return SensorPacket(
            device_id=str(packet["device_id"]),
            sensor=str(packet.get("sensor", "heart_rate")),
            samples=packet.get("samples", []),
            labels=packet.get("labels"),
            sample_rate=float(packet.get("sample_rate", 64.0)),
            timestamp=packet.get("timestamp"),
            metadata=dict(packet.get("metadata", {})),
        )