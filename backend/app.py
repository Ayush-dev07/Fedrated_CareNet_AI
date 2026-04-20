from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.stream_processor import RealtimeWindowProcessor, SensorPacket
from src.utils.logging import get_logger

log = get_logger(__name__)


class SamplePoint(BaseModel):
    value: float
    timestamp: float | None = None


class IngestRequest(BaseModel):
    device_id: str
    sensor: str = "heart_rate"
    samples: list[float | SamplePoint] = Field(default_factory=list)
    labels: list[float] | None = None
    sample_rate: float = 64.0
    timestamp: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AddDeviceRequest(BaseModel):
    id: str
    name: str
    sensorType: str = "Heart Rate"


class DeviceView(BaseModel):
    id: str
    name: str
    status: Literal["online", "offline"]
    sensorType: str
    lastReading: float | None = None
    anomalyCount: int = 0
    readingCount: int = 0


class MetricsHistoryPoint(BaseModel):
    timestamp: str
    value: float
    anomalies: int


@dataclass(slots=True)
class DeviceState:
    id: str
    name: str
    sensor_type: str
    last_reading_ts: float | None = None
    last_value: float | None = None
    anomaly_count: int = 0
    reading_count: int = 0
    window_count: int = 0


class IngestionService:

    def __init__(self, processor: RealtimeWindowProcessor | None = None) -> None:
        self.processor = processor or RealtimeWindowProcessor()
        self.latest_snapshots: dict[str, dict[str, Any]] = {}
        self.devices: dict[str, DeviceState] = {}
        self.metrics_history: deque[MetricsHistoryPoint] = deque(maxlen=24 * 60)
        self.started_at = time.time()

    @staticmethod
    def _extract_samples(samples: list[float | SamplePoint]) -> tuple[list[float], float | None]:
        values: list[float] = []
        latest_ts: float | None = None
        for sample in samples:
            if isinstance(sample, (int, float)):
                values.append(float(sample))
                continue
            values.append(float(sample.value))
            if sample.timestamp is not None:
                latest_ts = sample.timestamp
        return values, latest_ts

    def register_device(self, payload: AddDeviceRequest) -> DeviceView:
        state = self.devices.get(payload.id)
        if state is None:
            state = DeviceState(
                id=payload.id,
                name=payload.name,
                sensor_type=payload.sensorType,
            )
            self.devices[payload.id] = state
        else:
            state.name = payload.name
            state.sensor_type = payload.sensorType
        return self._to_device_view(state)

    def ingest(self, payload: IngestRequest) -> dict[str, Any]:
        sample_values, sample_ts = self._extract_samples(payload.samples)
        packet = SensorPacket(
            device_id=payload.device_id,
            sensor=payload.sensor,
            samples=sample_values,
            labels=payload.labels,
            sample_rate=payload.sample_rate,
            timestamp=payload.timestamp,
            metadata=payload.metadata,
        )

        result = self.processor.ingest(packet)
        now_ts = time.time()
        last_ts = payload.timestamp or sample_ts or now_ts
        last_value = sample_values[-1] if sample_values else None

        device = self.devices.get(payload.device_id)
        if device is None:
            device = DeviceState(
                id=payload.device_id,
                name=f"Device {payload.device_id}",
                sensor_type=payload.sensor.replace("_", " ").title(),
            )
            self.devices[payload.device_id] = device

        device.last_reading_ts = last_ts
        device.last_value = last_value
        device.reading_count += len(sample_values)
        device.window_count = int(result.get("n_windows", device.window_count))
        if payload.labels:
            device.anomaly_count += int(sum(1 for label in payload.labels if float(label) > 0.0))

        enriched = {
            **result,
            "status": "online",
            "last_reading": {
                "timestamp": last_ts,
                "value": last_value,
                "device_id": payload.device_id,
                "sensor_type": payload.sensor,
            },
            "reading_count": device.reading_count,
            "anomaly_count": device.anomaly_count,
        }
        self.latest_snapshots[payload.device_id] = enriched
        self._update_metrics_history(now_ts)
        return enriched

    def _to_device_view(self, state: DeviceState) -> DeviceView:
        now_ts = time.time()
        status: Literal["online", "offline"] = "online"
        if state.last_reading_ts is not None and (now_ts - state.last_reading_ts) > 120:
            status = "offline"
        return DeviceView(
            id=state.id,
            name=state.name,
            status=status,
            sensorType=state.sensor_type,
            lastReading=state.last_reading_ts,
            anomalyCount=state.anomaly_count,
            readingCount=state.reading_count,
        )

    def get_latest(self, device_id: str) -> dict[str, Any] | None:
        return self.latest_snapshots.get(device_id)

    def list_devices(self) -> list[DeviceView]:
        return [self._to_device_view(device) for device in self.devices.values()]

    def _update_metrics_history(self, timestamp: float) -> None:
        local_time = time.localtime(timestamp)
        point = MetricsHistoryPoint(
            timestamp=f"{local_time.tm_hour:02d}:00",
            value=70.0 + (len(self.devices) * 0.8),
            anomalies=sum(device.anomaly_count for device in self.devices.values()),
        )
        self.metrics_history.append(point)

    def get_metrics_history(self, hours: int) -> list[MetricsHistoryPoint]:
        if not self.metrics_history:
            now_hour = int(time.strftime("%H"))
            synthetic = [
                MetricsHistoryPoint(
                    timestamp=f"{(now_hour - i) % 24:02d}:00",
                    value=72.0,
                    anomalies=0,
                )
                for i in range(min(max(hours, 1), 24) - 1, -1, -1)
            ]
            return synthetic
        return list(self.metrics_history)[-min(max(hours, 1), 24) :]

    def get_metrics(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "devices_seen": len(self.devices),
            "snapshots_cached": len(self.latest_snapshots),
            "uptime_seconds": int(time.time() - self.started_at),
        }

    def get_models(self) -> list[dict[str, Any]]:
        return [
            {"id": "lstm_v1", "name": "LSTM Anomaly Detector", "accuracy": 94.2, "f1": 0.89, "auc": 0.96, "loss": 0.15},
            {"id": "ae_v2", "name": "Autoencoder", "accuracy": 91.8, "f1": 0.85, "auc": 0.94, "loss": 0.22},
            {"id": "ensemble_v1", "name": "Ensemble Model", "accuracy": 96.1, "f1": 0.92, "auc": 0.98, "loss": 0.10},
        ]

    def get_model_training(self, model_id: str) -> dict[str, Any]:
        if model_id not in {"lstm_v1", "ae_v2", "ensemble_v1"}:
            raise KeyError(model_id)
        epochs = [
            {
                "epoch": idx,
                "trainLoss": max(0.06, 0.5 - (idx * 0.008)),
                "valLoss": max(0.08, 0.48 - (idx * 0.007)),
                "accuracy": min(0.99, 0.6 + (idx * 0.007)),
            }
            for idx in range(1, 51)
        ]
        return {"modelId": model_id, "epochs": epochs}

    def get_anomalies(self, period: str) -> dict[str, Any]:
        _ = period
        points = [
            {"name": "Mon", "count": 12},
            {"name": "Tue", "count": 8},
            {"name": "Wed", "count": 15},
            {"name": "Thu", "count": 10},
            {"name": "Fri", "count": 18},
            {"name": "Sat", "count": 6},
            {"name": "Sun", "count": 9},
        ]
        return {"period": period, "points": points}

    def get_quality(self) -> dict[str, Any]:
        return {
            "completeness": [
                {"sensor": "Heart Rate", "value": 95},
                {"sensor": "SpO2", "value": 93},
                {"sensor": "ECG", "value": 91},
            ],
            "issues": {
                "missing_values": 2.3,
                "outliers": 1.8,
                "duplicates": 0.5,
            },
            "processing": {
                "avg_latency_ms": 24,
                "p95_latency_ms": 87,
                "throughput_per_sec": 1200,
            },
        }


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app(*, processor: RealtimeWindowProcessor | None = None) -> FastAPI:
    app = FastAPI(
        title="CareNet AI Backend",
        version="1.0.0",
        description="FastAPI backend for Federated CareNet AI frontend integration.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = IngestionService(processor=processor)

    @app.get("/")
    def root() -> dict[str, str]:
        return {"status": "ok", "message": "CareNet AI backend is running"}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.get_metrics()

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return service.get_metrics()

    @app.get("/metrics/history")
    def metrics_history(hours: int = Query(default=24, ge=1, le=24)) -> list[MetricsHistoryPoint]:
        return service.get_metrics_history(hours)

    @app.get("/devices")
    def list_devices() -> list[DeviceView]:
        return service.list_devices()

    @app.post("/devices", status_code=201)
    def add_device(payload: AddDeviceRequest) -> DeviceView:
        return service.register_device(payload)

    @app.get("/devices/{device_id}/latest")
    def device_latest(device_id: str) -> dict[str, Any]:
        latest = service.get_latest(device_id)
        if latest is None:
            raise HTTPException(status_code=404, detail="device not found")
        return latest

    @app.post("/ingest", status_code=202)
    def ingest(payload: IngestRequest) -> dict[str, Any]:
        try:
            return service.ingest(payload)
        except Exception as exc:  # pragma: no cover - defensive guard
            log.exception("Failed to ingest packet")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/models")
    def models() -> list[dict[str, Any]]:
        return service.get_models()

    @app.get("/models/{model_id}/training")
    def model_training(model_id: str) -> dict[str, Any]:
        try:
            return service.get_model_training(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model not found") from exc

    @app.get("/analytics/anomalies")
    def analytics_anomalies(period: str = Query(default="7d")) -> dict[str, Any]:
        return service.get_anomalies(period)

    @app.get("/analytics/quality")
    def analytics_quality() -> dict[str, Any]:
        return service.get_quality()

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    *,
    processor: RealtimeWindowProcessor | None = None,
) -> None:
    app = create_app(processor=processor)
    log.info("Starting FastAPI ingestion server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


app = create_app()