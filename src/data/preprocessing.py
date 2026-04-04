from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.utils.logging import get_logger

log = get_logger(__name__)

def bandpass_filter(
    signal: np.ndarray,
    low_hz: float = 0.5,
    high_hz: float = 4.0,
    fs: float = 64.0,
    order: int = 4,
) -> np.ndarray:

    if signal.ndim != 1:
        raise ValueError(f"Expected 1-D signal, got shape {signal.shape}")
    if not (0 < low_hz < high_hz < fs / 2):
        raise ValueError(
            f"Invalid cutoffs: low={low_hz}, high={high_hz}, nyquist={fs/2}"
        )

    nyq = fs / 2.0
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")
    filtered = sosfiltfilt(sos, signal.astype(np.float64))
    return filtered.astype(np.float32)

def normalize(
    signal: np.ndarray,
    method: str = "z_score",
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    signal = signal.astype(np.float32)

    if method == "z_score":
        mean = np.mean(signal, axis=axis, keepdims=True)
        std  = np.std(signal,  axis=axis, keepdims=True)
        return (signal - mean) / (std + eps)

    elif method == "min_max":
        vmin = np.min(signal, axis=axis, keepdims=True)
        vmax = np.max(signal, axis=axis, keepdims=True)
        return (signal - vmin) / (vmax - vmin + eps)

    else:
        raise ValueError(f"Unknown normalization method: '{method}'. Use 'z_score' or 'min_max'.")

def sliding_window(
    signal: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:

    if signal.ndim != 1:
        raise ValueError(f"Expected 1-D signal, got shape {signal.shape}")
    if len(signal) < window_size:
        raise ValueError(
            f"Signal length {len(signal)} < window_size {window_size}"
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    n_windows = (len(signal) - window_size) // stride + 1
    shape = (n_windows, window_size)
    strides = (signal.strides[0] * stride, signal.strides[0])
    windows = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    return np.ascontiguousarray(windows, dtype=np.float32)

def sliding_window_labels(
    labels: np.ndarray,
    window_size: int,
    stride: int,
    aggregation: str = "max",
) -> np.ndarray:

    windows = sliding_window(labels.astype(np.float32), window_size, stride)

    if aggregation == "max":
        return (windows.max(axis=1) > 0).astype(np.int64)
    elif aggregation == "mean":
        return windows.mean(axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown aggregation: '{aggregation}'. Use 'max' or 'mean'.")

def preprocess_signal(
    signal: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    apply_bandpass: bool = True,
    low_hz: float = 0.5,
    high_hz: float = 4.0,
    fs: float = 64.0,
    filter_order: int = 4,
    normalize_method: str = "z_score",
    window_size: int = 192,         
    stride: int = 96,
    label_aggregation: str = "max",
) -> tuple[np.ndarray, np.ndarray | None]:
    if apply_bandpass:
        signal = bandpass_filter(signal, low_hz, high_hz, fs, filter_order)

    signal = normalize(signal, method=normalize_method)
    windows = sliding_window(signal, window_size, stride)

    window_labels = None
    if labels is not None:
        window_labels = sliding_window_labels(labels, window_size, stride, label_aggregation)

    log.debug(
        "preprocess_signal: windows=%s labels=%s",
        windows.shape,
        window_labels.shape if window_labels is not None else None,
    )
    return windows, window_labels