from __future__ import annotations

import io
from typing import List

import numpy as np

def weights_to_bytes(weights: List[np.ndarray]) -> bytes:

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{f"w_{i}": arr for i, arr in enumerate(weights)})
    return buffer.getvalue()

def bytes_to_weights(data: bytes) -> List[np.ndarray]:

    if not data:
        raise ValueError("Cannot deserialize empty bytes.")

    buffer = io.BytesIO(data)
    npz = np.load(buffer, allow_pickle=False)

    keys = sorted(npz.files, key=lambda k: int(k.split("_")[1]))
    return [npz[k] for k in keys]

def weights_size_mb(weights: List[np.ndarray]) -> float:
    total_bytes = sum(arr.nbytes for arr in weights)
    return round(total_bytes / (1024 ** 2), 4)