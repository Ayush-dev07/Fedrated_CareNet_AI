from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a synthetic device packet to the ingestion server.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/ingest")
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--sensor", default="heart_rate")
    parser.add_argument("--n-samples", type=int, default=256)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng()
    samples = (60 + rng.normal(0, 2, args.n_samples)).astype(float).tolist()
    payload = {
        "device_id": args.device_id,
        "sensor": args.sensor,
        "samples": samples,
        "sample_rate": 64.0,
        "timestamp": None,
    }
    req = Request(
        args.server_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req) as resp:
        print(resp.read().decode("utf-8"))
