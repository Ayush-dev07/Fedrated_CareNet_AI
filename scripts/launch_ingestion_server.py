from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import run_server
from backend.stream_processor import RealtimeWindowProcessor

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the real-time ingestion server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", default="data/realtime")
    parser.add_argument("--window-size", type=int, default=192)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--sampling-rate", type=float, default=64.0)
    parser.add_argument("--normalize-method", default="z_score")
    parser.add_argument("--no-bandpass", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    processor = RealtimeWindowProcessor(
        window_size=args.window_size,
        stride=args.stride,
        sampling_rate=args.sampling_rate,
        output_dir=args.output_dir,
        apply_bandpass=not args.no_bandpass,
        normalize_method=args.normalize_method,
    )
    run_server(host=args.host, port=args.port, processor=processor)
