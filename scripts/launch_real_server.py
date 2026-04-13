from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fl.server.server import start_server
from src.utils.config import load_config
from src.utils.metrics import MetricsTracker

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a real Flower server.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--fl-config", default="configs/fl.yaml")
    parser.add_argument("--privacy-config", default="configs/privacy.yaml")
    parser.add_argument("--server-address", default="0.0.0.0:8080")
    parser.add_argument("--results-dir", default="results/realtime_run")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    fl_cfg = load_config(args.fl_config)
    privacy_cfg = load_config(args.privacy_config)

    _ = data_cfg, privacy_cfg

    metrics_tracker = MetricsTracker(Path(args.results_dir) / "metrics.csv")
    start_server(
        fl_cfg=fl_cfg,
        model_cfg=model_cfg,
        val_loader=None,
        metrics_tracker=metrics_tracker,
        server_address=args.server_address,
    )
