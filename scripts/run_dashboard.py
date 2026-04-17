from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch Streamlit monitoring dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port",        type=int, default=8501,    help="Streamlit server port.")
    p.add_argument("--results_dir", type=str, default="results", help="Results directory to read CSVs from.")
    p.add_argument("--no_browser",  action="store_true",        help="Do not auto-open browser.")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    app_path  = repo_root / "dashboard" / "app.py"

    if not app_path.exists():
        print(f"Error: dashboard/app.py not found at {app_path}")
        print("Run this script from the repo root after Phase 11 is complete.")
        sys.exit(1)

    # Pass results_dir as env var — Streamlit app reads it on startup
    env = os.environ.copy()
    env["FHI_RESULTS_DIR"] = str(Path(args.results_dir).resolve())

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.headless", "true" if args.no_browser else "false",
        "--server.fileWatcherType", "none",   # disable hot-reload in simulation runs
        "--theme.base", "dark",
    ]

    print(f"\nStarting dashboard at http://localhost:{args.port}")
    print(f"Reading results from: {Path(args.results_dir).resolve()}")
    print("Press Ctrl+C to stop.\n")

    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except FileNotFoundError:
        print("Error: streamlit not found. Install with: pip install streamlit>=1.28.0")
        sys.exit(1)

if __name__ == "__main__":
    main()