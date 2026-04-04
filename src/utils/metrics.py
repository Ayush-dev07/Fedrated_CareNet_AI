from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

log = get_logger(__name__)


class MetricsTracker:

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = None
        self._writer: csv.DictWriter | None = None
        self._fieldnames: list[str] | None = None

    def log(self, metrics: dict[str, Any]) -> None:

        if self._writer is None:
            self._open(list(metrics.keys()))

        row = {k: metrics.get(k, "") for k in self._fieldnames}  
        self._writer.writerow(row)  
        self._file.flush()  

    def log_many(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.log(row)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            log.debug("MetricsTracker closed: %s", self.path)

    def __enter__(self) -> "MetricsTracker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _open(self, fieldnames: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames = fieldnames
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if write_header:
            self._writer.writeheader()
        log.info("MetricsTracker writing to: %s", self.path)

class PrivacyMetricsTracker(MetricsTracker):

    DEFAULT_FIELDS = ["round", "epsilon", "delta", "noise_multiplier", "rounds_remaining"]

    def __init__(self, path: str | Path = "results/privacy.csv") -> None:
        super().__init__(path)

    def log_privacy(
        self,
        round_num: int,
        epsilon: float,
        delta: float,
        noise_multiplier: float,
        rounds_remaining: int,
    ) -> None:
        self.log({
            "round": round_num,
            "epsilon": round(epsilon, 6),
            "delta": delta,
            "noise_multiplier": noise_multiplier,
            "rounds_remaining": rounds_remaining,
        })