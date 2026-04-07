from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from src.utils.logging import get_logger
from src.utils.metrics import MetricsTracker

log = get_logger(__name__)

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]

class CustomFedAvg(fl.server.strategy.FedAvg):

    def __init__(
        self,
        metrics_tracker: Optional[MetricsTracker] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.metrics_tracker = metrics_tracker
        self._round = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self._round = server_round

        if failures:
            log.warning(
                "Round %d: %d client failures (ignored by FedAvg).",
                server_round, len(failures),
            )

        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if results:
            client_metrics = self._weighted_average_metrics(results)
            aggregated_metrics.update(client_metrics)

            log.info(
                "Round %d | clients=%d | loss=%.4f | acc=%.4f",
                server_round,
                len(results),
                client_metrics.get("train_loss", 0.0),
                client_metrics.get("train_accuracy", 0.0),
            )

            if self.metrics_tracker is not None:
                self.metrics_tracker.log({
                    "round":          server_round,
                    "strategy":       "fedavg",
                    "num_clients":    len(results),
                    "num_failures":   len(failures),
                    **{k: round(float(v), 6) if isinstance(v, float) else v
                       for k, v in client_metrics.items()},
                })

        return aggregated_params, aggregated_metrics

    @staticmethod
    def _weighted_average_metrics(
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Dict[str, float]:

        total_samples = sum(fit_res.num_examples for _, fit_res in results)
        if total_samples == 0:
            return {}

        weighted: Dict[str, float] = {}
        scalar_keys = ["train_loss", "train_accuracy"]

        for key in scalar_keys:
            values = [
                (fit_res.metrics.get(key, 0.0), fit_res.num_examples)
                for _, fit_res in results
                if key in (fit_res.metrics or {})
            ]
            if values:
                weighted[key] = sum(v * n for v, n in values) / total_samples

        weighted["total_train_samples"] = float(total_samples)
        return weighted