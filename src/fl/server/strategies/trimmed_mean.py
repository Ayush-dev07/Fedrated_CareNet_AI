from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from src.fl.server.strategies.fedavg import CustomFedAvg
from src.utils.logging import get_logger
from src.utils.metrics import MetricsTracker

log = get_logger(__name__)

class TrimmedMean(CustomFedAvg):

    def __init__(
        self,
        trim_fraction: float = 0.1,
        metrics_tracker: Optional[MetricsTracker] = None,
        **kwargs,
    ) -> None:
        super().__init__(metrics_tracker=metrics_tracker, **kwargs)

        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError(
                f"trim_fraction must be in [0, 0.5). Got {trim_fraction}."
            )
        self.trim_fraction = trim_fraction
        log.info("TrimmedMean: trim_fraction=%.2f", trim_fraction)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        weights_list = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        n_clients = len(weights_list)
        n_trim    = int(n_clients * self.trim_fraction)
        n_keep    = n_clients - 2 * n_trim

        if n_keep < 1:
            log.warning(
                "TrimmedMean: trim_fraction=%.2f with %d clients would remove all. "
                "Falling back to standard FedAvg aggregation.",
                self.trim_fraction, n_clients,
            )
            return super().aggregate_fit(server_round, results, failures)

        norms = np.array([
            np.sqrt(sum(np.sum(w ** 2) for w in weights))
            for weights, _ in weights_list
        ])
        sorted_indices = np.argsort(norms)             
        kept_indices   = sorted_indices[n_trim: n_clients - n_trim]

        trimmed_results = [results[i] for i in kept_indices]
        trimmed_weights = [weights_list[i] for i in kept_indices]

        log.info(
            "Round %d | TrimmedMean: %d clients → trimmed %d low + %d high → kept %d | "
            "norms [%.4f, %.4f]",
            server_round, n_clients, n_trim, n_trim, len(kept_indices),
            norms[sorted_indices[0]], norms[sorted_indices[-1]],
        )

        if self.metrics_tracker is not None:
            self.metrics_tracker.log({
                "round":          server_round,
                "strategy":       "trimmed_mean",
                "total_clients":  n_clients,
                "trimmed":        2 * n_trim,
                "kept":           len(kept_indices),
                "trim_fraction":  self.trim_fraction,
                "norm_min":       round(float(norms.min()), 6),
                "norm_max":       round(float(norms.max()), 6),
                "norm_mean":      round(float(norms.mean()), 6),
            })

        aggregated_ndarrays = self._weighted_aggregate(trimmed_weights)
        aggregated_params   = ndarrays_to_parameters(aggregated_ndarrays)

        client_metrics = self._weighted_average_metrics(trimmed_results)

        return aggregated_params, client_metrics

    @staticmethod
    def _weighted_aggregate(
        weights_and_counts: List[Tuple[List[np.ndarray], int]],
    ) -> List[np.ndarray]:

        total_samples = sum(count for _, count in weights_and_counts)
        if total_samples == 0:
            return weights_and_counts[0][0]
        aggregated = [
            np.zeros_like(layer) for layer in weights_and_counts[0][0]
        ]
        for weights, count in weights_and_counts:
            fraction = count / total_samples
            for i, layer in enumerate(weights):
                aggregated[i] += layer * fraction

        return aggregated