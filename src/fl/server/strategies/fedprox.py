from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitIns, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from src.fl.server.strategies.fedavg import CustomFedAvg
from src.utils.logging import get_logger
from src.utils.metrics import MetricsTracker

log = get_logger(__name__)

class CustomFedProx(CustomFedAvg):
    def __init__(
        self,
        mu: float = 0.01,
        metrics_tracker: Optional[MetricsTracker] = None,
        **kwargs,
    ) -> None:
        super().__init__(metrics_tracker=metrics_tracker, **kwargs)
        self.mu = mu
        log.info("FedProx: mu=%.4f", mu)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        client_instructions = super().configure_fit(server_round, parameters, client_manager)

        proximal_config = {"mu": self.mu, "proximal": True}
        updated_instructions = []
        for client, fit_ins in client_instructions:
            updated_config = dict(fit_ins.config)
            updated_config.update(proximal_config)
            updated_instructions.append(
                (client, fl.common.FitIns(fit_ins.parameters, updated_config))
            )

        return updated_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if self.metrics_tracker is not None and results:
            
            self.metrics_tracker.log({
                "round":    server_round,
                "strategy": "fedprox",
                "mu":       self.mu,
            })
        return aggregated_params, aggregated_metrics