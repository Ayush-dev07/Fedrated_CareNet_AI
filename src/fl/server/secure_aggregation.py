from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import flwr as fl

from src.utils.logging import get_logger

log = get_logger(__name__)

def setup_secure_aggregation(
    fl_cfg: SimpleNamespace,
) -> Optional[fl.server.strategy.SecureAggregation]:
    
    sa_cfg = getattr(fl_cfg, "secure_aggregation", SimpleNamespace(enabled=False))
    enabled = getattr(sa_cfg, "enabled", False)

    if not enabled:
        log.info("Secure aggregation disabled")
        return None
    try:
        from flwr.server.strategy import SecureAggregation
    except ImportError:
        log.error(
            "SecureAggregation not available. "
            "Ensure flwr >= 1.5.0 is installed with secure aggregation support."
        )
        return None

    min_num_clients = int(getattr(sa_cfg, "min_num_clients", 2))
    timeout_in_seconds = float(getattr(sa_cfg, "timeout_in_seconds", 120.0))

    log.info(
        "Secure aggregation enabled: min_num_clients=%d, timeout=%f seconds",
        min_num_clients,
        timeout_in_seconds,
    )

    return SecureAggregation(
        min_num_clients=min_num_clients,
        timeout_in_seconds=timeout_in_seconds,
    )