from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple

import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.fl.privacy.accountant import RDPAccountant
from src.utils.logging import get_logger

log = get_logger(__name__)

def setup_dp(
    model: nn.Module,
    optimizer: Optimizer,
    loader: DataLoader,
    config: SimpleNamespace,
    num_rounds: int | None = None,
) -> Tuple[nn.Module, Optimizer, DataLoader, RDPAccountant]:

    try:
        from opacus import PrivacyEngine
    except ImportError as e:
        raise ImportError(
            "Opacus is required for DP training. "
            "Install with: pip install opacus>=1.4.0"
        ) from e

    dp_cfg = config.dp_sgd

    noise_multiplier = float(dp_cfg.noise_multiplier)
    max_grad_norm    = float(dp_cfg.max_grad_norm)
    target_epsilon   = float(dp_cfg.epsilon)
    delta            = float(dp_cfg.delta)
    secure_mode      = bool(getattr(dp_cfg, "secure_mode", False))

    privacy_engine = PrivacyEngine(secure_mode=secure_mode)
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=True,       
    )

    log.info(
        "DP-SGD enabled: σ=%.3f  C=%.3f  ε_target=%.3f  δ=%.1e  secure=%s",
        noise_multiplier, max_grad_norm, target_epsilon, delta, secure_mode,
    )

    n_samples   = len(loader.dataset)  
    batch_size  = loader.batch_size or 32
    sample_rate = batch_size / max(n_samples, 1)

    _num_rounds = num_rounds or getattr(config, "total_rounds", 20)

    accountant = RDPAccountant(
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        delta=delta,
        sample_rate=sample_rate,
        target_epsilon=target_epsilon,
        total_rounds=_num_rounds,
        warning_threshold=getattr(
            getattr(config, "accounting", None), "budget_warning_threshold", 0.8
        ),
    )

    return dp_model, dp_optimizer, dp_loader, accountant

def is_dp_enabled(config: SimpleNamespace) -> bool:
    return bool(getattr(config, "enabled", False))

def calibrate_noise_multiplier(
    target_epsilon: float,
    delta: float,
    sample_rate: float,
    num_steps: int,
    max_grad_norm: float = 1.0,
    tolerance: float = 0.01,
    max_iterations: int = 100,
) -> float:
    try:
        from opacus.accountants import RDPAccountant as OpacusRDP
    except ImportError as e:
        raise RuntimeError("Opacus required for noise calibration.") from e

    lo, hi = 0.1, 100.0

    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        acc = OpacusRDP()
        acc.history = [(mid, sample_rate, num_steps)]
        eps = acc.get_epsilon(delta=delta)

        if abs(eps - target_epsilon) < tolerance:
            log.info(
                "Calibrated noise_multiplier=%.4f → ε=%.4f (target=%.4f)",
                mid, eps, target_epsilon,
            )
            return mid

        if eps > target_epsilon:
            lo = mid   
        else:
            hi = mid   
    log.warning(
        "Noise calibration did not converge in %d iterations. "
        "Returning σ=%.4f (ε≈%.4f, target=%.4f).",
        max_iterations, mid, eps, target_epsilon,
    )
    return mid