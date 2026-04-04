from __future__ import annotations

from types import SimpleNamespace
import torch
import torch.nn as nn

from src.models.autoencoder import LSTMAutoencoder
from src.models.lstm import LSTMAnomalyDetector
from src.utils.logging import get_logger

log = get_logger(__name__)

_MODEL_REGISTRY: dict[str, type] = {
    "lstm":        LSTMAnomalyDetector,
    "autoencoder": LSTMAutoencoder,
}

def get_model(
    config: SimpleNamespace,
    seq_len: int | None = None,
) -> nn.Module:

    arch = getattr(config, "architecture", "lstm").lower()

    if arch not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture: '{arch}'. "
            f"Available: {sorted(_MODEL_REGISTRY.keys())}"
        )

    if arch == "lstm":
        model = LSTMAnomalyDetector.from_config(config)

    elif arch == "autoencoder":
        if seq_len is None:
            seq_len = getattr(getattr(config, "autoencoder", None), "seq_len", None)
        if seq_len is None:
            raise ValueError(
                "seq_len must be provided for autoencoder. "
                "Pass seq_len=window_size to get_model()."
            )
        model = LSTMAutoencoder.from_config(config, seq_len=seq_len)

    else:
        model = _MODEL_REGISTRY[arch](config)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %s | trainable params: %s", arch, f"{n_params:,}")

    return model

def get_loss_fn(config: SimpleNamespace, class_weights=None) -> nn.Module:

    import torch

    loss_name = getattr(config.training, "loss", "cross_entropy").lower()

    if loss_name == "cross_entropy":
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=weights)
        return nn.CrossEntropyLoss()

    elif loss_name in ("bce", "binary_cross_entropy"):
        return nn.BCEWithLogitsLoss()

    elif loss_name == "mse":
        return nn.MSELoss()

    else:
        raise ValueError(
            f"Unknown loss: '{loss_name}'. "
            f"Available: cross_entropy | bce | mse"
        )

def get_optimizer(
    model: nn.Module,
    config: SimpleNamespace,
) -> "torch.optim.Optimizer":

    import torch.optim as optim

    lr = getattr(config.training, "learning_rate", 1e-3)
    wd = getattr(config.training, "weight_decay", 1e-5)
    opt_name = getattr(config.training, "optimizer", "adam").lower()

    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(
            f"Unknown optimizer: '{opt_name}'. "
            f"Available: adam | adamw | sgd"
        )

def get_scheduler(
    optimizer: "torch.optim.Optimizer",
    config: SimpleNamespace,
) -> "torch.optim.lr_scheduler._LRScheduler | None":

    import torch.optim.lr_scheduler as sched

    sched_name = getattr(config.training, "scheduler", "none").lower()

    if sched_name == "none":
        return None
    elif sched_name == "cosine":
        t_max = getattr(config.training, "scheduler_t_max", 10)
        return sched.CosineAnnealingLR(optimizer, T_max=t_max)
    elif sched_name == "step":
        return sched.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        raise ValueError(
            f"Unknown scheduler: '{sched_name}'. "
            f"Available: cosine | step | none"
        )

def register_model(name: str, cls: type) -> None:
    _MODEL_REGISTRY[name.lower()] = cls
    log.info("Registered model: %s → %s", name, cls.__name__)