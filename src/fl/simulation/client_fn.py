from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Dict

import flwr as fl
from torch.utils.data import DataLoader

from src.fl.client.client import HealthClient
from src.utils.logging import get_logger

log = get_logger(__name__)

def make_client_fn(
    train_loaders: Dict[int, DataLoader],
    val_loaders: Dict[int, DataLoader],
    model_cfg: SimpleNamespace,
    privacy_cfg: SimpleNamespace,
    fl_cfg: SimpleNamespace,
) -> Callable[[str], fl.client.Client]:

    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)

        if client_id not in train_loaders:
            raise KeyError(
                f"client_fn: no DataLoader for client_id={client_id}. "
                f"Available: {sorted(train_loaders.keys())}"
            )

        client = HealthClient(
            client_id=client_id,
            train_loader=train_loaders[client_id],
            val_loader=val_loaders[client_id],
            model_cfg=model_cfg,
            privacy_cfg=privacy_cfg,
            fl_cfg=fl_cfg,
        )

        log.debug("Spawned HealthClient %d", client_id)
        return client.to_client()

    return client_fn

def make_client_fn_from_config(
    config: SimpleNamespace,
    model_cfg: SimpleNamespace,
    privacy_cfg: SimpleNamespace,
    fl_cfg: SimpleNamespace,
) -> Callable[[str], fl.client.Client]:
    from src.data.loaders import get_dataloader

    n_clients = int(getattr(config.partitioning, "num_clients", 20))

    log.info("Building DataLoaders for %d clients...", n_clients)

    train_loaders = {
        cid: get_dataloader(cid, config, split="train")
        for cid in range(n_clients)
    }
    val_loaders = {
        cid: get_dataloader(cid, config, split="val")
        for cid in range(n_clients)
    }

    log.info("DataLoaders ready. Building client_fn closure.")
    return make_client_fn(train_loaders, val_loaders, model_cfg, privacy_cfg, fl_cfg)