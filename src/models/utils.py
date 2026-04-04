from __future__ import annotations

from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.utils.logging import get_logger

log = get_logger(__name__)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [
        val.cpu().detach().numpy().astype(np.float32)
        for val in model.state_dict().values()
    ]

def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:

    state_dict_keys = list(model.state_dict().keys())

    if len(parameters) != len(state_dict_keys):
        raise ValueError(
            f"Parameter count mismatch: got {len(parameters)}, "
            f"model expects {len(state_dict_keys)}"
        )

    current_shapes = [v.shape for v in model.state_dict().values()]
    incoming_shapes = [np.asarray(p).shape for p in parameters]
    mismatches = [
        (k, cs, ins)
        for k, cs, ins in zip(state_dict_keys, current_shapes, incoming_shapes)
        if cs != ins
    ]
    if mismatches:
        details = "; ".join(f"{k}: model={cs} incoming={ins}" for k, cs, ins in mismatches)
        raise ValueError(f"Shape mismatch in parameters: {details}")

    new_state_dict: OrderedDict = OrderedDict(
        {k: torch.tensor(np.asarray(v), dtype=torch.float32)
         for k, v in zip(state_dict_keys, parameters)}
    )
    model.load_state_dict(new_state_dict, strict=True)

def get_state_dict(model: nn.Module) -> List[np.ndarray]:

    return [
        val.cpu().detach().numpy().astype(np.float32)
        for val in model.state_dict().values()
    ]


def set_state_dict(model: nn.Module, state: List[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    new_sd: OrderedDict = OrderedDict(
        {k: torch.tensor(np.asarray(v)) for k, v in zip(keys, state)}
    )
    model.load_state_dict(new_sd, strict=True)


def clone_model(model: nn.Module) -> nn.Module:

    import copy
    return copy.deepcopy(model)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def parameters_norm(model: nn.Module) -> float:

    total = sum(
        p.data.norm(2).item() ** 2
        for p in model.parameters()
        if p.requires_grad
    )
    return total ** 0.5


def parameters_l2_distance(
    params_a: List[np.ndarray],
    params_b: List[np.ndarray],
) -> float:
    total = sum(
        np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2)
        for a, b in zip(params_a, params_b)
    )
    return float(total ** 0.5)

def zero_parameters(parameters: List[np.ndarray]) -> List[np.ndarray]:
    return [np.zeros_like(p) for p in parameters]
