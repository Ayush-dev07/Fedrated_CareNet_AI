from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict

from src.utils.config import load_config, merge_configs
from src.utils.logging import get_logger

log = get_logger(__name__)

_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "iid_20": {
        "description":   "20 clients, IID data, FedAvg, no DP",
        "num_clients":   20,
        "num_rounds":    20,
        "strategy":      "fedavg",
        "iid":           True,
        "dirichlet_alpha": None,
        "dp_enabled":    False,
        "local_epochs":  2,
        "clients_per_round": 5,
        "fraction_fit":  0.5,
    },
    "noniid_20": {
        "description":   "20 clients, non-IID (α=0.5), FedAvg, no DP",
        "num_clients":   20,
        "num_rounds":    20,
        "strategy":      "fedavg",
        "iid":           False,
        "dirichlet_alpha": 0.5,
        "dp_enabled":    False,
        "local_epochs":  2,
        "clients_per_round": 5,
        "fraction_fit":  0.5,
    },
    "noniid_extreme": {
        "description":   "20 clients, extreme non-IID (α=0.1), FedAvg, no DP",
        "num_clients":   20,
        "num_rounds":    20,
        "strategy":      "fedavg",
        "iid":           False,
        "dirichlet_alpha": 0.1,
        "dp_enabled":    False,
        "local_epochs":  2,
        "clients_per_round": 5,
        "fraction_fit":  0.5,
    },
    "dp_enabled": {
        "description":   "20 clients, non-IID (α=0.5), FedAvg + DP-SGD (ε=3.0)",
        "num_clients":   20,
        "num_rounds":    20,
        "strategy":      "fedavg",
        "iid":           False,
        "dirichlet_alpha": 0.5,
        "dp_enabled":    True,
        "local_epochs":  2,
        "clients_per_round": 5,
        "fraction_fit":  0.5,
    },
    "fedprox": {
        "description":   "20 clients, non-IID (α=0.5), FedProx (μ=0.01), no DP",
        "num_clients":   20,
        "num_rounds":    20,
        "strategy":      "fedprox",
        "iid":           False,
        "dirichlet_alpha": 0.5,
        "dp_enabled":    False,
        "local_epochs":  2,
        "clients_per_round": 5,
        "fraction_fit":  0.5,
        "mu":            0.01,
    },
    "trimmed_mean": {
        "description":   "20 clients, non-IID (α=0.5), TrimmedMean defense, no DP",
        "num_clients":   20,
        "num_rounds":    20,
        "strategy":      "trimmed_mean",
        "iid":           False,
        "dirichlet_alpha": 0.5,
        "dp_enabled":    False,
        "local_epochs":  2,
        "clients_per_round": 5,
        "fraction_fit":  0.5,
        "trim_fraction": 0.1,
    },
    "quick_test": {
        "description":   "5 clients, IID, 3 rounds — fast smoke test",
        "num_clients":   5,
        "num_rounds":    3,
        "strategy":      "fedavg",
        "iid":           True,
        "dirichlet_alpha": None,
        "dp_enabled":    False,
        "local_epochs":  1,
        "clients_per_round": 3,
        "fraction_fit":  0.6,
    },
}

@dataclass
class SimulationConfig:

    scenario_name:    str
    description:      str
    num_clients:      int
    num_rounds:       int
    strategy:         str
    iid:              bool
    dirichlet_alpha:  float | None
    dp_enabled:       bool
    local_epochs:     int
    clients_per_round: int
    fraction_fit:     float
    mu:               float = 0.01
    trim_fraction:    float = 0.1
    seed:             int   = 42

    fl_cfg:           SimpleNamespace | None = field(default=None, repr=False)
    model_cfg:        SimpleNamespace | None = field(default=None, repr=False)
    data_cfg:         SimpleNamespace | None = field(default=None, repr=False)
    privacy_cfg:      SimpleNamespace | None = field(default=None, repr=False)

    def summary(self) -> str:
        dp = f"DP(ε={self.privacy_cfg.dp_sgd.epsilon})" if self.dp_enabled and self.privacy_cfg else "no-DP"
        iid_str = "IID" if self.iid else f"non-IID(α={self.dirichlet_alpha})"
        return (
            f"[{self.scenario_name}] {self.num_clients} clients | "
            f"{self.num_rounds} rounds | {self.strategy} | {iid_str} | {dp}"
        )

def get_scenario(
    name: str,
    configs_dir: str = "configs",
) -> SimulationConfig:

    if name not in _SCENARIOS:
        raise ValueError(
            f"Unknown scenario: '{name}'. "
            f"Available: {sorted(_SCENARIOS.keys())}"
        )

    overrides = _SCENARIOS[name]

    import os
    fl_cfg      = load_config(os.path.join(configs_dir, "fl.yaml"))
    model_cfg   = load_config(os.path.join(configs_dir, "model.yaml"))
    data_cfg    = load_config(os.path.join(configs_dir, "data.yaml"))
    privacy_cfg = load_config(os.path.join(configs_dir, "privacy.yaml"))

    fl_cfg.strategy                          = overrides["strategy"]
    fl_cfg.rounds.num_rounds                 = overrides["num_rounds"]
    fl_cfg.rounds.clients_per_round          = overrides["clients_per_round"]
    fl_cfg.rounds.fraction_fit               = overrides["fraction_fit"]
    fl_cfg.local_training.local_epochs       = overrides["local_epochs"]

    data_cfg.partitioning.num_clients        = overrides["num_clients"]
    data_cfg.partitioning.iid                = overrides.get("iid", False)
    if overrides.get("dirichlet_alpha") is not None:
        data_cfg.partitioning.dirichlet_alpha = overrides["dirichlet_alpha"]

    if overrides["strategy"] == "fedprox":
        fl_cfg.aggregation.fedprox.mu        = overrides.get("mu", 0.01)
    if overrides["strategy"] == "trimmed_mean":
        fl_cfg.aggregation.trimmed_mean.trim_fraction = overrides.get("trim_fraction", 0.1)

    privacy_cfg.enabled = overrides["dp_enabled"]

    sim_cfg = SimulationConfig(
        scenario_name=name,
        description=overrides["description"],
        num_clients=overrides["num_clients"],
        num_rounds=overrides["num_rounds"],
        strategy=overrides["strategy"],
        iid=overrides.get("iid", False),
        dirichlet_alpha=overrides.get("dirichlet_alpha"),
        dp_enabled=overrides["dp_enabled"],
        local_epochs=overrides["local_epochs"],
        clients_per_round=overrides["clients_per_round"],
        fraction_fit=overrides["fraction_fit"],
        mu=overrides.get("mu", 0.01),
        trim_fraction=overrides.get("trim_fraction", 0.1),
        fl_cfg=fl_cfg,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        privacy_cfg=privacy_cfg,
    )

    log.info("Scenario loaded: %s", sim_cfg.summary())
    return sim_cfg

def list_scenarios() -> Dict[str, str]:
    return {k: v["description"] for k, v in _SCENARIOS.items()}