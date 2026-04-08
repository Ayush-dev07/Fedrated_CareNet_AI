from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from src.utils.logging import get_logger
from src.utils.metrics import MetricsTracker

log = get_logger(__name__)

def compare_fl_vs_centralized(
    sim_cfg,
    results_dir: str | Path = "results",
    centralized_epochs: int = 20,
    device: str = "cpu",
) -> Dict[str, object]:

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    tracker = MetricsTracker(results_path / "benchmark.csv")

    data_cfg    = sim_cfg.data_cfg
    model_cfg   = sim_cfg.model_cfg
    privacy_cfg = sim_cfg.privacy_cfg
    fl_cfg      = sim_cfg.fl_cfg

    from src.data.loaders import get_dataloader
    n_clients = sim_cfg.num_clients

    log.info("Building data loaders for %d clients...", n_clients)
    train_loaders = [get_dataloader(cid, data_cfg, split="train") for cid in range(n_clients)]
    val_loaders   = [get_dataloader(cid, data_cfg, split="val")   for cid in range(n_clients)]

    log.info("=== Training centralised baseline ===")
    central_results = _train_centralized(
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        model_cfg=model_cfg,
        epochs=centralized_epochs,
        device=device,
    )
    tracker.log({"run": "centralized", **central_results})
    log.info(
        "Centralized: acc=%.4f  auc=%s  loss=%.4f  time=%.1fs",
        central_results["accuracy"],
        f"{central_results.get('auc', 'N/A'):.4f}" if "auc" in central_results else "N/A",
        central_results["loss"],
        central_results["train_time_s"],
    )

    log.info("=== Running FL simulation (%s) ===", sim_cfg.scenario_name)
    fl_results = _run_fl_and_eval(
        sim_cfg=sim_cfg,
        val_loaders=val_loaders,
        model_cfg=model_cfg,
        device=device,
    )
    tracker.log({"run": "federated", **fl_results})
    log.info(
        "Federated: acc=%.4f  auc=%s  loss=%.4f  time=%.1fs  comm=%.2fMB",
        fl_results["accuracy"],
        f"{fl_results.get('auc', 'N/A'):.4f}" if "auc" in fl_results else "N/A",
        fl_results["loss"],
        fl_results["train_time_s"],
        fl_results.get("comm_cost_mb", 0),
    )

    accuracy_gap    = round(central_results["accuracy"] - fl_results["accuracy"], 6)
    fl_comm         = fl_results.get("comm_cost_mb", 0)
    central_comm    = central_results.get("comm_cost_mb", 0)
    comm_ratio      = round(fl_comm / max(central_comm, 1e-9), 4)

    comparison = {
        "centralized":       central_results,
        "federated":         fl_results,
        "accuracy_gap":      accuracy_gap,
        "comm_cost_ratio":   comm_ratio,
        "scenario":          sim_cfg.scenario_name,
    }

    tracker.log({
        "run":            "comparison",
        "accuracy_gap":   accuracy_gap,
        "comm_ratio":     comm_ratio,
        "scenario":       sim_cfg.scenario_name,
    })
    tracker.close()

    log.info(
        "Benchmark complete: accuracy_gap=%.4f (centralised - FL)  "
        "comm_ratio=%.2fx  results → %s",
        accuracy_gap, comm_ratio, results_path / "benchmark.csv",
    )

    return comparison

def _train_centralized(
    train_loaders: List[DataLoader],
    val_loaders: List[DataLoader],
    model_cfg,
    epochs: int,
    device: str,
) -> Dict:
    from src.models.factory import get_loss_fn, get_model, get_optimizer

    seq_len = None
    try:
        x, _ = next(iter(train_loaders[0]))
        seq_len = x.shape[-1]
    except Exception:
        pass

    model     = get_model(model_cfg, seq_len=seq_len).to(device)
    optimizer = get_optimizer(model, model_cfg)
    loss_fn   = get_loss_fn(model_cfg)

    pooled_train = ConcatDataset([loader.dataset for loader in train_loaders])
    pooled_val   = ConcatDataset([loader.dataset for loader in val_loaders])
    batch_size   = train_loaders[0].batch_size or 32

    train_loader = DataLoader(pooled_train, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(pooled_val,   batch_size=batch_size, shuffle=False, num_workers=0)

    n_params = sum(p.numel() for p in model.parameters())
    comm_mb  = round(n_params * 4 * 2 / (1024 ** 2), 4)   

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    train_time = round(time.time() - t0, 2)
    metrics    = _evaluate_model(model, val_loader, device)
    metrics["train_time_s"] = train_time
    metrics["comm_cost_mb"] = comm_mb
    return metrics

def _run_fl_and_eval(
    sim_cfg,
    val_loaders: List[DataLoader],
    model_cfg,
    device: str,
) -> Dict:
    from src.fl.simulation.simulator import run_simulation
    from src.models.factory import get_model
    from src.models.utils import set_parameters

    t0 = time.time()
    history = run_simulation(sim_cfg, results_dir="results/benchmark_fl_run")
    train_time = round(time.time() - t0, 2)

    final_loss = 0.0
    if history.losses_distributed:
        _, final_loss = max(history.losses_distributed, key=lambda x: x[0])

    seq_len = None
    try:
        x, _ = next(iter(val_loaders[0]))
        seq_len = x.shape[-1]
    except Exception:
        pass

    model    = get_model(model_cfg, seq_len=seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    comm_mb  = round(
        n_params * 4 * sim_cfg.num_rounds * sim_cfg.clients_per_round * 2 / (1024 ** 2),
        4,
    )

    pooled_val   = ConcatDataset([loader.dataset for loader in val_loaders])
    batch_size   = val_loaders[0].batch_size or 32
    val_loader   = DataLoader(pooled_val, batch_size=batch_size, shuffle=False, num_workers=0)

    metrics = _evaluate_model(model, val_loader, device)
    metrics["loss"]         = round(float(final_loss), 6)
    metrics["train_time_s"] = train_time
    metrics["comm_cost_mb"] = comm_mb
    return metrics

def _evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> Dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    all_probs:  List[float] = []
    all_labels: List[int]   = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            bs             = y.size(0)
            total_loss    += loss.item() * bs
            preds          = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += bs

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().tolist())

    n       = max(total_samples, 1)
    metrics = {
        "loss":     round(total_loss / n, 6),
        "accuracy": round(total_correct / n, 6),
    }

    try:
        from sklearn.metrics import roc_auc_score
        if len(set(all_labels)) > 1:
            metrics["auc"] = round(float(roc_auc_score(all_labels, all_probs)), 6)
    except ImportError:
        pass

    return metrics