from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.models.factory import get_loss_fn, get_model, get_optimizer
from src.models.utils import clone_model
from src.utils.logging import get_logger

log = get_logger(__name__)

def shadow_model_attack(
    target_model: nn.Module,
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    n_shadow_models: int = 5,
    shadow_train_epochs: int = 5,
    attack_train_epochs: int = 10,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, float]:

    t_start = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    log.info(
        "Starting shadow model attack: %d shadow models, %d member batches, %d non-member batches",
        n_shadow_models, len(member_loader), len(non_member_loader),
    )

    member_confs    = _get_confidence_vectors(target_model, member_loader,    device)
    nonmember_confs = _get_confidence_vectors(target_model, non_member_loader, device)

    member_conf_mean    = float(member_confs.max(axis=1).mean())
    nonmember_conf_mean = float(nonmember_confs.max(axis=1).mean())

    log.debug(
        "Target model: member_conf=%.4f  nonmember_conf=%.4f",
        member_conf_mean, nonmember_conf_mean,
    )

    attack_features, attack_labels = _build_attack_dataset(
        target_model=target_model,
        member_loader=member_loader,
        non_member_loader=non_member_loader,
        n_shadow_models=n_shadow_models,
        train_epochs=shadow_train_epochs,
        device=device,
        seed=seed,
    )

    n_features  = attack_features.shape[1]
    attack_clf  = _AttackClassifier(input_dim=n_features).to(device)
    attack_auc, attack_acc = _train_attack_classifier(
        attack_clf, attack_features, attack_labels,
        epochs=attack_train_epochs, device=device,
    )

    advantage           = round(attack_auc - 0.5, 6)
    epsilon_empirical   = round(_auc_to_epsilon(attack_auc), 6)
    confidence_gap      = round(member_conf_mean - nonmember_conf_mean, 6)
    dp_auc_upper_bound  = 0.5   

    duration = round(time.time() - t_start, 2)

    results = {
        "attack_auc":            round(attack_auc, 6),
        "attack_accuracy":       round(attack_acc, 6),
        "attack_advantage":      advantage,
        "member_confidence":     round(member_conf_mean, 6),
        "nonmember_confidence":  round(nonmember_conf_mean, 6),
        "confidence_gap":        confidence_gap,
        "dp_auc_upper_bound":    dp_auc_upper_bound,
        "epsilon_empirical":     epsilon_empirical,
        "n_shadow_models":       n_shadow_models,
        "duration_seconds":      duration,
    }

    log.info(
        "Privacy audit complete: attack_auc=%.4f  advantage=%.4f  "
        "ε_empirical=%.4f  duration=%.1fs",
        attack_auc, advantage, epsilon_empirical, duration,
    )

    return results

def compare_dp_vs_nodp_leakage(
    dp_model: nn.Module,
    nodp_model: nn.Module,
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:

    log.info("Comparing DP vs no-DP membership inference leakage...")

    dp_results   = shadow_model_attack(dp_model,   member_loader, non_member_loader, device=device)
    nodp_results = shadow_model_attack(nodp_model, member_loader, non_member_loader, device=device)

    improvement = round(nodp_results["attack_auc"] - dp_results["attack_auc"], 6)

    result = {
        "dp_attack_auc":       dp_results["attack_auc"],
        "nodp_attack_auc":     nodp_results["attack_auc"],
        "dp_advantage":        dp_results["attack_advantage"],
        "nodp_advantage":      nodp_results["attack_advantage"],
        "privacy_improvement": improvement,
    }

    log.info(
        "DP vs no-DP: dp_auc=%.4f  nodp_auc=%.4f  improvement=%.4f",
        result["dp_attack_auc"], result["nodp_attack_auc"], improvement,
    )
    return result

def _get_confidence_vectors(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    model.eval()
    confs: List[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader:
            x    = x.to(device)
            prob = torch.softmax(model(x), dim=-1).cpu().numpy()
            confs.append(prob)
    return np.vstack(confs) if confs else np.empty((0, 2))

def _build_attack_dataset(
    target_model: nn.Module,
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    n_shadow_models: int,
    train_epochs: int,
    device: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:

    all_features: List[np.ndarray] = []
    all_labels:   List[np.ndarray] = []

    mem_x, mem_y     = _loader_to_tensors(member_loader)
    nonmem_x, nonmem_y = _loader_to_tensors(non_member_loader)
    n_total          = min(len(mem_x), len(nonmem_x))

    rng = np.random.default_rng(seed)

    for shadow_idx in range(n_shadow_models):
        
        indices    = rng.choice(n_total, size=max(n_total // 2, 1), replace=False)
        shadow_x   = mem_x[indices]
        shadow_y   = mem_y[indices]

        held_idx   = np.setdiff1d(np.arange(n_total), indices)
        held_x     = mem_x[held_idx]
        held_y     = mem_y[held_idx]

        shadow_model = clone_model(target_model).to(device)
        shadow_model = _train_shadow_model(
            shadow_model, shadow_x, shadow_y, device, train_epochs, seed + shadow_idx
        )

        shadow_mem_conf    = _confs_from_tensor(shadow_model, shadow_x, device)
        shadow_nonmem_conf = _confs_from_tensor(shadow_model, held_x,   device)

        all_features.append(shadow_mem_conf)
        all_labels.append(np.ones(len(shadow_mem_conf), dtype=np.int64))
        all_features.append(shadow_nonmem_conf)
        all_labels.append(np.zeros(len(shadow_nonmem_conf), dtype=np.int64))

        log.debug("Shadow model %d/%d trained.", shadow_idx + 1, n_shadow_models)

    features = np.vstack(all_features)
    labels   = np.concatenate(all_labels)
    return features, labels

def _train_shadow_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: str,
    epochs: int,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    dataset   = TensorDataset(x, y)
    loader    = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    for _ in range(epochs):
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    return model

def _train_attack_classifier(
    clf: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    device: str,
) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    feat_t  = torch.tensor(features, dtype=torch.float32)
    label_t = torch.tensor(labels,   dtype=torch.long)
    dataset = TensorDataset(feat_t, label_t)

    n_val   = max(int(len(dataset) * 0.2), 1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    clf.train()

    for _ in range(epochs):
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            loss = criterion(clf(bx), by)
            loss.backward()
            optimizer.step()

    clf.eval()
    val_feats  = feat_t[[i for i in val_ds.indices]]
    val_labels = label_t[[i for i in val_ds.indices]].numpy()

    with torch.no_grad():
        probs = torch.softmax(clf(val_feats.to(device)), dim=-1)[:, 1].cpu().numpy()
        preds = (probs > 0.5).astype(int)

    auc = float(roc_auc_score(val_labels, probs)) if len(np.unique(val_labels)) > 1 else 0.5
    acc = float((preds == val_labels).mean())
    return auc, acc

def _confs_from_tensor(model: nn.Module, x: torch.Tensor, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        loader = DataLoader(TensorDataset(x), batch_size=64, num_workers=0)
        confs  = [torch.softmax(model(bx[0].to(device)), dim=-1).cpu().numpy()
                  for bx in loader]
    return np.vstack(confs) if confs else np.empty((0, 2))

def _loader_to_tensors(
    loader: DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def _auc_to_epsilon(auc: float) -> float:

    import math
    advantage = max(auc - 0.5, 0.0)
    return round(advantage * 2 * math.sqrt(2 * math.pi), 6)


class _AttackClassifier(nn.Module):

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)