from __future__ import annotations

from pathlib import Path
import numpy as np

from src.utils.logging import get_logger

log = get_logger(__name__)

def iid_partition(
    n_samples: int,
    n_clients: int,
    seed: int = 42,
) -> list[np.ndarray]:

    if n_clients > n_samples:
        raise ValueError(
            f"n_clients ({n_clients}) > n_samples ({n_samples})"
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    splits = np.array_split(indices, n_clients)

    sizes = [len(s) for s in splits]
    log.info(
        "IID partition: %d clients, sizes min=%d max=%d mean=%.1f",
        n_clients, min(sizes), max(sizes), np.mean(sizes),
    )
    return splits

def dirichlet_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float = 0.5,
    min_samples: int = 10,
    seed: int = 42,
    max_retries: int = 100,
) -> list[np.ndarray]:

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels)
    n_classes = len(classes)
    n_total_samples = len(labels)

    class_indices: dict[int, np.ndarray] = {
        c: np.where(labels == c)[0] for c in classes
    }

    # Adaptively reduce min_samples if it's infeasible
    adaptive_min = max(1, min(min_samples, n_total_samples // (n_clients * 2)))
    if adaptive_min < min_samples:
        log.warning(
            "min_samples=%d is too strict for %d samples and %d clients. "
            "Reducing to %d.",
            min_samples, n_total_samples, n_clients, adaptive_min,
        )
        min_samples = adaptive_min

    for attempt in range(max_retries):
        client_indices: list[list[int]] = [[] for _ in range(n_clients)]

        for c in classes:
            idx = class_indices[c]
            rng.shuffle(idx)

            proportions = rng.dirichlet(alpha=np.full(n_clients, alpha))

            proportions = np.cumsum(proportions) * len(idx)
            splits = np.split(idx, proportions[:-1].astype(int))

            for cid, split in enumerate(splits):
                client_indices[cid].extend(split.tolist())

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) >= min_samples:
            result = [np.array(ci, dtype=np.int64) for ci in client_indices]
            log.info(
                "Dirichlet partition (α=%.2f, attempt=%d): %d clients, "
                "sizes min=%d max=%d mean=%.1f",
                alpha, attempt + 1, n_clients,
                min(sizes), max(sizes), np.mean(sizes),
            )
            _log_label_distribution(result, labels, n_classes)
            return result

        log.debug("Attempt %d: min_samples=%d < threshold=%d, retrying...", attempt + 1, min(sizes), min_samples)

    # Fallback: relax constraint if strict partitioning fails
    log.warning(
        "Could not find valid partition after %d attempts with min_samples=%d. "
        "Relaxing constraint to 1 sample per client.",
        max_retries, min_samples,
    )
    client_indices = [[] for _ in range(n_clients)]
    for c in classes:
        idx = class_indices[c]
        rng.shuffle(idx)
        proportions = rng.dirichlet(alpha=np.full(n_clients, alpha))
        proportions = np.cumsum(proportions) * len(idx)
        splits = np.split(idx, proportions[:-1].astype(int))
        for cid, split in enumerate(splits):
            client_indices[cid].extend(split.tolist())
    
    result = [np.array(ci, dtype=np.int64) for ci in client_indices if len(ci) > 0]
    sizes = [len(ci) for ci in result]
    log.info(
        "Fallback Dirichlet partition (α=%.2f): %d clients, "
        "sizes min=%d max=%d mean=%.1f",
        alpha, len(result),
        min(sizes), max(sizes), np.mean(sizes),
    )
    _log_label_distribution(result, labels, n_classes)
    return result

def _log_label_distribution(
    partitions: list[np.ndarray],
    labels: np.ndarray,
    n_classes: int,
) -> None:
    distributions = []
    for cid, idx in enumerate(partitions):
        if len(idx) == 0:
            continue
        client_labels = labels[idx]
        dist = np.bincount(client_labels, minlength=n_classes) / len(client_labels)
        distributions.append(dist)

    class0_fracs = [d[0] for d in distributions]
    log.info(
        "Label heterogeneity (class-0 fraction): mean=%.3f  std=%.3f  "
        "min=%.3f  max=%.3f",
        np.mean(class0_fracs), np.std(class0_fracs),
        np.min(class0_fracs), np.max(class0_fracs),
    )

def save_partitions(
    partitions: list[np.ndarray],
    save_dir: str | Path,
    prefix: str = "client",
) -> None:

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for cid, idx in enumerate(partitions):
        path = save_dir / f"{prefix}_{cid}_indices.npy"
        np.save(path, idx)

    log.info("Saved %d partition index files to %s", len(partitions), save_dir)

def load_partitions(
    load_dir: str | Path,
    n_clients: int,
    prefix: str = "client",
) -> list[np.ndarray]:

    load_dir = Path(load_dir)
    partitions = []

    for cid in range(n_clients):
        path = load_dir / f"{prefix}_{cid}_indices.npy"
        if not path.exists():
            raise FileNotFoundError(f"Partition file not found: {path}")
        partitions.append(np.load(path))

    log.info("Loaded %d partition index files from %s", n_clients, load_dir)
    return partitions

def partition_windows_and_labels(
    windows: np.ndarray,
    labels: np.ndarray,
    partitions: list[np.ndarray],
    save_dir: str | Path,
) -> None:

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for cid, idx in enumerate(partitions):
        np.save(save_dir / f"client_{cid}_windows.npy", windows[idx])
        np.save(save_dir / f"client_{cid}_labels.npy",  labels[idx])

    log.info(
        "Partitioned %d windows across %d clients → %s",
        len(windows), len(partitions), save_dir,
    )