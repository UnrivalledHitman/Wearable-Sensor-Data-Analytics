"""Leave-one-subject-out training and per-fold test evaluation."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .config import CLASS_NAMES, CODE_DIR, NUM_CLASSES, RunConfig
from .model import build_model, count_parameters
from .preprocess import preprocessed_files


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def subject_of(npz_path: Path) -> str:
    return npz_path.stem.split("_")[0]


def channel_mean_std(npz_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over the given files, one file in memory at a time."""
    total, s, ss = 0, None, None
    for p in npz_paths:
        with np.load(p) as arr:
            flat = arr["X"].astype(np.float64).reshape(-1, arr["X"].shape[-1])
        s = flat.sum(axis=0) if s is None else s + flat.sum(axis=0)
        ss = (flat ** 2).sum(axis=0) if ss is None else ss + (flat ** 2).sum(axis=0)
        total += flat.shape[0]

    mean = (s / total).astype(np.float32)
    var = (ss / total) - mean.astype(np.float64) ** 2
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    return mean, std


def load_normalised(npz_paths: list[Path], mean, std) -> tuple[torch.Tensor, torch.Tensor]:
    Xs, ys = [], []
    for p in npz_paths:
        with np.load(p) as arr:
            X = arr["X"].astype(np.float32)
            y = arr["y"].astype(np.int64)
        Xs.append(torch.from_numpy((X - mean[None, None, :]) / (std[None, None, :] + 1e-9)))
        ys.append(torch.from_numpy(y))
    return torch.cat(Xs), torch.cat(ys)


def balanced_class_weights(y: torch.Tensor) -> torch.Tensor:
    y_np = y.numpy()
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    present = np.unique(y_np)
    for c, w in zip(present, compute_class_weight("balanced", classes=present, y=y_np)):
        weights[int(c)] = float(w)
    return torch.from_numpy(weights)


def balanced_sampler(y: torch.Tensor, generator: torch.Generator) -> WeightedRandomSampler:
    y_np = y.numpy()
    class_weights = 1.0 / np.maximum(np.bincount(y_np, minlength=NUM_CLASSES), 1)
    sample_weights = torch.tensor(class_weights[y_np], dtype=torch.double)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)


def _mean_loss(model, loader, criterion, device, use_amp) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.autocast(device.type, enabled=use_amp):
                loss = criterion(model(xb), yb)
            total += loss.item() * xb.size(0)
            n += xb.size(0)
    return total / max(1, n)


def train_fold(cfg: RunConfig, model, X_train, y_train, X_val, y_val, ckpt: Path, device, generator, tag: str):
    """Train with early stopping on validation loss; best weights go to ``ckpt``."""
    tc = cfg.train
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    if tc.balancing == "weighted_sampler":
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=tc.batch_size,
            sampler=balanced_sampler(y_train, generator),
        )
    else:
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=tc.batch_size, shuffle=True, generator=generator,
        )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=tc.batch_size, shuffle=False)

    weight = balanced_class_weights(y_train).to(device) if tc.balancing == "weighted_loss" else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr)
    scheduler = None
    if tc.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(1, tc.patience // 2)
        )

    best_val, epochs_no_improve, epochs_run = float("inf"), 0, 0
    for epoch in range(tc.epochs):
        model.train()
        train_loss, n_train = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.autocast(device.type, enabled=use_amp):
                loss = criterion(model(xb), yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * xb.size(0)
            n_train += xb.size(0)
        train_loss /= max(1, n_train)

        val_loss = _mean_loss(model, val_loader, criterion, device, use_amp)
        if scheduler is not None:
            scheduler.step(val_loss)
        epochs_run = epoch + 1
        print(f"[{tag}] Epoch {epochs_run:02d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val - tc.min_delta:
            best_val, epochs_no_improve = val_loss, 0
            torch.save({"model_state": model.state_dict()}, ckpt)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= tc.patience:
                print(f"[{tag}] Early stopping")
                break

    return best_val, epochs_run


def predict(model, X, batch_size: int, device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False):
            preds.append(model(xb.to(device)).argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(NUM_CLASSES))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=CLASS_NAMES, output_dict=True, zero_division=0,
        ),
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=CODE_DIR, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_manifest(cfg: RunConfig, device: torch.device) -> None:
    manifest = {
        "config": cfg.to_dict(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    (cfg.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def run_loso(cfg: RunConfig, device: torch.device | None = None) -> list[dict]:
    """Train and test one model per held-out subject; write ``fold_<subject>.json``."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = preprocessed_files(cfg)

    ckpt_dir = cfg.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(cfg, device)

    results = []
    for fold_idx, test_file in enumerate(files):
        test_subj = subject_of(test_file)
        print("\n" + "=" * 80)
        print(f"[{cfg.name}] FOLD {fold_idx + 1}/{len(files)} - test subject {test_subj}")
        print("=" * 80)

        # One seed per fold so any single fold can be re-run on its own.
        fold_seed = cfg.seed + fold_idx
        set_seed(fold_seed)
        generator = torch.Generator().manual_seed(fold_seed)

        train_files = [f for f in files if f != test_file]
        mean, std = channel_mean_std(train_files)
        X_all, y_all = load_normalised(train_files, mean, std)
        X_test, y_test = load_normalised([test_file], mean, std)

        idx = torch.randperm(len(X_all), generator=generator)
        n_val = max(1, int(cfg.train.val_fraction * len(idx)))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        model = build_model(X_all.shape[2], cfg.model).to(device)
        n_params = count_parameters(model)
        print(f"Model built - {n_params:,} trainable parameters")

        ckpt = ckpt_dir / f"best_{test_subj}.pt"
        t0 = time.perf_counter()
        best_val, epochs_run = train_fold(
            cfg, model, X_all[train_idx], y_all[train_idx], X_all[val_idx], y_all[val_idx],
            ckpt, device, generator, test_subj,
        )
        train_seconds = time.perf_counter() - t0

        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
        y_pred = predict(model, X_test, cfg.train.test_batch_size or cfg.train.batch_size, device)

        result = {
            "subject": test_subj,
            **fold_metrics(y_test.numpy(), y_pred),
            "best_val_loss": float(best_val),
            "epochs_run": epochs_run,
            "train_seconds": round(train_seconds, 1),
            "n_parameters": n_params,
            "seed": fold_seed,
        }
        (cfg.out_dir / f"fold_{test_subj}.json").write_text(json.dumps(result, indent=2))
        results.append(result)
        print(f"[{test_subj}] TEST | acc={result['accuracy']:.4f} | f1_macro={result['f1_macro']:.4f}")

        del model, X_all, y_all, X_test, y_test
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results
