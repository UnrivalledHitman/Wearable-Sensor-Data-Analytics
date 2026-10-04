"""Preprocessing: raw WESAD pickles -> windowed ``S*_combined.npz``.

Two pipelines share this entry point. ``mode = "clean"`` (preprocess_clean.py)
is the corrected one. The legacy one below is a faithful port of the notebook
that produced the results in the current paper draft. It deliberately keeps
that pipeline's known defects so the published numbers can be regenerated.

Known defects kept here on purpose:
  * WESAD labels 0 and 5-7 are mapped to class 0, so "baseline" is mostly
    undefined/transient data.
  * Windows with no dominant label also fall back to class 0.
  * Labels are block-averaged as floats, which invents labels at transitions.
  * ``int(700 / 32) == 21`` gives an effective rate of 33.3 Hz, not 32 Hz.
  * Block averaging smears the ECG; wrist BVP and chest EMG are unused.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import datasets
from .config import (
    CHANNEL_NAMES, CLASS_NAMES, LABEL_MAP, CleanPreprocessConfig, PreprocessConfig, RunConfig, upgrade_saved_config,
)
from .preprocess_clean import window_subject_clean

MANIFEST_NAME = "manifest.json"


def block_downsample_1d(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr.copy()
    n = len(arr) // factor
    if n == 0:
        return np.array([], dtype=arr.dtype)
    return arr[: n * factor].reshape(n, factor).mean(axis=1)


def acc_magnitude(acc: np.ndarray) -> np.ndarray:
    return np.sqrt((acc ** 2).sum(axis=1))


def majority_label(labels: np.ndarray, threshold: float) -> int:
    counts = np.bincount(np.asarray(labels))
    maj = int(np.argmax(counts))
    if counts[maj] / len(labels) >= threshold:
        return maj
    return 0  # baseline fallback


def _upsample(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) < 2:
        return np.zeros(target_len)
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, arr)


def window_subject(data: dict, cfg: PreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    """Turn one subject's raw WESAD dict into (windows, labels)."""
    signals = data["signal"]
    labels = np.array(data["label"], dtype=np.int32)

    chest = signals["chest"]
    ch_acc = np.array(chest["ACC"])
    ch_ecg = np.array(chest["ECG"]).squeeze()
    ch_eda = np.array(chest["EDA"]).squeeze()
    ch_resp = np.array(chest["Resp"]).squeeze()
    ch_temp = np.array(chest["Temp"]).squeeze()

    wrist = signals["wrist"]
    wr_acc = np.array(wrist["ACC"])
    wr_eda = np.array(wrist["EDA"]).squeeze()
    wr_temp = np.array(wrist["TEMP"]).squeeze()

    # Chest ECG is the reference length; wrist signals are stretched onto it.
    ref_len = len(ch_ecg)
    labels = labels[:ref_len]
    ch_acc = ch_acc[:ref_len]

    wr_acc = np.vstack([_upsample(wr_acc[:, i], ref_len) for i in range(3)]).T
    wr_eda = _upsample(wr_eda, ref_len)
    wr_temp = _upsample(wr_temp, ref_len)

    ds_factor = max(1, int(cfg.chest_rate / cfg.target_rate))
    channels = [
        block_downsample_1d(sig, ds_factor)
        for sig in (
            acc_magnitude(wr_acc), wr_eda, wr_temp,
            ch_ecg[:ref_len], ch_resp[:ref_len], acc_magnitude(ch_acc),
            ch_eda[:ref_len], ch_temp[:ref_len],
        )
    ]
    labels = block_downsample_1d(labels.astype(np.float32), ds_factor).astype(np.int32)

    min_len = min(len(labels), *(len(c) for c in channels))
    X_all = np.stack([c[:min_len] for c in channels], axis=1)
    labels = labels[:min_len]

    win_len = cfg.target_rate * cfg.window_sec
    step = int(win_len * (1 - cfg.overlap))

    X_list, y_list = [], []
    for start in range(0, len(X_all) - win_len, step):
        y_win = np.array([LABEL_MAP.get(int(l), 0) for l in labels[start:start + win_len]])
        X_list.append(X_all[start:start + win_len])
        y_list.append(majority_label(y_win, cfg.majority_threshold))

    X = np.array(X_list, dtype=np.float32).reshape(-1, win_len, len(CHANNEL_NAMES))
    y = np.array(y_list, dtype=np.int64)
    return X, y


def preprocess_all(cfg: RunConfig, force: bool = False) -> Path:
    """Preprocess every subject, unless data_dir already holds these settings."""
    manifest_path = cfg.data_dir / MANIFEST_NAME
    settings = asdict(cfg.preprocess)

    if not force and manifest_path.exists():
        if upgrade_saved_config(json.loads(manifest_path.read_text())["settings"]) == settings:
            print(f"Preprocessed data up to date: {cfg.data_dir}")
            return manifest_path

    clean = isinstance(cfg.preprocess, CleanPreprocessConfig)
    dataset = cfg.preprocess.dataset if clean else "wesad"
    pkl_files = datasets.subject_files(dataset, cfg.raw_dir)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    subjects = {}
    for pkl_path in pkl_files:
        subject = datasets.subject_name(pkl_path)
        data = datasets.load_subject(dataset, pkl_path, clean and cfg.preprocess.exclude_hyperventilation)
        raw_counts = Counter(np.asarray(data["label"]).astype(int).tolist())
        if clean:
            X, y, starts = window_subject_clean(data, cfg.preprocess)
            # Window start times (s) let calibration split a subject's recording in time.
            np.savez_compressed(cfg.data_dir / f"{subject}_combined.npz", X=X, y=y, starts=starts)
        else:
            X, y = window_subject(data, cfg.preprocess)
            np.savez_compressed(cfg.data_dir / f"{subject}_combined.npz", X=X, y=y)

        counts = np.bincount(y, minlength=len(CLASS_NAMES)).tolist()
        subjects[subject] = {
            "n_windows": int(len(y)),
            "window_counts": dict(zip(CLASS_NAMES, counts)),
            "raw_label_samples": {str(k): int(v) for k, v in sorted(raw_counts.items())},
        }
        print(f"  {subject}: {X.shape}, windows per class: {counts}")

    manifest_path.write_text(json.dumps({"settings": settings, "subjects": subjects}, indent=2))
    print(f"Preprocessing finished: {cfg.data_dir}")
    return manifest_path


def preprocessed_files(cfg: RunConfig) -> list[Path]:
    """Return the per-subject files, refusing data made with other settings."""
    files = sorted(cfg.data_dir.glob("*_combined.npz"))
    if not files:
        raise FileNotFoundError(f"No preprocessed files in {cfg.data_dir}; run the preprocess stage first.")

    manifest_path = cfg.data_dir / MANIFEST_NAME
    settings = asdict(cfg.preprocess)
    if manifest_path.exists():
        found = upgrade_saved_config(json.loads(manifest_path.read_text())["settings"])
        if found != settings:
            raise ValueError(
                f"{cfg.data_dir} was preprocessed with {found}, but this config asks for {settings}. "
                "Re-run the preprocess stage or point paths.data_dir elsewhere."
            )
    else:
        # Data written by the old notebook has no manifest; the window length is
        # the one setting that can still be checked from the arrays themselves.
        expected = cfg.preprocess.target_rate * cfg.preprocess.window_sec
        with np.load(files[0]) as arr:
            got = arr["X"].shape[1]
        if got != expected:
            raise ValueError(
                f"{files[0].name} has {got}-step windows but this config expects {expected}. "
                "Re-run the preprocess stage."
            )
    return files
