"""Corrected preprocessing (milestone M1).

Differences from the legacy pipeline, each fixing a defect found in review:
  * only windows lying inside one protocol condition (labels 1-4) are kept;
    undefined (0) and to-be-ignored (5-7) data never becomes a training window;
  * every signal is brought to the target rate on its true time axis, using
    anti-aliased polyphase resampling when downsampling;
  * wrist BVP and chest EMG are available, and accelerometer axes are kept;
  * optional per-subject standardisation, using that subject's whole recording
    and no labels, to remove between-person offsets in EDA and temperature.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .config import LABEL_MAP, CleanPreprocessConfig
from .wesad import CHEST_RATE, WRIST_RATES, chest, labels, pure_windows, wrist

# channel name -> (loader, signal key, source rate, axis or None)
_SOURCES = {
    "ch_ecg": (chest, "ECG", CHEST_RATE, None),
    "ch_eda": (chest, "EDA", CHEST_RATE, None),
    "ch_emg": (chest, "EMG", CHEST_RATE, None),
    "ch_resp": (chest, "Resp", CHEST_RATE, None),
    "ch_temp": (chest, "Temp", CHEST_RATE, None),
    "ch_acc_x": (chest, "ACC", CHEST_RATE, 0),
    "ch_acc_y": (chest, "ACC", CHEST_RATE, 1),
    "ch_acc_z": (chest, "ACC", CHEST_RATE, 2),
    "wr_bvp": (wrist, "BVP", WRIST_RATES["BVP"], None),
    "wr_eda": (wrist, "EDA", WRIST_RATES["EDA"], None),
    "wr_temp": (wrist, "TEMP", WRIST_RATES["TEMP"], None),
    "wr_acc_x": (wrist, "ACC", WRIST_RATES["ACC"], 0),
    "wr_acc_y": (wrist, "ACC", WRIST_RATES["ACC"], 1),
    "wr_acc_z": (wrist, "ACC", WRIST_RATES["ACC"], 2),
}


def to_rate(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a 1-D signal, keeping its duration."""
    if src_rate == dst_rate:
        return x
    if src_rate > dst_rate:
        g = gcd(src_rate, dst_rate)
        return resample_poly(x, dst_rate // g, src_rate // g, padtype="line")
    # Slow signals (4 Hz EDA/TEMP, 32 Hz ACC) carry nothing above their own
    # Nyquist rate, so linear interpolation is enough and cannot ring.
    n_out = int(len(x) * dst_rate / src_rate)
    return np.interp(np.arange(n_out) / dst_rate, np.arange(len(x)) / src_rate, x)


def resample_recording(data: dict, cfg: CleanPreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    """Whole recording at the target rate: signals (T, C) and WESAD labels (T,)."""
    channels = []
    for name in cfg.channels:
        loader, key, rate, axis = _SOURCES[name]
        sig = loader(data, key)
        channels.append(to_rate(sig if axis is None else sig[:, axis], rate, cfg.target_rate))

    lab = labels(data)
    lengths = [len(c) for c in channels]
    expected = len(lab) * cfg.target_rate / CHEST_RATE
    if max(abs(n - expected) for n in lengths) > cfg.target_rate:
        raise ValueError(f"Resampled channel lengths {lengths} disagree with the label track ({expected:.0f}).")

    n = min(min(lengths), int(expected))
    signals = np.stack([c[:n] for c in channels], axis=1)
    lab_at_rate = lab[(np.arange(n) * CHEST_RATE / cfg.target_rate).astype(np.int64)]
    return signals, lab_at_rate


def window_subject_clean(data: dict, cfg: CleanPreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    """Turn one subject's raw WESAD dict into (windows, class indices)."""
    signals, lab = resample_recording(data, cfg)

    if cfg.subject_normalisation:
        mean, std = signals.mean(axis=0), signals.std(axis=0)
        signals = (signals - mean) / np.maximum(std, 1e-8)

    starts, window_labels = pure_windows(lab, cfg.target_rate, cfg.window_sec, cfg.stride_sec, cfg.min_label_purity)
    win = int(round(cfg.window_sec * cfg.target_rate))
    idx = np.round(starts * cfg.target_rate).astype(np.int64)

    X = np.stack([signals[i:i + win] for i in idx]).astype(np.float32) if len(idx) else \
        np.zeros((0, win, len(cfg.channels)), dtype=np.float32)
    y = np.array([LABEL_MAP[int(l)] for l in window_labels], dtype=np.int64)
    return X, y
