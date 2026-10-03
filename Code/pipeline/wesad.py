"""Raw WESAD access: file discovery, loading and the dataset's constants.

Facts taken from the dataset paper (Schmidt et al., ICMI 2018) and the
readme shipped with the data:
  * chest RespiBAN: every signal at 700 Hz;
  * wrist Empatica E4: ACC 32 Hz, BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz;
  * labels at 700 Hz: 0 = not defined / transient, 1 = baseline, 2 = stress,
    3 = amusement, 4 = meditation, 5-7 = to be ignored.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np

CHEST_RATE = 700
WRIST_RATES = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}

LABEL_BASELINE, LABEL_STRESS, LABEL_AMUSEMENT, LABEL_MEDITATION = 1, 2, 3, 4
CONDITION_LABELS = (LABEL_BASELINE, LABEL_STRESS, LABEL_AMUSEMENT, LABEL_MEDITATION)


def subject_files(raw_dir: Path) -> list[Path]:
    """Subject pickles in either layout (``S2.pkl`` or ``S2/S2.pkl``), S2..S17."""
    raw_dir = Path(raw_dir)
    files = list(raw_dir.glob("S*.pkl")) + list(raw_dir.glob("S*/S*.pkl"))
    if not files:
        raise FileNotFoundError(
            f"No WESAD subject pickles (S*.pkl) found in {raw_dir}. "
            "Download WESAD and place it there, or set paths.raw_dir in the config."
        )
    return sorted(files, key=lambda p: int(p.stem.lstrip("S")))


def load_subject(pkl_path: Path) -> dict:
    """The dataset's own dict: ``signal`` -> ``chest``/``wrist`` -> arrays, plus ``label``."""
    with open(pkl_path, "rb") as fh, warnings.catch_warnings():
        # The pickles predate NumPy 2 and trigger a harmless dtype deprecation notice.
        warnings.simplefilter("ignore")
        return pickle.load(fh, encoding="latin1")


def chest(data: dict, name: str) -> np.ndarray:
    """One chest signal as float64; 1-D except ACC, which is (n, 3)."""
    return np.asarray(data["signal"]["chest"][name], dtype=np.float64).squeeze()


def wrist(data: dict, name: str) -> np.ndarray:
    return np.asarray(data["signal"]["wrist"][name], dtype=np.float64).squeeze()


def labels(data: dict) -> np.ndarray:
    return np.asarray(data["label"], dtype=np.int64)


def pure_windows(lab: np.ndarray, rate: float, window_sec: float, stride_sec: float,
                 min_purity: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Window start times (s) and WESAD label for windows dominated by one condition.

    A window is kept only if a single condition label (1-4) covers at least
    ``min_purity`` of it. Undefined (0) and to-be-ignored (5-7) samples never
    produce a window of their own and count against purity.
    """
    win = int(round(window_sec * rate))
    step = max(1, int(round(stride_sec * rate)))
    starts, window_labels = [], []
    for start in range(0, len(lab) - win + 1, step):
        counts = np.bincount(lab[start:start + win], minlength=8)
        best = max(CONDITION_LABELS, key=lambda c: counts[c])
        if counts[best] / win >= min_purity:
            starts.append(start / rate)
            window_labels.append(best)
    return np.array(starts, dtype=np.float64), np.array(window_labels, dtype=np.int64)
