"""Stress-Predict dataset (Iqbal et al., Sensors 2022) in WESAD's in-memory format.

35 volunteers wore an Empatica E4 (the same wristband as in WESAD) through a
Stroop test, an interview and a hyperventilation task, with rest between
them. Labels come from the authors' per-second file
``Processed_data/Improved_All_Combined_hr_rsp_binary.csv`` (0 = rest,
1 = stress task). The raw tag files are not used: they lack task names and
some hold stray timestamps. Participant S01 has no labels and is skipped.

``load_subject`` returns the same structure as a WESAD pickle (wrist signals
only) with labels at WESAD's 700 Hz clock: 1 = rest, 2 = stress, 0 = outside
the labelled period, 6 = hyperventilation when it is excluded (WESAD's
"ignore" range), so the rest of the pipeline works unchanged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .wesad import CHEST_RATE, LABEL_BASELINE, LABEL_STRESS

LABELS_FILE = Path("Processed_data") / "Improved_All_Combined_hr_rsp_binary.csv"
LABEL_IGNORED = 6
STRESS_TASKS = ("stroop", "interview", "hyperventilation")  # the three stress blocks, in protocol order


@lru_cache(maxsize=4)
def _labels(raw_dir: str) -> pd.DataFrame:
    path = Path(raw_dir) / LABELS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Stress-Predict labels not found at {path}")
    return pd.read_csv(path)


def subject_files(raw_dir: Path) -> list[Path]:
    """Participant folders (Raw_data/S02 ...) that have labels."""
    labelled = {f"S{int(p):02d}" for p in _labels(str(raw_dir))["Participant"].unique()}
    folders = sorted((Path(raw_dir) / "Raw_data").glob("S*"), key=lambda p: int(p.name.lstrip("S")))
    if not folders:
        raise FileNotFoundError(f"No Stress-Predict participant folders in {Path(raw_dir) / 'Raw_data'}")
    return [f for f in folders if f.name in labelled]


def _read_e4(path: Path) -> tuple[float, float, np.ndarray]:
    """Start time (unix s), sample rate and samples of one Empatica E4 CSV."""
    raw = pd.read_csv(path, header=None)
    return float(raw.iloc[0, 0]), float(raw.iloc[1, 0]), raw.iloc[2:].to_numpy(dtype=np.float64)


def stress_blocks(labels: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) indices of each run of stress labels, in order."""
    padded = np.r_[0, (labels == 1).astype(int), 0]
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def load_subject(folder: Path, exclude_hyperventilation: bool = False) -> dict:
    folder = Path(folder)
    signals, t0 = {}, None
    for name in ("ACC", "BVP", "EDA", "TEMP"):
        start, _, samples = _read_e4(folder / f"{name}.csv")
        t0 = start if t0 is None else t0
        if start != t0:
            raise ValueError(f"{folder.name}: {name}.csv starts at {start}, other signals at {t0}")
        signals[name] = samples if name == "ACC" else samples[:, :1]

    duration = len(signals["BVP"]) / 64.0
    label = np.zeros(int(duration * CHEST_RATE), dtype=np.int64)

    rows = _labels(str(folder.parents[1]))
    rows = rows[rows["Participant"] == int(folder.name.lstrip("S"))].sort_values("Time(sec)")
    per_second = rows["Label"].to_numpy()
    wesad_id = np.where(per_second == 1, LABEL_STRESS, LABEL_BASELINE)

    blocks = stress_blocks(per_second)
    if len(blocks) != len(STRESS_TASKS):
        raise ValueError(f"{folder.name}: expected {len(STRESS_TASKS)} stress blocks, found {len(blocks)}")
    if exclude_hyperventilation:
        a, b = blocks[STRESS_TASKS.index("hyperventilation")]
        wesad_id[a:b] = LABEL_IGNORED

    offsets = np.round(rows["Time(sec)"].to_numpy() - t0).astype(np.int64)
    for second, value in zip(offsets, wesad_id):
        a, b = second * CHEST_RATE, (second + 1) * CHEST_RATE
        if 0 <= a and b <= len(label):
            label[a:b] = value

    return {"subject": folder.name, "label": label, "signal": {"wrist": signals}}
