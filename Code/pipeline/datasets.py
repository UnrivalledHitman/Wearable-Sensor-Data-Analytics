"""One entry point for every dataset: list its subjects and load one."""

from __future__ import annotations

from pathlib import Path

from . import stresspredict, wesad


def subject_files(dataset: str, raw_dir: Path) -> list[Path]:
    if dataset == "stress_predict":
        return stresspredict.subject_files(raw_dir)
    return wesad.subject_files(raw_dir)


def load_subject(dataset: str, path: Path, exclude_hyperventilation: bool = False) -> dict:
    if dataset == "stress_predict":
        return stresspredict.load_subject(path, exclude_hyperventilation)
    return wesad.load_subject(path)


def subject_name(path: Path) -> str:
    """S2 for WESAD's S2/S2.pkl, S02 for Stress-Predict's Raw_data/S02."""
    return path.stem
