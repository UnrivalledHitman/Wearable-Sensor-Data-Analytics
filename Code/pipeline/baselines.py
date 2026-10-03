"""Reference baselines: hand-crafted features + classical classifiers under LOSO.

Classifiers and their settings follow the WESAD dataset paper (decision tree,
random forest, AdaBoost, LDA, kNN), so the results can be set beside its
reported 80 % (three-class) and 93 % (binary) accuracies. Trivial guessers
are reported with every result.
"""

from __future__ import annotations

import json
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import CODE_DIR
from .features import subject_features
from .wesad import load_subject, subject_files

META_COLUMNS = ["subject", "label", "start_sec"]

# WESAD label id -> class index. Binary follows the dataset paper:
# stress against non-stress (baseline + amusement); meditation is left out.
TASKS = {
    "binary": {"classes": ["non-stress", "stress"], "map": {1: 0, 3: 0, 2: 1}},
    "three_class": {"classes": ["baseline", "stress", "amusement"], "map": {1: 0, 2: 1, 3: 2}},
    "four_class": {"classes": ["baseline", "stress", "amusement", "meditation"], "map": {1: 0, 2: 1, 3: 2, 4: 3}},
}

_CHEST_PHYSIO = ["chest_ecg_", "chest_eda_", "chest_emg_", "chest_resp_", "chest_temp_"]
_WRIST_PHYSIO = ["wrist_bvp_", "wrist_eda_", "wrist_temp_"]
FEATURE_SETS = {
    "chest_physio": _CHEST_PHYSIO,
    "chest_all": _CHEST_PHYSIO + ["chest_acc_"],
    "wrist_physio": _WRIST_PHYSIO,
    "wrist_all": _WRIST_PHYSIO + ["wrist_acc_"],
    "all_physio": _CHEST_PHYSIO + _WRIST_PHYSIO,
    "all": _CHEST_PHYSIO + _WRIST_PHYSIO + ["chest_acc_", "wrist_acc_"],
}


def make_classifier(name: str, seed: int):
    tree = dict(criterion="entropy", min_samples_split=20, random_state=seed)
    if name == "DT":
        return DecisionTreeClassifier(**tree)
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, n_jobs=1, **tree)
    if name == "AB":
        return AdaBoostClassifier(DecisionTreeClassifier(**tree), n_estimators=100, random_state=seed)
    if name == "LDA":
        return LinearDiscriminantAnalysis()
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=9)
    raise ValueError(f"Unknown classifier {name!r}")


CLASSIFIERS = ("DT", "RF", "AB", "LDA", "kNN")


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    seed: int
    raw_dir: Path
    features_dir: Path
    out_dir: Path
    window_sec: list[int]
    stride_sec: float
    min_label_purity: float
    tasks: list[str]
    feature_sets: list[str]
    classifiers: list[str]
    subject_normalisation: list[bool]


def load_baseline_config(path: str | Path) -> BaselineConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else CODE_DIR / p

    cfg = BaselineConfig(
        name=raw["run"]["name"], seed=int(raw["run"]["seed"]),
        raw_dir=resolve(raw["paths"]["raw_dir"]),
        features_dir=resolve(raw["paths"]["features_dir"]),
        out_dir=resolve(raw["paths"]["out_dir"]),
        **raw["windows"], **raw["eval"],
    )
    for value, allowed, what in (
        (cfg.tasks, TASKS, "eval.tasks"), (cfg.feature_sets, FEATURE_SETS, "eval.feature_sets"),
        (cfg.classifiers, CLASSIFIERS, "eval.classifiers"),
    ):
        unknown = [v for v in value if v not in allowed]
        if unknown:
            raise ValueError(f"Unknown {what} {unknown}; choose from {list(allowed)}")
    return cfg


# --------------------------------------------------------------------------- features

def features_path(cfg: BaselineConfig, window_sec: int) -> Path:
    return cfg.features_dir / f"features_w{window_sec}_s{cfg.stride_sec:g}_p{cfg.min_label_purity:g}.csv.gz"


def extract_features(cfg: BaselineConfig, window_sec: int, n_jobs: int = 1) -> pd.DataFrame:
    """Feature table for every subject, computed once and cached."""
    path = features_path(cfg, window_sec)
    if path.exists():
        return pd.read_csv(path)

    def one(pkl_path: Path) -> pd.DataFrame:
        return subject_features(load_subject(pkl_path), pkl_path.stem, window_sec, cfg.stride_sec, cfg.min_label_purity)

    files = subject_files(cfg.raw_dir)
    print(f"Extracting {window_sec} s features for {len(files)} subjects ...")
    table = pd.concat(Parallel(n_jobs=n_jobs)(delayed(one)(f) for f in files), ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(f"Features saved: {path} {table.shape}")
    return table


def standardise_per_subject(table: pd.DataFrame) -> pd.DataFrame:
    """z-score every feature within each subject, over all of that subject's
    windows (labelled or not), so no label information is used."""
    features = [c for c in table.columns if c not in META_COLUMNS]
    grouped = table.groupby("subject")[features]
    out = table.copy()
    out[features] = (table[features] - grouped.transform("mean")) / grouped.transform("std").replace(0, 1.0)
    return out


# --------------------------------------------------------------------------- evaluation

def _fold(model, X: np.ndarray, y: np.ndarray, test_mask: np.ndarray, n_classes: int) -> dict:
    model = clone(model).fit(X[~test_mask], y[~test_mask])
    y_true, y_pred = y[test_mask], model.predict(X[test_mask])
    labels = list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def guesser_reference(y: np.ndarray, n_classes: int) -> dict:
    """Always-majority and uniform-random guessers on the pooled windows."""
    share = np.bincount(y, minlength=n_classes) / len(y)
    majority = int(share.argmax())
    p = share[majority]
    random_f1 = np.mean([2 * s / (n_classes * s + 1) for s in share])  # precision = s, recall = 1/k
    return {
        "class_share": share.round(4).tolist(),
        "majority_accuracy": float(p), "majority_f1_macro": float((2 * p / (1 + p)) / n_classes),
        "random_accuracy": 1.0 / n_classes, "random_f1_macro": float(random_f1),
    }


def evaluate(cfg: BaselineConfig, n_jobs: int = -1, max_minutes: float | None = None) -> pd.DataFrame:
    """Run every (window, task, feature set, normalisation, classifier) under LOSO.

    Each combination is saved as soon as it finishes, so an interrupted run
    continues where it stopped. With ``max_minutes``, no new combination is
    started after that time; run again to do the rest.
    """
    cache_dir = cfg.out_dir / "combinations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows, pending, started = [], 0, time.perf_counter()

    for window_sec in cfg.window_sec:
        raw_table = extract_features(cfg, window_sec, n_jobs=min(4, n_jobs) if n_jobs > 0 else 4)
        tables = {False: raw_table}
        if True in cfg.subject_normalisation:
            tables[True] = standardise_per_subject(raw_table)

        for task in cfg.tasks:
            spec = TASKS[task]
            n_classes = len(spec["classes"])
            for normalised in cfg.subject_normalisation:
                table = tables[normalised]
                table = table[table["label"].isin(list(spec["map"]))]
                y = table["label"].map(spec["map"]).to_numpy()
                subjects = table["subject"].to_numpy()
                order = sorted(np.unique(subjects), key=lambda s: int(s.lstrip("S")))
                reference = guesser_reference(y, n_classes)

                for feature_set in cfg.feature_sets:
                    columns = [c for c in table.columns if c.startswith(tuple(FEATURE_SETS[feature_set]))]
                    X = table[columns].to_numpy(dtype=np.float64)

                    for clf_name in cfg.classifiers:
                        key = f"w{window_sec}_{task}_{feature_set}_{'subjnorm' if normalised else 'raw'}_{clf_name}"
                        cache = cache_dir / f"{key}.json"
                        if cache.exists():
                            rows.append(json.loads(cache.read_text()))
                            continue
                        if max_minutes is not None and time.perf_counter() - started > max_minutes * 60:
                            pending += 1
                            continue

                        model = make_pipeline(
                            SimpleImputer(strategy="median"), StandardScaler(), make_classifier(clf_name, cfg.seed)
                        )
                        folds = Parallel(n_jobs=n_jobs)(
                            delayed(_fold)(model, X, y, subjects == s, n_classes) for s in order
                        )
                        acc = np.array([f["accuracy"] for f in folds])
                        f1 = np.array([f["f1_macro"] for f in folds])
                        pooled = np.sum([f["confusion_matrix"] for f in folds], axis=0)
                        row = {
                            "window_sec": window_sec, "task": task, "feature_set": feature_set,
                            "subject_normalisation": normalised, "classifier": clf_name,
                            "n_features": len(columns), "n_windows": int(len(y)),
                            "accuracy_mean": float(acc.mean()), "accuracy_std": float(acc.std(ddof=1)),
                            "f1_macro_mean": float(f1.mean()), "f1_macro_std": float(f1.std(ddof=1)),
                            "accuracy_pooled": float(np.trace(pooled) / pooled.sum()),
                            **reference,
                            "per_subject": {s: f for s, f in zip(order, folds)},
                        }
                        cache.write_text(json.dumps(row, indent=2))
                        rows.append(row)
                        print(f"{key:<58s} acc={row['accuracy_mean']:.3f}  f1={row['f1_macro_mean']:.3f}"
                              f"   (majority acc={reference['majority_accuracy']:.3f})")

    summary = pd.DataFrame([{k: v for k, v in r.items() if k not in ("per_subject", "class_share")} for r in rows])
    summary.to_csv(cfg.out_dir / "summary.csv", index=False)
    if pending:
        print(f"\nTime budget reached: {pending} combinations still to run. Run the same command again.")
    return summary


def print_best(summary: pd.DataFrame) -> None:
    """Best classifier per task, feature set and normalisation, by mean LOSO accuracy."""
    best = summary.loc[summary.groupby(
        ["window_sec", "task", "subject_normalisation", "feature_set"])["accuracy_mean"].idxmax()]
    cols = ["window_sec", "task", "subject_normalisation", "feature_set", "classifier",
            "accuracy_mean", "accuracy_std", "f1_macro_mean", "majority_accuracy", "random_accuracy"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(best[cols].to_string(index=False))
