"""Reference baselines: hand-crafted features + classical classifiers under LOSO.

Classifiers and their settings follow the WESAD dataset paper (decision tree,
random forest, AdaBoost, LDA, kNN), so the results can be set beside its
reported 80 % (three-class) and 93 % (binary) accuracies. Trivial guessers
are reported with every result.

Beyond plain LOSO it supports (milestone M5): per-subject normalisation from
only part of the recording, per-user calibration with a few labelled minutes
of the test subject, and training on one dataset while testing on another.
"""

from __future__ import annotations

import json
import os
import time
import tomllib
from dataclasses import dataclass, field
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
from . import datasets
from .features import subject_features

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
    # Per-subject feature standardisation, one entry per variant to evaluate:
    # "none", "recording" (all of the subject's windows), "first_minutes:N"
    # (windows inside the first N minutes) or "rest_minutes:N" (the first N
    # minutes of the first rest/baseline block).
    normalisation: list[str]
    dataset: str = "wesad"
    exclude_hyperventilation: bool = False
    # Per-user calibration: add the first N minutes of each condition of the
    # test subject to training, weighted calibration_weight each.
    calibration_minutes: list[float] = field(default_factory=list)
    calibration_weight: float = 10.0
    # Cross-dataset: train on all of `dataset`, test on every subject of this one.
    test_dataset: str | None = None
    test_raw_dir: Path | None = None
    test_exclude_hyperventilation: bool = False


def load_baseline_config(path: str | Path) -> BaselineConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else CODE_DIR / p

    paths, ev = raw["paths"], dict(raw["eval"])
    if "subject_normalisation" in ev:  # the original boolean form
        ev["normalisation"] = ["recording" if v else "none" for v in ev.pop("subject_normalisation")]
    data = dict(raw.get("data", {}))
    cfg = BaselineConfig(
        name=raw["run"]["name"], seed=int(raw["run"]["seed"]),
        raw_dir=resolve(paths["raw_dir"]),
        features_dir=resolve(paths["features_dir"]),
        out_dir=resolve(paths["out_dir"]),
        test_raw_dir=resolve(paths["test_raw_dir"]) if "test_raw_dir" in paths else None,
        **raw["windows"], **ev, **data,
    )
    for value, allowed, what in (
        (cfg.tasks, TASKS, "eval.tasks"), (cfg.feature_sets, FEATURE_SETS, "eval.feature_sets"),
        (cfg.classifiers, CLASSIFIERS, "eval.classifiers"),
    ):
        unknown = [v for v in value if v not in allowed]
        if unknown:
            raise ValueError(f"Unknown {what} {unknown}; choose from {list(allowed)}")
    for spec in cfg.normalisation:
        normalisation_tag(spec)  # validates
    if (cfg.test_dataset is None) != (cfg.test_raw_dir is None):
        raise ValueError("Cross-dataset runs need both data.test_dataset and paths.test_raw_dir")
    return cfg


# --------------------------------------------------------------------------- features

def features_path(features_dir: Path, dataset: str, exclude_hv: bool, window_sec: int,
                  stride_sec: float, purity: float) -> Path:
    prefix = "" if dataset == "wesad" else f"{dataset}{'_nohv' if exclude_hv else ''}_"
    return Path(features_dir) / f"{prefix}features_w{window_sec}_s{stride_sec:g}_p{purity:g}.csv.gz"


def extract_features(cfg: BaselineConfig, window_sec: int, n_jobs: int = 1, test: bool = False) -> pd.DataFrame:
    """Feature table for every subject of the training (or, with ``test``, the
    cross-dataset test) dataset, computed once and cached."""
    dataset = cfg.test_dataset if test else cfg.dataset
    raw_dir = cfg.test_raw_dir if test else cfg.raw_dir
    exclude_hv = cfg.test_exclude_hyperventilation if test else cfg.exclude_hyperventilation
    path = features_path(cfg.features_dir, dataset, exclude_hv, window_sec, cfg.stride_sec, cfg.min_label_purity)
    if path.exists():
        return pd.read_csv(path)

    def one(subject_path: Path) -> pd.DataFrame:
        data = datasets.load_subject(dataset, subject_path, exclude_hv)
        return subject_features(data, datasets.subject_name(subject_path), window_sec, cfg.stride_sec,
                                cfg.min_label_purity)

    files = datasets.subject_files(dataset, raw_dir)
    print(f"Extracting {window_sec} s features for {len(files)} {dataset} subjects ...")
    table = pd.concat(Parallel(n_jobs=n_jobs)(delayed(one)(f) for f in files), ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(f"Features saved: {path} {table.shape}")
    return table


def normalisation_tag(spec: str) -> str:
    """Short name used in result keys: raw, subjnorm, first5, rest5 ..."""
    if spec == "none":
        return "raw"
    if spec == "recording":
        return "subjnorm"
    kind, _, minutes = spec.partition(":")
    if kind in ("first_minutes", "rest_minutes") and minutes:
        return f"{kind.split('_')[0]}{float(minutes):g}"
    raise ValueError(f"Unknown normalisation {spec!r}; use none, recording, first_minutes:N or rest_minutes:N")


def _first_block(starts: np.ndarray, stride_sec: float) -> np.ndarray:
    """Positions (into sorted ``starts``) of the first run of evenly spaced windows."""
    order = np.argsort(starts)
    gaps = np.flatnonzero(np.diff(starts[order]) > stride_sec * 1.5)
    end = gaps[0] + 1 if len(gaps) else len(order)
    return order[:end]


def _reference_rows(group: pd.DataFrame, spec: str, window_sec: float, stride_sec: float) -> pd.DataFrame:
    if spec == "recording":
        return group
    kind, _, minutes = spec.partition(":")
    limit = float(minutes) * 60
    if kind == "first_minutes":
        ref = group[group["start_sec"] + window_sec <= group["start_sec"].min() + limit]
    else:
        rest = group[group["label"] == 1]
        block = rest.iloc[_first_block(rest["start_sec"].to_numpy(), stride_sec)]
        ref = block[block["start_sec"] + window_sec <= block["start_sec"].min() + limit]
    return ref if len(ref) else group.nsmallest(1, "start_sec")


def standardise_per_subject(table: pd.DataFrame, spec: str = "recording", window_sec: float = 60,
                            stride_sec: float = 5) -> pd.DataFrame:
    """z-score every feature within each subject, using that subject's statistics
    from the part of the recording ``spec`` names. Only "rest_minutes" uses a
    label, and only to find the rest period."""
    if spec == "none":
        return table
    features = [c for c in table.columns if c not in META_COLUMNS]
    out = table.copy()
    out[features] = out[features].astype(np.float64)
    for _, group in table.groupby("subject"):
        ref = _reference_rows(group, spec, window_sec, stride_sec)[features]
        std = ref.std().replace(0, 1.0).fillna(1.0)
        out.loc[group.index, features] = (group[features] - ref.mean()) / std
    return out


# --------------------------------------------------------------------------- evaluation

def _scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    labels = list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _fold(model, X: np.ndarray, y: np.ndarray, test_mask: np.ndarray, n_classes: int) -> dict:
    model = clone(model).fit(X[~test_mask], y[~test_mask])
    return _scores(y[test_mask], model.predict(X[test_mask]), n_classes)


def calibration_split(starts: np.ndarray, y: np.ndarray, minutes: float, window_sec: float,
                      stride_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Masks over one subject's windows: calibration windows (inside the first
    ``minutes`` of the first block of each class) and test windows (all the
    others that do not overlap a calibration period)."""
    calib = np.zeros(len(y), dtype=bool)
    overlap = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        block = idx[_first_block(starts[idx], stride_sec)]
        t0 = starts[block].min()
        t1 = t0 + minutes * 60
        calib[block[starts[block] + window_sec <= t1]] = True
        overlap |= (starts < t1) & (starts + window_sec > t0) & (y == c)
    return calib, ~(calib | overlap)


def _calibration_fold(model, X, y, subjects, starts, subject, minutes, window_sec, stride_sec,
                      weight, n_classes) -> dict:
    own = np.flatnonzero(subjects == subject)
    calib_local, test_local = calibration_split(starts[own], y[own], minutes, window_sec, stride_sec)
    calib, test = own[calib_local], own[test_local]
    train = np.flatnonzero(subjects != subject)
    final_step = model.steps[-1][0]

    before = clone(model).fit(X[train], y[train])
    after_idx = np.r_[train, calib]
    sample_weight = np.r_[np.ones(len(train)), np.full(len(calib), weight)]
    try:
        after = clone(model).fit(X[after_idx], y[after_idx], **{f"{final_step}__sample_weight": sample_weight})
    except TypeError:  # kNN and LDA take no sample weights: repeat the calibration rows instead
        repeats = np.r_[train, np.repeat(calib, max(1, int(round(weight))))]
        after = clone(model).fit(X[repeats], y[repeats])
    return {
        "n_calibration": int(len(calib)), "n_test": int(len(test)),
        "before": _scores(y[test], before.predict(X[test]), n_classes),
        **_scores(y[test], after.predict(X[test]), n_classes),
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


def _task_table(table: pd.DataFrame, task: str):
    spec = TASKS[task]
    table = table[table["label"].isin(list(spec["map"]))]
    return table, table["label"].map(spec["map"]).to_numpy()


def _summarise(folds: dict, extra: dict) -> dict:
    acc = np.array([f["accuracy"] for f in folds.values()])
    f1 = np.array([f["f1_macro"] for f in folds.values()])
    pooled = np.sum([f["confusion_matrix"] for f in folds.values()], axis=0)
    row = {
        **extra,
        "accuracy_mean": float(acc.mean()), "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
        "f1_macro_mean": float(f1.mean()), "f1_macro_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
        "accuracy_pooled": float(np.trace(pooled) / pooled.sum()),
    }
    if all("before" in f for f in folds.values()):
        row["accuracy_before_mean"] = float(np.mean([f["before"]["accuracy"] for f in folds.values()]))
        row["f1_macro_before_mean"] = float(np.mean([f["before"]["f1_macro"] for f in folds.values()]))
        row["n_calibration_mean"] = float(np.mean([f["n_calibration"] for f in folds.values()]))
    row["per_subject"] = folds
    return row


def _tagged(subject, fn, *args):
    return subject, fn(*args)


def _cached_folds(fold_dir: Path, order: list[str], n_jobs: int, task, deadline: float | None = None) -> dict | None:
    """Run one combination's folds in parallel, saving each fold as it finishes.

    ``task(subject)`` returns (function, *arguments) for that subject's fold.
    Folds already saved are reused, so a combination interrupted part-way
    continues instead of starting again. Folds run in batches of ``n_jobs``;
    after ``deadline`` (a perf_counter time) no new batch starts and None is
    returned.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    for subject in order:
        path = fold_dir / f"{subject}.json"
        if path.exists():
            done[subject] = json.loads(path.read_text())
    todo = [subject for subject in order if subject not in done]
    batch = max(1, n_jobs if n_jobs > 0 else os.cpu_count() or 1)
    for i in range(0, len(todo), batch):
        if deadline is not None and time.perf_counter() > deadline:
            return None
        results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_tagged)(subject, *task(subject)) for subject in todo[i:i + batch])
        for subject, result in results:
            (fold_dir / f"{subject}.json").write_text(json.dumps(result))
            done[subject] = result
    return {subject: done[subject] for subject in order}


def _order(subjects) -> list[str]:
    return sorted(np.unique(subjects), key=lambda s: int(str(s).lstrip("S")))


def evaluate(cfg: BaselineConfig, n_jobs: int = -1, max_minutes: float | None = None) -> pd.DataFrame:
    """Run every (window, task, normalisation, feature set, classifier[, calibration]).

    LOSO within ``cfg.dataset``, or with ``test_dataset`` set, trained on all of
    ``cfg.dataset`` and tested on each subject of the other dataset. Each
    combination is saved as soon as it finishes, so an interrupted run
    continues where it stopped. With ``max_minutes``, no new combination is
    started after that time; run again to do the rest.
    """
    cache_dir = cfg.out_dir / "combinations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows, pending, started = [], 0, time.perf_counter()
    deadline = None if max_minutes is None else started + max_minutes * 60
    cross = cfg.test_dataset is not None
    feature_jobs = min(4, n_jobs) if n_jobs > 0 else 4

    for window_sec in cfg.window_sec:
        raw_table = extract_features(cfg, window_sec, feature_jobs)
        raw_test = extract_features(cfg, window_sec, feature_jobs, test=True) if cross else None

        for task in cfg.tasks:
            n_classes = len(TASKS[task]["classes"])
            for norm in cfg.normalisation:
                table, y = _task_table(standardise_per_subject(raw_table, norm, window_sec, cfg.stride_sec), task)
                subjects = table["subject"].to_numpy()
                starts = table["start_sec"].to_numpy()
                if cross:
                    test_table, y_test = _task_table(
                        standardise_per_subject(raw_test, norm, window_sec, cfg.stride_sec), task)
                    reference = guesser_reference(y_test, n_classes)
                else:
                    reference = guesser_reference(y, n_classes)

                for feature_set in cfg.feature_sets:
                    columns = [c for c in table.columns if c.startswith(tuple(FEATURE_SETS[feature_set]))]
                    if not columns or (cross and not all(c in test_table.columns for c in columns)):
                        raise ValueError(f"Feature set {feature_set} is not available in both datasets")
                    X = table[columns].to_numpy(dtype=np.float64)

                    for clf_name in cfg.classifiers:
                        variants = [None] + list(cfg.calibration_minutes) if not cross else [None]
                        for minutes in variants:
                            key = f"w{window_sec}_{task}_{feature_set}_{normalisation_tag(norm)}_{clf_name}"
                            if minutes is not None:
                                key += f"_cal{minutes:g}"
                            if cross:
                                key = f"cross_{cfg.test_dataset}{'_nohv' if cfg.test_exclude_hyperventilation else ''}_{key}"
                            cache = cache_dir / f"{key}.json"
                            if cache.exists():
                                rows.append(json.loads(cache.read_text()))
                                continue
                            if max_minutes is not None and time.perf_counter() - started > max_minutes * 60:
                                pending += 1
                                continue

                            model = make_pipeline(
                                SimpleImputer(strategy="median"), StandardScaler(),
                                make_classifier(clf_name, cfg.seed),
                            )
                            if cross:
                                fitted = clone(model).fit(X, y)
                                X_test = test_table[columns].to_numpy(dtype=np.float64)
                                test_subjects = test_table["subject"].to_numpy()
                                folds = {s: _scores(y_test[test_subjects == s],
                                                    fitted.predict(X_test[test_subjects == s]), n_classes)
                                         for s in _order(test_subjects)}
                                n_windows = int(len(y_test))
                            elif minutes is None:
                                order = _order(subjects)
                                folds = _cached_folds(cache_dir / key, order, n_jobs, lambda s: (
                                    _fold, model, X, y, subjects == s, n_classes), deadline)
                                n_windows = int(len(y))
                            else:
                                order = _order(subjects)
                                folds = _cached_folds(cache_dir / key, order, n_jobs, lambda s: (
                                    _calibration_fold, model, X, y, subjects, starts, s, minutes,
                                    window_sec, cfg.stride_sec, cfg.calibration_weight, n_classes), deadline)
                                n_windows = int(len(y))
                            if folds is None:  # time budget reached part-way; folds so far are saved
                                pending += 1
                                continue

                            row = _summarise(folds, {
                                "dataset": cfg.dataset, "test_dataset": cfg.test_dataset,
                                "window_sec": window_sec, "task": task, "feature_set": feature_set,
                                "normalisation": norm, "subject_normalisation": norm != "none",
                                "classifier": clf_name, "calibration_minutes": minutes,
                                "n_features": len(columns), "n_windows": n_windows, **reference,
                            })
                            cache.write_text(json.dumps(row, indent=2))
                            rows.append(row)
                            gain = (f"  (before calibration {row['accuracy_before_mean']:.3f})"
                                    if "accuracy_before_mean" in row else "")
                            print(f"{key:<64s} acc={row['accuracy_mean']:.3f}  f1={row['f1_macro_mean']:.3f}"
                                  f"   (majority acc={reference['majority_accuracy']:.3f}){gain}")

    summary = pd.DataFrame([{k: v for k, v in r.items() if k not in ("per_subject", "class_share")} for r in rows])
    if "normalisation" in summary:  # results cached before normalisation variants existed
        summary["normalisation"] = summary["normalisation"].fillna(
            summary["subject_normalisation"].map({True: "recording", False: "none"}))
    summary.to_csv(cfg.out_dir / "summary.csv", index=False)
    if pending:
        print(f"\nTime budget reached: {pending} combinations still to run. Run the same command again.")
    return summary


def print_best(summary: pd.DataFrame) -> None:
    """Best classifier per setting, by mean accuracy over test subjects."""
    if summary.empty:
        return
    summary = summary.copy()
    for col, default in (("normalisation", None), ("calibration_minutes", -1), ("test_dataset", "-")):
        if col not in summary:
            summary[col] = default
    summary["normalisation"] = summary["normalisation"].fillna(
        summary["subject_normalisation"].map({True: "recording", False: "none"}))
    summary["calibration_minutes"] = summary["calibration_minutes"].fillna(-1)
    summary["test_dataset"] = summary["test_dataset"].fillna("-")
    keys = ["test_dataset", "window_sec", "task", "normalisation", "calibration_minutes", "feature_set"]
    best = summary.loc[summary.groupby(keys)["accuracy_mean"].idxmax()]
    cols = keys + ["classifier", "accuracy_mean", "accuracy_std", "f1_macro_mean", "majority_accuracy"]
    if "accuracy_before_mean" in best:
        cols.append("accuracy_before_mean")
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(best[cols].to_string(index=False))
