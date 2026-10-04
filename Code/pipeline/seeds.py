"""Summarise runs repeated over seeds (``<name>_seed<N>`` folders)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .report import fold_class_names, fold_costs, load_folds, per_subject_table

SEED_SUFFIX = re.compile(r"^(?P<name>.+)_seed(?P<seed>\d+)$")


def run_metrics(run_dir: Path) -> dict:
    """Mean over the LOSO folds of one run, per metric, plus its training cost."""
    names = fold_class_names(run_dir)
    table = per_subject_table(load_folds(run_dir), names)
    metrics = table.drop(columns="subject").mean().to_dict()
    return {**metrics, **fold_costs(run_dir), "n_folds": len(table), "class_names": names}


def summarise_seeds(runs_dir: Path, out_path: Path, n_folds: int = 15) -> pd.DataFrame:
    """One row per config: mean, std, min and max over seeds of each run-level metric."""
    rows = []
    for run_dir in sorted(Path(runs_dir).iterdir()):
        match = SEED_SUFFIX.match(run_dir.name)
        if not (run_dir.is_dir() and match):
            continue
        m = run_metrics(run_dir)
        if m["n_folds"] < n_folds:
            print(f"Skipping unfinished run {run_dir.name} ({m['n_folds']}/{n_folds} folds)")
            continue
        rows.append({"config": match["name"], "seed": int(match["seed"]), **m})

    runs = pd.DataFrame(rows)
    metric_cols = [c for c in runs.columns
                   if c in ("accuracy", "f1_macro", "balanced_accuracy", "kappa") or c.startswith("f1_")]
    metric_cols = list(dict.fromkeys(metric_cols))
    summary = []
    for config, group in runs.groupby("config"):
        row = {"config": config, "n_seeds": len(group)}
        for col in metric_cols:
            values = group[col].dropna()
            if values.empty:
                continue
            row[f"{col}_mean"] = values.mean()
            row[f"{col}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f"{col}_min"], row[f"{col}_max"] = values.min(), values.max()
        for col in ("majority_accuracy", "n_parameters", "seconds_per_epoch", "epochs_run", "peak_gpu_mb"):
            if col in group:
                row[col] = group[col].mean()
        summary.append(row)

    out = pd.DataFrame(summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    runs.drop(columns="class_names").to_csv(out_path.with_name(out_path.stem + "_per_seed.csv"), index=False)
    print(f"{len(runs)} runs over {len(out)} configs -> {out_path}")
    return out
