"""Aggregate per-fold results into the tables and figures used in the paper.

Reads both this pipeline's ``fold_<subject>.json`` files and the older
notebook outputs (``results_<subject>.json`` / ``eval_<subject>.json``), so the
numbers in the current draft can be regenerated from the committed results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .config import CLASS_NAMES

FOLD_PATTERNS = ("fold_*.json", "results_*.json", "eval_*.json")


def load_folds(results_dir: Path) -> dict[str, np.ndarray]:
    """Map subject -> confusion matrix for every fold file in ``results_dir``."""
    folds = {}
    for pattern in FOLD_PATTERNS:
        for path in sorted(Path(results_dir).glob(pattern)):
            res = json.loads(path.read_text())
            folds[res["subject"]] = np.array(res["confusion_matrix"], dtype=np.int64)
    if not folds:
        raise FileNotFoundError(f"No fold result files {FOLD_PATTERNS} in {results_dir}")
    # S2..S17 in numeric order
    return dict(sorted(folds.items(), key=lambda kv: int(kv[0].lstrip("S"))))


def metrics_from_cm(cm: np.ndarray) -> dict:
    n = cm.sum()
    support, predicted, correct = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    recall = np.divide(correct, support, out=np.zeros(len(cm)), where=support > 0)
    precision = np.divide(correct, predicted, out=np.zeros(len(cm)), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(len(cm)), where=denom > 0)

    accuracy = correct.sum() / n
    expected = (support * predicted).sum() / n ** 2
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else 0.0

    # Reference: a classifier that always predicts this subject's largest class.
    majority_share = support.max() / n
    majority_f1 = (2 * majority_share / (1 + majority_share)) / len(cm)

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "kappa": float(kappa),
        "majority_accuracy": float(majority_share),
        "majority_f1_macro": float(majority_f1),
        **{f"f1_{name}": float(v) for name, v in zip(CLASS_NAMES, f1)},
    }


def per_subject_table(folds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = [{"subject": subj, **metrics_from_cm(cm)} for subj, cm in folds.items()]
    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame, folds: dict[str, np.ndarray]) -> dict:
    metric_cols = [c for c in table.columns if c != "subject"]
    pooled = sum(folds.values())
    return {
        "n_folds": int(len(table)),
        "n_windows": int(pooled.sum()),
        "mean": {c: float(table[c].mean()) for c in metric_cols},
        # Sample standard deviation across folds (ddof=1).
        "std": {c: float(table[c].std(ddof=1)) if len(table) > 1 else 0.0 for c in metric_cols},
        "folds_below_majority_accuracy": int((table["accuracy"] < table["majority_accuracy"]).sum()),
        "pooled_confusion_matrix": pooled.tolist(),
        "class_share": dict(zip(CLASS_NAMES, (pooled.sum(axis=1) / pooled.sum()).round(4).tolist())),
    }


def plot_confusion_matrix(pooled: np.ndarray, title: str, path: Path) -> None:
    row_sums = pooled.sum(axis=1, keepdims=True)
    cm_norm = np.divide(pooled, row_sums, out=np.zeros(pooled.shape), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def report_run(results_dir: Path, out_dir: Path, label: str) -> tuple[pd.DataFrame, dict]:
    """Write per-subject CSV, summary JSON and confusion matrix for one run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = load_folds(results_dir)
    table = per_subject_table(folds)
    summary = {"label": label, "source": str(results_dir), **summarise(table, folds)}

    table.to_csv(out_dir / f"{label}_per_subject.csv", index=False)
    (out_dir / f"{label}_summary.json").write_text(json.dumps(summary, indent=2))
    plot_confusion_matrix(
        np.array(summary["pooled_confusion_matrix"]),
        f"{label} - confusion matrix (normalised)",
        out_dir / f"{label}_confusion_matrix.png",
    )

    m, s = summary["mean"], summary["std"]
    print(f"\n[{label}] {summary['n_folds']} folds, {summary['n_windows']} windows")
    print(f"  Accuracy          : {m['accuracy']:.4f} +/- {s['accuracy']:.4f}"
          f"   (always-majority: {m['majority_accuracy']:.4f})")
    print(f"  Macro-F1          : {m['f1_macro']:.4f} +/- {s['f1_macro']:.4f}"
          f"   (always-majority: {m['majority_f1_macro']:.4f})")
    print(f"  Balanced accuracy : {m['balanced_accuracy']:.4f}   Cohen's kappa: {m['kappa']:.4f}")
    print(f"  Folds below always-majority accuracy: {summary['folds_below_majority_accuracy']}/{summary['n_folds']}")
    print("  Per-class F1      : " + ", ".join(f"{c}={m[f'f1_{c}']:.3f}" for c in CLASS_NAMES))
    return table, summary


def _paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    diff = b - a
    out = {"mean_difference": float(diff.mean()), "t_test_p": None, "wilcoxon_p": None}
    if len(diff) > 1 and np.any(diff != 0):
        out["t_test_p"] = float(stats.ttest_rel(b, a).pvalue)
        out["wilcoxon_p"] = float(stats.wilcoxon(b, a).pvalue)
    return out


def compare_runs(tables: dict[str, pd.DataFrame], out_dir: Path) -> dict:
    """Paired comparison of the first run against each other run, per subject."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(tables)
    merged = None
    for label, table in tables.items():
        renamed = table.rename(columns={c: f"{label}:{c}" for c in table.columns if c != "subject"})
        merged = renamed if merged is None else merged.merge(renamed, on="subject", how="inner")
    merged.to_csv(out_dir / "comparison_per_subject.csv", index=False)

    ref = labels[0]
    comparison = {}
    for other in labels[1:]:
        comparison[f"{other} vs {ref}"] = {
            metric: _paired_test(merged[f"{ref}:{metric}"].to_numpy(), merged[f"{other}:{metric}"].to_numpy())
            for metric in ("accuracy", "f1_macro", "balanced_accuracy")
        }
    (out_dir / "comparison_tests.json").write_text(json.dumps(comparison, indent=2))

    # Macro-F1 per subject
    long_f1 = pd.concat(
        [t[["subject", "f1_macro"]].assign(setting=label) for label, t in tables.items()], ignore_index=True
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=long_f1, x="subject", y="f1_macro", hue="setting", ax=ax)
    ax.set(ylim=(0, 1), title="Macro-F1 per test subject (LOSO)")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_f1_macro.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Mean per-class F1
    long_cls = pd.DataFrame([
        {"class": c, "setting": label, "f1": t[f"f1_{c}"].mean()}
        for label, t in tables.items() for c in CLASS_NAMES
    ])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=long_cls, x="class", y="f1", hue="setting", ax=ax)
    ax.set(ylim=(0, 1), ylabel="F1 score", title="Per-class F1 (LOSO mean)")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_per_class_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for name, tests in comparison.items():
        print(f"\n[{name}] paired over {len(merged)} subjects")
        for metric, t in tests.items():
            p = "n/a" if t["wilcoxon_p"] is None else f"t-test p={t['t_test_p']:.3f}, Wilcoxon p={t['wilcoxon_p']:.3f}"
            print(f"  {metric:<18s}: mean difference {t['mean_difference']:+.4f}  ({p})")
    return comparison


def report(results: dict[str, Path], out_dir: Path) -> None:
    """``results`` maps a label to the directory holding that run's fold files."""
    tables = {}
    for label, results_dir in results.items():
        tables[label], _ = report_run(Path(results_dir), out_dir, label)
    if len(tables) > 1:
        compare_runs(tables, out_dir)
    print(f"\nReport written to {out_dir}")
