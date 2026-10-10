"""M6: confidence intervals, paired significance tests and the amusement error analysis.

Everything is computed from saved results; nothing is retrained.

* Every result is reduced to one number per test subject (deep models:
  averaged over seeds first), so all tests are paired by subject.
* 95 % confidence intervals: percentile bootstrap over subjects.
* Paired comparisons: Wilcoxon signed-rank test over subjects, Holm
  correction within each family of comparisons, rank-biserial effect size,
  and a bootstrap interval for the mean difference.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .config import CODE_DIR

FOUR_CLASS = ["baseline", "stress", "amusement", "meditation"]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else CODE_DIR / p


# --------------------------------------------------------------------------- per-subject scores

@dataclass
class Scores:
    """One result reduced to per-subject values, plus run-level spread for deep models."""

    label: str
    kind: str
    per_subject: pd.DataFrame          # index = subject, columns = metrics
    seed_means: pd.DataFrame | None    # rows = seeds, columns = metrics (deep only)
    confusion: np.ndarray | None       # pooled over subjects (and seeds)
    class_names: list[str] | None


def _subject_key(subject: str) -> int:
    return int(str(subject).lstrip("S"))


def _fold_metrics(res: dict, calibration: float | None, which: str) -> dict:
    if calibration is None:
        return {"accuracy": res["accuracy"], "f1_macro": res["f1_macro"]}
    entry = res["calibration"][f"{calibration:g}"]
    entry = entry["before"] if which == "before" else entry
    return {"accuracy": entry["accuracy"], "f1_macro": entry["f1_macro"]}


def load_deep(label: str, runs: Path, config: str, seeds: list[int], calibration: float | None = None,
              which: str = "after") -> Scores:
    """Seeded deep runs ``<runs>/<config>_seed<N>/fold_*.json``, averaged over seeds per subject."""
    frames, seed_rows, pooled, names = [], [], None, None
    for seed in seeds:
        run_dir = runs / f"{config}_seed{seed}"
        folds = sorted(run_dir.glob("fold_*.json"))
        if not folds:
            raise FileNotFoundError(f"No fold results in {run_dir}")
        rows = []
        for path in folds:
            res = json.loads(path.read_text())
            rows.append({"subject": res["subject"], **_fold_metrics(res, calibration, which)})
            if calibration is None:
                cm = np.array(res["confusion_matrix"])
                pooled = cm if pooled is None else pooled + cm
                names = res.get("class_names", FOUR_CLASS)
        df = pd.DataFrame(rows).set_index("subject")
        frames.append(df)
        seed_rows.append(df.mean().rename(seed))
    per_subject = pd.concat(frames).groupby(level=0).mean()
    per_subject = per_subject.loc[sorted(per_subject.index, key=_subject_key)]
    return Scores(label, "deep", per_subject, pd.DataFrame(seed_rows), pooled, names)


def load_classical(label: str, file: Path, which: str = "after") -> Scores:
    """A classical combination file (``combinations/<key>.json``) with per-subject results."""
    res = json.loads(Path(file).read_text())
    rows, pooled = [], None
    for subject, fold in res["per_subject"].items():
        entry = fold["before"] if which == "before" else fold
        rows.append({"subject": subject, "accuracy": entry["accuracy"], "f1_macro": entry["f1_macro"]})
        if "confusion_matrix" in entry:
            cm = np.array(entry["confusion_matrix"])
            pooled = cm if pooled is None else pooled + cm
    per_subject = pd.DataFrame(rows).set_index("subject")
    per_subject = per_subject.loc[sorted(per_subject.index, key=_subject_key)]
    n = pooled.shape[0] if pooled is not None else None
    names = FOUR_CLASS[:n] if n in (3, 4) else (["non-stress", "stress"] if n == 2 else None)
    return Scores(label, "classical", per_subject, None, pooled, names)


def load_source(label: str, spec: dict, default_seeds: list[int]) -> Scores:
    kind = spec["kind"]
    if kind == "deep":
        return load_deep(label, _resolve(spec["runs"]), spec["config"], spec.get("seeds", default_seeds),
                         spec.get("calibration"), spec.get("which", "after"))
    if kind == "classical":
        return load_classical(label, _resolve(spec["file"]), spec.get("which", "after"))
    raise ValueError(f"Unknown source kind {kind!r} for {label}")


# --------------------------------------------------------------------------- statistics

def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator, level: float = 0.95):
    """Percentile bootstrap interval for the mean of ``values`` (resampling subjects)."""
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def paired_test(a: pd.Series, b: pd.Series, n_boot: int, rng: np.random.Generator) -> dict:
    """b - a over the subjects both share."""
    common = a.index.intersection(b.index)
    if len(common) < 5:
        raise ValueError(f"Only {len(common)} subjects in common; need at least 5 for a paired test")
    diff = (b.loc[common] - a.loc[common]).to_numpy(dtype=float)
    lo, hi = bootstrap_ci(diff, n_boot, rng)
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        p, r = 1.0, 0.0
    else:
        p = float(stats.wilcoxon(nonzero).pvalue)
        ranks = stats.rankdata(np.abs(nonzero))
        r = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
    return {"n_subjects": int(len(common)), "mean_difference": float(diff.mean()),
            "diff_ci_low": lo, "diff_ci_high": hi, "wilcoxon_p": p, "rank_biserial": r}


def summarise(scores: dict[str, Scores], metrics: list[str], n_boot: int, rng) -> pd.DataFrame:
    rows = []
    for label, sc in scores.items():
        for metric in metrics:
            values = sc.per_subject[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(values, n_boot, rng)
            row = {"label": label, "kind": sc.kind, "metric": metric, "n_subjects": len(values),
                   "mean": float(values.mean()), "ci_low": lo, "ci_high": hi,
                   "subject_std": float(values.std(ddof=1))}
            if sc.seed_means is not None and metric in sc.seed_means:
                row["seed_std"] = float(sc.seed_means[metric].std(ddof=1)) if len(sc.seed_means) > 1 else 0.0
                row["n_seeds"] = int(len(sc.seed_means))
            rows.append(row)
    return pd.DataFrame(rows)


def compare(scores: dict[str, Scores], families: list[dict], n_boot: int, rng) -> pd.DataFrame:
    rows = []
    for fam in families:
        labels, metric = fam["labels"], fam.get("metric", "accuracy")
        pairs = ([(labels[0], other) for other in labels[1:]] if fam.get("mode", "vs_first") == "vs_first"
                 else [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]])
        fam_rows = []
        for a, b in pairs:
            res = paired_test(scores[a].per_subject[metric], scores[b].per_subject[metric], n_boot, rng)
            fam_rows.append({"family": fam["name"], "metric": metric, "a": a, "b": b, **res})
        for row, adj in zip(fam_rows, holm([r["wilcoxon_p"] for r in fam_rows])):
            row["holm_p"] = adj
            row["significant_0.05"] = adj < 0.05
        rows.extend(fam_rows)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- amusement analysis

QUEST_CONDITIONS = {"Base": "baseline", "TSST": "stress", "Fun": "amusement", "Medi 1": "meditation 1",
                    "Medi 2": "meditation 2"}


def read_self_reports(quest_csv: Path) -> pd.DataFrame:
    """SAM valence/arousal (the ``# DIM`` rows) per condition from a WESAD ``S*_quest.csv``.

    The five questionnaire blocks follow the conditions in the order given by
    the ``# ORDER`` row, skipping the short reading periods (sRead, fRead).
    """
    order, dims = None, []
    for line in Path(quest_csv).read_text(encoding="latin-1").splitlines():
        cells = [c.strip() for c in line.split(";")]
        if cells[0] == "# ORDER":
            order = [c for c in cells[1:] if c and c in QUEST_CONDITIONS]
        elif cells[0] == "# DIM":
            dims.append([float(c) for c in cells[1:3]])
    if order is None or len(dims) != len(order):
        raise ValueError(f"Unexpected questionnaire layout in {quest_csv}")
    return pd.DataFrame(dims, columns=["valence", "arousal"], index=[QUEST_CONDITIONS[c] for c in order])


def amusement_analysis(scores: dict[str, Scores], labels: list[str], raw_dir: Path, out_dir: Path) -> dict:
    """Where amusement windows go, per model, and whether subjects whose amusement
    was recognised better also reported a stronger amusement effect."""
    out = {"confusion_of_amusement": {}, "self_report_correlation": {}}

    reports = {}
    for quest in sorted(Path(raw_dir).glob("S*/S*_quest.csv")):
        sam = read_self_reports(quest)
        reports[quest.parent.name] = {
            "valence_change": sam.loc["amusement", "valence"] - sam.loc["baseline", "valence"],
            "arousal_change": sam.loc["amusement", "arousal"] - sam.loc["baseline", "arousal"],
        }
    reports = pd.DataFrame(reports).T
    reports.to_csv(out_dir / "amusement_self_reports.csv")
    out["self_report_summary"] = {
        "subjects": int(len(reports)),
        "valence_change_mean": float(reports["valence_change"].mean()),
        "arousal_change_mean": float(reports["arousal_change"].mean()),
        "subjects_reporting_no_valence_gain": int((reports["valence_change"] <= 0).sum()),
    }

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    for ax, label in zip(axes[0], labels):
        sc = scores[label]
        cm = sc.confusion
        if cm is None or cm.shape[0] != 4:
            raise ValueError(f"{label}: the amusement analysis needs a four-class confusion matrix")
        norm = cm / cm.sum(axis=1, keepdims=True)
        out["confusion_of_amusement"][label] = dict(zip(FOUR_CLASS, norm[2].round(4).tolist()))
        sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    xticklabels=FOUR_CLASS, yticklabels=FOUR_CLASS, ax=ax)
        ax.set(title=label, xlabel="Predicted", ylabel="True")

        recall = _per_subject_amusement_recall(sc)
        if recall is not None:
            common = recall.index.intersection(reports.index)
            corr = {}
            for col in ("valence_change", "arousal_change"):
                rho, p = stats.spearmanr(recall.loc[common], reports.loc[common, col])
                corr[col] = {"spearman_rho": float(rho), "p": float(p), "n": int(len(common))}
            out["self_report_correlation"][label] = corr
    fig.tight_layout()
    fig.savefig(out_dir / "amusement_confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    (out_dir / "amusement.json").write_text(json.dumps(out, indent=2))
    return out


def _per_subject_amusement_recall(sc: Scores) -> pd.Series | None:
    return sc.per_subject["amusement_recall"] if "amusement_recall" in sc.per_subject else None


def add_amusement_recall(sc: Scores, spec: dict, default_seeds: list[int]) -> None:
    """Per-subject amusement recall, read from the same fold files as the scores."""
    recalls = {}
    if sc.kind == "deep":
        runs = _resolve(spec["runs"])
        for seed in spec.get("seeds", default_seeds):
            for path in (runs / f"{spec['config']}_seed{seed}").glob("fold_*.json"):
                res = json.loads(path.read_text())
                cm = np.array(res["confusion_matrix"])
                recalls.setdefault(res["subject"], []).append(cm[2, 2] / max(cm[2].sum(), 1))
    else:
        res = json.loads(_resolve(spec["file"]).read_text())
        for subject, fold in res["per_subject"].items():
            cm = np.array(fold["confusion_matrix"])
            recalls[subject] = [cm[2, 2] / max(cm[2].sum(), 1)]
    sc.per_subject["amusement_recall"] = pd.Series({s: float(np.mean(v)) for s, v in recalls.items()})


# --------------------------------------------------------------------------- driver

def run(config_path: str | Path) -> None:
    with open(config_path, "rb") as fh:
        cfg = tomllib.load(fh)
    out_dir = _resolve(cfg["run"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["run"].get("seed", 42))
    n_boot = int(cfg["run"].get("bootstrap", 10000))
    seeds = cfg["run"].get("seeds", [42, 43, 44, 45, 46])

    sources = cfg["sources"]
    scores = {label: load_source(label, spec, seeds) for label, spec in sources.items()}
    print(f"Loaded {len(scores)} results")

    summary = summarise(scores, ["accuracy", "f1_macro"], n_boot, rng)
    summary.to_csv(out_dir / "summary.csv", index=False)

    comparisons = compare(scores, cfg.get("families", []), n_boot, rng)
    comparisons.to_csv(out_dir / "comparisons.csv", index=False)

    amusement = None
    if "amusement" in cfg:
        labels = cfg["amusement"]["labels"]
        for label in labels:
            add_amusement_recall(scores[label], sources[label], seeds)
        amusement = amusement_analysis(scores, labels, _resolve(cfg["amusement"]["raw_dir"]), out_dir)

    write_report(summary, comparisons, amusement, out_dir / "report.md")
    print(f"M6 results written to {out_dir}")


def write_report(summary: pd.DataFrame, comparisons: pd.DataFrame, amusement: dict | None, path: Path) -> None:
    lines = ["# M6 statistics", "", "Generated by `run_pipeline.py stats`. Means over test subjects, 95 % "
             "bootstrap confidence intervals over subjects; deep models averaged over seeds first.", "",
             "## Results with confidence intervals", "",
             "| Result | Metric | Mean | 95 % CI | Seed std |", "| --- | --- | --- | --- | --- |"]
    for _, r in summary.iterrows():
        seed = f"{r['seed_std']:.3f}" if "seed_std" in r and pd.notna(r.get("seed_std")) else "-"
        lines.append(f"| {r['label']} | {r['metric']} | {r['mean']:.3f} | {r['ci_low']:.3f}-{r['ci_high']:.3f} | {seed} |")
    lines += ["", "## Paired comparisons (b - a)", "",
              "| Family | a | b | Metric | Mean diff | 95 % CI | Wilcoxon p | Holm p | Effect (r) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in comparisons.iterrows():
        mark = " *" if r["significant_0.05"] else ""
        lines.append(f"| {r['family']} | {r['a']} | {r['b']} | {r['metric']} | {r['mean_difference']:+.3f} | "
                     f"{r['diff_ci_low']:+.3f} to {r['diff_ci_high']:+.3f} | {r['wilcoxon_p']:.3g} | "
                     f"{r['holm_p']:.3g}{mark} | {r['rank_biserial']:+.2f} |")
    lines += ["", "\\* Holm-adjusted p < 0.05 within its family."]
    if amusement:
        s = amusement["self_report_summary"]
        lines += ["", "## Amusement", "",
                  f"Self-reports (SAM, amusement minus baseline, {s['subjects']} subjects): mean valence change "
                  f"{s['valence_change_mean']:+.2f}, mean arousal change {s['arousal_change_mean']:+.2f}; "
                  f"{s['subjects_reporting_no_valence_gain']} subjects reported no valence gain.", "",
                  "| Model | Amusement predicted as baseline | as stress | correctly | as meditation |",
                  "| --- | --- | --- | --- | --- |"]
        for label, row in amusement["confusion_of_amusement"].items():
            lines.append(f"| {label} | {row['baseline']:.2f} | {row['stress']:.2f} | {row['amusement']:.2f} | "
                         f"{row['meditation']:.2f} |")
        lines += ["", "Spearman correlation between per-subject amusement recall and the self-reported change:", "",
                  "| Model | Valence rho (p) | Arousal rho (p) |", "| --- | --- | --- |"]
        for label, corr in amusement["self_report_correlation"].items():
            v, a = corr["valence_change"], corr["arousal_change"]
            lines.append(f"| {label} | {v['spearman_rho']:+.2f} ({v['p']:.2g}) | {a['spearman_rho']:+.2f} ({a['p']:.2g}) |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def classical_key_from_label(path: str) -> str:
    return re.sub(r"\.json$", "", Path(path).name)
