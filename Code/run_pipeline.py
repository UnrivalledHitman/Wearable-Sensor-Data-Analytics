"""Command-line entry point for the WESAD pipeline.

Train and report (one or more configs; several configs are also compared):
    python Code/run_pipeline.py run --config Code/configs/legacy_weighted_loss.toml Code/configs/legacy_weighted_sampler.toml

Classical reference baselines (resumable):
    python Code/run_pipeline.py baselines --config Code/configs/baselines.toml

Rebuild tables and figures from existing fold results (no dataset needed):
    python Code/run_pipeline.py report --results weighted_loss=Code/results_training_hybrid weighted_sampler=Code/results_evaluation --out Code/runs/paper_committed
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import time
from pathlib import Path

from pipeline.config import CODE_DIR, load_config

STAGES = ("preprocess", "train", "report")


def expand_globs(patterns: list[str]) -> list[str]:
    """Expand wildcards here, since PowerShell passes them to programs unexpanded."""
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not matches:
            raise SystemExit(f"No config files match {pattern!r}")
        paths.extend(matches)
    return paths


def cmd_run(args) -> None:
    import torch

    from pipeline.preprocess import preprocess_all
    from pipeline.report import report
    from pipeline.train import run_loso

    device = torch.device(args.device) if args.device else None
    configs = [load_config(path) for path in expand_globs(args.config)]
    if args.seeds:
        configs = [cfg.with_seed(seed) for cfg in configs for seed in args.seeds]
    names = [cfg.name for cfg in configs]
    if len(set(names)) != len(names):
        raise SystemExit(f"Config run names must be unique, got {names}")

    # One time budget shared by every config in this invocation.
    deadline = None if args.max_minutes is None else time.perf_counter() + args.max_minutes * 60
    for cfg in configs:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        if "preprocess" in args.stages:
            preprocess_all(cfg, force=args.force_preprocess)
        if "train" in args.stages:
            remaining = None if deadline is None else max(0.0, (deadline - time.perf_counter()) / 60)
            results = run_loso(cfg, device=device, resume=args.resume, max_minutes=remaining)
            if len(results) < len(list(cfg.data_dir.glob("*_combined.npz"))):
                return  # stopped by the time budget; nothing complete to report yet
        if "report" in args.stages:
            report({cfg.name: cfg.out_dir}, cfg.out_dir / "report")

    if "report" in args.stages and len(configs) > 1:
        label = "_vs_".join(names)
        if len(configs) > 3:  # keep Windows paths short
            label = f"{len(configs)}_runs_" + hashlib.sha1(label.encode()).hexdigest()[:8]
        report({cfg.name: cfg.out_dir for cfg in configs}, CODE_DIR / "runs" / f"compare_{label}")


def cmd_report(args) -> None:
    from pipeline.report import report

    results = {}
    for item in args.results:
        label, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"--results entries must look like LABEL=DIR, got {item!r}")
        results[label] = Path(path)
    report(results, Path(args.out))


def cmd_baselines(args) -> None:
    from pipeline.baselines import evaluate, load_baseline_config, print_best

    for path in expand_globs(args.config):
        cfg = load_baseline_config(path)
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        summary = evaluate(cfg, n_jobs=args.jobs, max_minutes=args.max_minutes)
        print(f"\n[{cfg.name}] best classifier per setting (LOSO mean accuracy):")
        print_best(summary)
        print(f"\nSummary written to {cfg.out_dir / 'summary.csv'}")


def cmd_seeds(args) -> None:
    from pipeline.seeds import summarise_seeds

    summarise_seeds(Path(args.runs), Path(args.out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="preprocess, train (LOSO) and report for each config")
    run.add_argument("--config", nargs="+", required=True, help="one or more TOML config files")
    run.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    run.add_argument("--device", default=None, help="e.g. cuda or cpu (default: cuda if available)")
    run.add_argument("--force-preprocess", action="store_true", help="redo preprocessing even if up to date")
    run.add_argument("--resume", action="store_true", help="keep folds already finished with the same config")
    run.add_argument("--max-minutes", type=float, default=None,
                     help="start no new fold after this long; rerun with --resume to continue")
    run.add_argument("--seeds", type=int, nargs="+", default=None,
                     help="run each config once per seed, in <out_dir>_seed<N> folders")
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="tables and figures from existing fold result files")
    rep.add_argument("--results", nargs="+", required=True, metavar="LABEL=DIR")
    rep.add_argument("--out", required=True)
    rep.set_defaults(func=cmd_report)

    base = sub.add_parser("baselines", help="hand-crafted features + classical classifiers under LOSO")
    base.add_argument("--config", nargs="+", required=True, help="one or more baseline TOML config files")
    base.add_argument("--jobs", type=int, default=-1, help="parallel workers (default: all cores)")
    base.add_argument("--max-minutes", type=float, default=None,
                      help="start no new combination after this long; rerun to continue")
    base.set_defaults(func=cmd_baselines)

    seeds = sub.add_parser("seeds", help="summarise runs repeated over seeds (<name>_seed<N> folders)")
    seeds.add_argument("--runs", required=True, help="folder holding the seeded run folders")
    seeds.add_argument("--out", required=True, help="CSV file to write")
    seeds.set_defaults(func=cmd_seeds)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
