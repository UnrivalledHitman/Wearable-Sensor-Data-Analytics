"""Command-line entry point for the WESAD pipeline.

Train and report (one or more configs; several configs are also compared):
    python Code/run_pipeline.py run --config Code/configs/legacy_weighted_loss.toml Code/configs/legacy_weighted_sampler.toml

Rebuild tables and figures from existing fold results (no dataset needed):
    python Code/run_pipeline.py report --results weighted_loss=Code/results_training_hybrid weighted_sampler=Code/results_evaluation --out Code/runs/paper_committed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.config import CODE_DIR, load_config

STAGES = ("preprocess", "train", "report")


def cmd_run(args) -> None:
    import torch

    from pipeline.preprocess import preprocess_all
    from pipeline.report import report
    from pipeline.train import run_loso

    device = torch.device(args.device) if args.device else None
    configs = [load_config(path) for path in args.config]
    names = [cfg.name for cfg in configs]
    if len(set(names)) != len(names):
        raise SystemExit(f"Config run names must be unique, got {names}")

    for cfg in configs:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        if "preprocess" in args.stages:
            preprocess_all(cfg, force=args.force_preprocess)
        if "train" in args.stages:
            run_loso(cfg, device=device)
        if "report" in args.stages:
            report({cfg.name: cfg.out_dir}, cfg.out_dir / "report")

    if "report" in args.stages and len(configs) > 1:
        report({cfg.name: cfg.out_dir for cfg in configs}, CODE_DIR / "runs" / ("compare_" + "_vs_".join(names)))


def cmd_report(args) -> None:
    from pipeline.report import report

    results = {}
    for item in args.results:
        label, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"--results entries must look like LABEL=DIR, got {item!r}")
        results[label] = Path(path)
    report(results, Path(args.out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="preprocess, train (LOSO) and report for each config")
    run.add_argument("--config", nargs="+", required=True, help="one or more TOML config files")
    run.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    run.add_argument("--device", default=None, help="e.g. cuda or cpu (default: cuda if available)")
    run.add_argument("--force-preprocess", action="store_true", help="redo preprocessing even if up to date")
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="tables and figures from existing fold result files")
    rep.add_argument("--results", nargs="+", required=True, metavar="LABEL=DIR")
    rep.add_argument("--out", required=True)
    rep.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
