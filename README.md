# Wearable Sensor Data Analytics
Edge Deloyable deep learning model to detect patient mental state on the go rather than relying on cloud deployed models for cost efficacy.

## Layout

| Path | Contents |
| --- | --- |
| `Code/pipeline/` | Preprocessing, model, leave-one-subject-out training, reporting |
| `Code/configs/` | One TOML file per experiment; every setting that affects a result lives here |
| `Code/run_pipeline.py` | Command-line entry point |
| `Code/tests/` | Smoke test on synthetic data, plus a check that committed results match the paper |
| `Code/runs/` | Output of each run (fold results, report, checkpoints) |
| `Code/results_training_hybrid/`, `Code/results_evaluation/` | Fold results behind the current paper draft (December 2025) |
| `Code/*.ipynb` | Original notebooks, kept for reference; no longer the source of truth |
| `references/` | Verified bibliography and the record of how each entry was checked |

## Setup

Python 3.13. Install the project dependencies, then PyTorch separately so you
get the build that matches your CUDA version (see pytorch.org for the command):

```bash
uv sync
```

`Code/requirements-frozen.txt` lists the exact versions the experiments were
developed with (PyTorch 2.13.0 with CUDA 13.2).

The WESAD dataset is not in this repository. Download it and place the subject
pickles in `Code/WESAD/` (either `S2.pkl` or `S2/S2.pkl` layouts work).

## Running

Reproduce both experiments from the current draft and compare them:

```bash
python Code/run_pipeline.py run --config Code/configs/legacy_weighted_loss.toml Code/configs/legacy_weighted_sampler.toml
```

Rebuild the paper's tables and figures from the committed fold results, with no
dataset or GPU needed:

```bash
python Code/run_pipeline.py report --results weighted_loss=Code/results_training_hybrid weighted_sampler=Code/results_evaluation --out Code/runs/paper_committed
```

Run the tests:

```bash
python -m unittest discover -s Code/tests -t Code
```

## Experiments

| Config | What it is |
| --- | --- |
| `legacy_weighted_loss.toml` | The run the draft calls "baseline": class-weighted cross-entropy |
| `legacy_weighted_sampler.toml` | The run the draft calls "balanced": weighted random sampling |
| `legacy_no_balancing.toml` | The unbalanced control the draft describes but never ran |

The `legacy_*` configs reproduce the December 2025 experiments exactly as they
were run, including their known defects (listed at the top of
`Code/pipeline/preprocess.py`). They exist so the published numbers can be
regenerated, not as a recommended setup.

A retrained run will not match the committed results digit for digit: the
original notebooks seeded once for all folds, while the pipeline seeds each
fold separately, and GPU training is not bit-reproducible. Expect agreement
within seed-to-seed variation (about 0.02 macro-F1).
