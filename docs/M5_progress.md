# M5 progress

Started 2026-10-04. Goal: test whether the results hold under realistic
deployment conditions and on a second dataset.

## What was built

- **Stress-Predict support** (`Code/pipeline/stresspredict.py`): 34 labelled
  participants (S01 has no labels), Empatica E4 wrist signals, labels from
  the authors' per-second file. Hyperventilation can be counted as stress or
  excluded (`exclude_hyperventilation`).
- **Deployable per-subject normalisation** (`preprocess.normalisation_source`,
  and `normalisation` in the baseline configs):
  - `recording`: statistics from the whole recording (used in M1-M3; a device
    cannot do this in advance);
  - `first_minutes`: the first N minutes of wear;
  - `rest_minutes`: the first N minutes of the first rest/baseline block, as a
    short calibration rest would provide.
- **Per-user calibration**: the first 1, 2 or 4 minutes of each condition of
  the test subject are used for training (classical: added with weight 10;
  deep: the classifier head is fine-tuned), and the rest of the recording is
  the test set. The uncalibrated model is scored on the same windows.
- **Cross-dataset runs**: train on all of WESAD, test on every Stress-Predict
  participant (`run_pipeline.py cross`, and `test_dataset` in baseline
  configs).
- 10 new tests (37 in total).

## Part 1 result: whole-recording normalisation inflates accuracy

Classical baselines, WESAD, 60 s windows; best of RF, AdaBoost and LDA per
cell, mean leave-one-subject-out accuracy. Source:
`Code/runs/m5/baselines_normalisation/summary.csv`.

| Task | Signals | None | Whole recording | First 5 min of wear | First 5 min of rest |
| --- | --- | --- | --- | --- | --- |
| Binary | chest + wrist physio | 0.937 | 0.984 | 0.897 | **0.945** |
| Binary | wrist physio | 0.931 | 0.951 | 0.853 | 0.886 |
| Three-class | chest + wrist physio | 0.798 | 0.935 | 0.817 | **0.865** |
| Three-class | wrist physio | 0.769 | 0.843 | 0.787 | **0.833** |
| Four-class | chest + wrist physio | 0.781 | 0.921 | 0.778 | **0.818** |
| Four-class | wrist physio | 0.652 | 0.772 | 0.694 | **0.740** |

- The whole-recording setting used so far is optimistic by **7-14 points**
  on the multi-class tasks. Headline results must use a deployable variant.
- A 5-minute rest at the start (`rest_minutes:5`) is the best deployable
  choice: it keeps 3-7 points over no normalisation on the multi-class tasks.
- For binary stress detection normalisation matters little, and for wrist
  signals no normalisation (0.931) beats every deployable variant.
- The first 5 minutes of wear is worse: in WESAD that period is sensor
  fitting, not rest.
- The rest of M5 uses `rest_minutes:5`.

## Calibration design note

A pilot (3 folds, one seed, Stress-Predict) fine-tuned every layer of the
deep model on the calibration windows; accuracy usually fell (for example
0.57 to 0.39), a sign of overfitting to one or two windows per class. The
grid therefore fine-tunes only the classifier head, the standard few-shot
choice, at learning rate 1e-3 for 10 epochs. No further tuning was done on
test data.

## Deep grid (run in your terminal)

| Group | Configs | Runs (x5 seeds) | Question |
| --- | --- | --- | --- |
| `e_` | 27 | 135 | Normalisation variants on WESAD, three models, three tasks; calibration on the rest-normalised binary and four-class runs |
| `g_` | 9 | 45 | Wrist only, rest normalisation, with calibration |
| `h_` | 12 | 60 | Stress-Predict LOSO, six models, with and without hyperventilation, with calibration |
| `i_` | 12 | 60 | Train on WESAD wrist, test on Stress-Predict |

```powershell
& ".venv\Scripts\python.exe" -u Code/run_pipeline.py run --config Code/configs/m5/e_*.toml Code/configs/m5/g_*.toml Code/configs/m5/h_*.toml --seeds 42 43 44 45 46 --resume; & ".venv\Scripts\python.exe" -u Code/run_pipeline.py cross --config Code/configs/m5/i_*.toml --seeds 42 43 44 45 46 --resume
```

## Classical runs still to do

`Code/configs/m5/baselines_calibration.toml`,
`baselines_stress_predict_{hv,nohv}.toml` and `baselines_cross_{hv,nohv}.toml`.
