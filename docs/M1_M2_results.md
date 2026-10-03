# M1 and M2 results

Recorded 2026-10-03. All figures are leave-one-subject-out (LOSO) over the 15
WESAD subjects, mean ± sample standard deviation across subjects, single seed.
Source files are under `Code/runs/`.

## M1: corrected data pipeline

Exit gate: class shares near 39 / 22 / 12 / 26 %. **Met.**

| | Legacy pipeline | Clean pipeline |
| --- | --- | --- |
| Windows (10 s, 5 s stride) | 18,076 | 8,841 |
| Baseline / stress / amusement / meditation | 68.6 / 11.5 / 6.4 / 13.5 % | 39.5 / 22.2 / 12.3 / 26.0 % |
| Undefined and ignored data (labels 0, 5-7) | folded into "baseline" | dropped |
| Resampling | block average, effective 33.3 Hz | anti-aliased, true time axis |
| Channels | 8 | 14 (adds wrist BVP, chest EMG, accelerometer axes) |
| Normalisation | global z-score from training subjects | per-subject z-score, label-free, then global |

Raw label shares measured on the dataset: label 0 is 45.5 % of all samples,
labels 5-7 are 2.7 %, real baseline is 20.3 %.

### Same model, corrected data

`clean_w10_weighted_sampler` keeps the model and training recipe of
`legacy_weighted_sampler` and changes only the data pipeline.

| Metric | Legacy data (paper draft) | Clean data |
| --- | --- | --- |
| Accuracy | 0.644 ± 0.082 | 0.782 ± 0.102 |
| Always-majority accuracy | 0.683 | 0.395 |
| Macro-F1 | 0.296 ± 0.095 | 0.675 ± 0.131 |
| Balanced accuracy | 0.337 | 0.692 |
| Cohen's kappa | 0.154 | 0.686 |
| Folds below always-majority | 7 of 15 | 0 of 15 |
| F1 baseline / stress / amusement / meditation | 0.78 / 0.27 / 0.00 / 0.13 | 0.86 / 0.86 / 0.20 / 0.78 |

Caveats: the two rows are not the same task (the legacy "baseline" class is
mostly unlabelled data), and the clean run changes labelling, channels and
normalisation together, so the gain cannot be attributed to one of them.
Amusement is still poorly recognised (48 % of it is predicted as baseline).

## M2: reference baselines

Exit gate: reproduce the dataset paper's LOSO results (about 80 % three-class,
93 % binary) within a few points. **Met.**

Hand-crafted features, 60 s windows, 5 s stride, no subject normalisation,
chest physiological signals (ECG, EDA, EMG, respiration, temperature):

| Task | Classifier | This repo | Dataset paper | Always-majority |
| --- | --- | --- | --- | --- |
| Three-class | AdaBoost | 80.8 ± 11.2 % | 80.34 % | 54.9 % |
| Binary (stress vs non-stress) | LDA | 93.5 ± 7.1 % | 93.12 % | 70.2 % |

Dataset-paper figures are from Schmidt et al., ICMI 2018.

### Reference points for the deep model (M3)

Best classifier per setting by mean LOSO accuracy; macro-F1 in brackets.

| Task | Signals | No subject normalisation | Per-subject normalisation |
| --- | --- | --- | --- |
| Binary | chest physio | 93.5 % (0.91) LDA | 97.0 % (0.97) RF |
| Binary | wrist physio | 93.1 % (0.91) LDA | 95.1 % (0.93) RF |
| Three-class | chest physio | 80.8 % (0.67) AdaBoost | 92.3 % (0.87) RF |
| Three-class | wrist physio | 76.9 % (0.62) RF | 84.3 % (0.75) AdaBoost |
| Four-class | chest physio | 80.9 % (0.70) AdaBoost | 90.6 % (0.84) AdaBoost |
| Four-class | wrist physio | 65.2 % (0.52) LDA | 77.2 % (0.66) AdaBoost |
| Four-class | chest + wrist physio | 78.1 % (0.68) AdaBoost | 92.1 % (0.86) AdaBoost |

Trivial guessers: always-majority accuracy is 70.2 % (binary), 54.9 %
(three-class) and 41.3 % (four-class); uniform random is 50 / 33.3 / 25 %.

Findings:

- **Per-subject normalisation is the largest single lever**: +10 to +12 points
  on the three- and four-class tasks. It standardises each feature over the
  subject's whole recording without labels, so it is deployable only if a
  recording from the new user is available first. State this in the paper.
- **The current deep model is below the classical baseline.** On four-class it
  reaches 78 % against 90.6 % for AdaBoost on hand-crafted chest features.
  M3 has to close that gap before any claim about the architecture.
- **Wrist-only is clearly harder** than chest on the multi-class tasks.
- **Amusement is the hard class** for every method (recall 0.62 for the best
  four-class random forest).

### Window-length ablation

Random forest, per-subject normalisation, mean LOSO accuracy:

| Task | Signals | 10 s | 30 s | 60 s |
| --- | --- | --- | --- | --- |
| Binary | chest physio | 94.1 % | 96.1 % | 97.0 % |
| Binary | wrist physio | 90.9 % | 94.0 % | 95.1 % |
| Three-class | chest physio | 87.7 % | 91.3 % | 92.3 % |
| Three-class | wrist physio | 77.5 % | 81.9 % | 83.9 % |
| Four-class | chest physio | 84.5 % | 87.8 % | 89.6 % |
| Four-class | wrist physio | 69.6 % | 73.1 % | 74.3 % |

Longer windows help in every setting; most of the gain comes between 10 and
30 s. The deep model in M3 should be tested at 30 and 60 s, not only 10 s.

## Limits of these numbers

- One seed. Seed variation is not measured yet (M6).
- "Best classifier per setting" is chosen on the test folds, as in the dataset
  paper, so those cells are optimistic. Compare like with like.
- Features deviate from the dataset paper in documented ways (see the top of
  `Code/pipeline/features.py`); matching its accuracy does not mean the
  feature sets are identical.
- The legacy reproduction run (retraining the paper's two configurations) is
  still at 4 of 45 folds.
