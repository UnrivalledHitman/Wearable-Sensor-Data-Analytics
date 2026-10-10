# M5 results

Finished 2026-10-09. Deep models: 300 runs (60 configs × 5 seeds), each a
leave-one-subject-out (LOSO) evaluation, or train-on-WESAD / test-on-each
Stress-Predict participant for the cross-dataset runs. Classical models: 510
combinations. Numbers are means over test subjects; for deep models, then
mean ± std over 5 seeds. "Deployable" normalisation = per-subject statistics
from the first 5 minutes of the first rest block (`rest_minutes:5`).

Sources: `Code/runs/m5_summary/seed_summary.csv` (deep), per-fold
`calibration` entries in `Code/runs/m5/*/fold_*.json`, and
`Code/runs/m5/baselines_*/summary.csv` (classical).

## Summary

1. **Whole-recording normalisation, used in M1-M3, inflates results, and far
   more for deep models than for classical ones.** Under deployable
   normalisation the deep four-class accuracy falls from 0.73-0.76 to 0.63,
   and stress F1 from 0.84 to about 0.47.
2. **Under deployable conditions, hand-crafted features beat every deep
   model by a wide margin:** 0.818 against 0.635 on four-class, 0.945 against
   0.818 on binary.
3. **Per-user calibration helps the classical models a lot on WESAD** (four
   minutes per condition lifts four-class from 0.866 to 0.950 on the same
   test windows) and the deep models moderately on four-class, but it **hurts
   on Stress-Predict**.
4. **Stress-Predict is hard:** the best deployable classical model reaches
   0.762 against a 0.703 majority guess; deep models stay below the majority
   guess in accuracy (0.61-0.64) though above it in macro-F1.
5. **Models trained on WESAD do not transfer to Stress-Predict:** accuracy
   at or below the majority guess for both classical and deep models.

## 1. Normalisation (WESAD, 60 s windows)

Deep models, accuracy (mean ± std over seeds). "Whole recording" is the M3
setting.

| Task | Model | Whole recording | None | First 5 min of wear | 5-min rest (deployable) | Wrist only, 5-min rest |
| --- | --- | --- | --- | --- | --- | --- |
| Binary | CNN + BiGRU + attention | 0.934 ± 0.014 | 0.706 ± 0.050 | 0.820 ± 0.023 | 0.806 ± 0.018 | 0.807 ± 0.015 |
| Binary | CNN | 0.927 ± 0.008 | 0.671 ± 0.023 | 0.814 ± 0.016 | 0.818 ± 0.019 | 0.783 ± 0.027 |
| Binary | Tiny separable CNN | 0.926 ± 0.016 | 0.686 ± 0.038 | 0.803 ± 0.038 | 0.807 ± 0.007 | 0.800 ± 0.021 |
| Three-class | CNN + BiGRU + attention | 0.784 ± 0.039 | 0.502 ± 0.031 | 0.649 ± 0.021 | 0.743 ± 0.032 | 0.674 ± 0.026 |
| Three-class | CNN | 0.808 ± 0.025 | 0.487 ± 0.029 | 0.620 ± 0.021 | 0.714 ± 0.019 | 0.674 ± 0.011 |
| Three-class | Tiny separable CNN | 0.748 ± 0.044 | 0.390 ± 0.031 | 0.622 ± 0.007 | 0.717 ± 0.044 | 0.698 ± 0.014 |
| Four-class | CNN + BiGRU + attention | 0.734 ± 0.012 | 0.435 ± 0.037 | 0.576 ± 0.009 | 0.635 ± 0.031 | 0.568 ± 0.016 |
| Four-class | CNN | 0.759 ± 0.007 | 0.394 ± 0.046 | 0.548 ± 0.023 | 0.635 ± 0.013 | 0.572 ± 0.013 |
| Four-class | Tiny separable CNN | 0.735 ± 0.027 | 0.351 ± 0.032 | 0.539 ± 0.037 | 0.630 ± 0.033 | 0.565 ± 0.019 |

Classical models (best of RF, AdaBoost, LDA; all physiological signals), for
comparison:

| Task | Whole recording | None | First 5 min of wear | 5-min rest (deployable) |
| --- | --- | --- | --- | --- |
| Binary | 0.984 | 0.937 | 0.897 | 0.945 |
| Three-class | 0.935 | 0.798 | 0.817 | 0.865 |
| Four-class | 0.921 | 0.781 | 0.778 | 0.818 |

- Deep models need some per-subject normalisation (without it, four-class
  falls to 0.35-0.44), and the whole-recording version hides a 3-13 point
  gap (10-13 on binary and four-class). Classical features lose 4-10 points.
- The three deep architectures are indistinguishable under deployable
  normalisation (four-class 0.630-0.635), consistent with M3: the proposed
  architecture has no advantage.
- Deployable four-class per-class F1 (CNN + BiGRU + attention): baseline
  0.85, stress 0.48, amusement 0.23, meditation 0.43. Stress recognition
  drops the most compared with M3 (0.84).

## 2. Per-user calibration

The first 1, 2 or 4 minutes of each condition of the test subject are used
for calibration; "before" is the uncalibrated model on the same remaining
windows. Classical: calibration windows added to training (weight 10). Deep:
classifier head fine-tuned.

WESAD, classical (best classifier per setting), accuracy before → after:

| Task | Signals | 1 min | 2 min | 4 min |
| --- | --- | --- | --- | --- |
| Binary | chest + wrist physio | 0.946 → 0.953 | 0.946 → 0.962 | 0.944 → 0.960 |
| Binary | wrist physio | 0.887 → 0.902 | 0.891 → 0.920 | 0.868 → 0.915 |
| Four-class | chest + wrist physio | 0.830 → 0.898 | 0.839 → 0.940 | 0.866 → 0.950 |
| Four-class | wrist physio | 0.756 → 0.776 | 0.774 → 0.836 | 0.819 → 0.857 |

Deep (mean over seeds and folds), accuracy gain from calibration:

| Setting | Model | 1 min | 2 min | 4 min |
| --- | --- | --- | --- | --- |
| WESAD four-class | CNN + BiGRU + attention | +0.036 | +0.007 | +0.062 |
| WESAD four-class | CNN | +0.036 | +0.046 | +0.086 |
| WESAD four-class | Tiny separable CNN | +0.022 | +0.027 | +0.054 |
| WESAD binary | CNN / Tiny CNN | +0.003 to +0.017 | | |
| WESAD binary | CNN + BiGRU + attention | -0.041 | -0.100 | -0.051 |
| Stress-Predict binary | all six models | -0.004 to -0.067 | -0.007 to -0.086 | -0.025 to -0.084 |

- Calibration is the strongest lever found so far for the multi-class
  problem: four minutes per condition takes classical four-class to 0.95.
  This requires the user to provide labelled minutes of each state, which is
  a strong assumption for stress and must be stated.
- On Stress-Predict calibration hurts every deep model. Its first block of
  each condition (consent forms; the Stroop task) is not representative of
  the rest of the session.

## 3. Stress-Predict (34 participants, wrist only, binary)

| Model | Normalisation | Hyperventilation as stress | Hyperventilation excluded |
| --- | --- | --- | --- |
| Best classical (RF / AdaBoost) | none | 0.742 | 0.754 |
| Best classical (AdaBoost) | 5-min rest | 0.742 | 0.762 |
| Best classical (RF) | whole recording | 0.794 | 0.807 |
| Deep, six models | 5-min rest | 0.611-0.643 | 0.617-0.640 |
| *Majority guess* | | *0.687* | *0.703* |

Macro-F1 for the deep models is 0.57-0.59 against 0.41 for the majority
guess, so they do separate the classes, but less well than the classical
features. Excluding hyperventilation changes results by about one point.

## 4. Cross-dataset: train on WESAD, test on Stress-Predict

| Model | Normalisation | Hyperventilation as stress | Excluded |
| --- | --- | --- | --- |
| Best classical (LDA) | whole recording | 0.718 | 0.728 |
| Best classical (LDA) | 5-min rest | 0.685 | 0.694 |
| Deep, best (CNN) | 5-min rest | 0.637 ± 0.026 | 0.642 ± 0.028 |
| Deep, worst (BiGRU only) | 5-min rest | 0.561 ± 0.029 | 0.558 ± 0.031 |
| *Majority guess* | | *0.687* | *0.703* |

Nothing learned on WESAD's lab protocol carries over usefully to a second
lab protocol with the same wristband. This is the clearest limitation of
WESAD-only results, and is worth stating prominently.

## What this means for the paper

- The defensible contribution is an **evaluation study**: on WESAD, a
  common normalisation shortcut inflates accuracy by 3-13 points; under
  deployable conditions, hand-crafted features beat deep models; per-user
  calibration helps classical models most; and nothing transfers to a second
  dataset.
- The deep-architecture comparison is a secondary result: small CNNs match
  the larger recurrent models in every setting.
- M4 should therefore measure the cost of the **classical feature pipeline**
  as carefully as the deep models, since it is the more accurate option.

## Caveats

- Classical "best classifier" cells are chosen on test folds (as in the
  dataset paper); deep results are single models averaged over seeds.
- Calibration "before" accuracy is computed on the windows left after
  removing calibration minutes, so it differs from the plain LOSO accuracy.
- Only three of the six deep architectures were run on WESAD with deployable
  normalisation. The other three can be added in about two GPU hours.
- No significance tests yet (M6).
