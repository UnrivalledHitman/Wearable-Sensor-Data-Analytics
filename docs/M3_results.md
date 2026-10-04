# M3 results

Grid finished 2026-10-04: 32 configs × 5 seeds = 160 runs, each a 15-fold
leave-one-subject-out evaluation. Each number is the mean over 5 seeds of a
run's mean over subjects; ± is the standard deviation across seeds. Clean
data pipeline, per-subject normalisation, early stopping on two held-out
training subjects. Source: `Code/runs/m3_summary/seed_summary.csv`
(regenerate with `python Code/run_pipeline.py seeds --runs Code/runs/m3 --out Code/runs/m3_summary/seed_summary.csv`).

## Verdict

**Exit gate (every class F1 above 0.5): not met.** Amusement F1 is 0.17-0.45
for every deep model. Baseline, stress and meditation are recognised well.

**The proposed CNN + BiGRU + attention model has no advantage.** On four- and
three-class it is matched or beaten by a plain CNN with 11× fewer parameters;
on binary all six models fall within 1 point. Every deep model stays well
below the hand-crafted feature baseline from M2.

## Main comparison (60 s windows)

Accuracy, mean ± std over 5 seeds:

| Model | Parameters | Binary | Three-class | Four-class | Four-class macro-F1 |
| --- | --- | --- | --- | --- | --- |
| CNN + BiGRU + attention (proposed) | 809k | **0.934 ± 0.014** | 0.784 ± 0.039 | 0.734 ± 0.012 | 0.646 |
| CNN + BiLSTM + attention | 957k | 0.927 ± 0.007 | 0.798 ± 0.058 | 0.745 ± 0.029 | 0.642 |
| CNN-LSTM, one direction | 315k | 0.925 ± 0.015 | 0.771 ± 0.016 | 0.741 ± 0.022 | 0.645 |
| CNN only | 71k | 0.927 ± 0.008 | **0.808 ± 0.025** | **0.759 ± 0.007** | **0.666** |
| BiGRU only | 704k | 0.931 ± 0.004 | 0.767 ± 0.023 | 0.738 ± 0.025 | 0.659 |
| Tiny separable CNN | 5k | 0.926 ± 0.016 | 0.748 ± 0.044 | 0.735 ± 0.027 | 0.635 |
| *Hand-crafted features + RF/AdaBoost (M2)* | | *0.970* | *0.923* | *0.906* | *0.837* |
| *Always-majority guesser* | | *0.702* | *0.549* | *0.413* | *0.146* |

Per-class F1, four-class:

| Model | Baseline | Stress | Amusement | Meditation |
| --- | --- | --- | --- | --- |
| CNN + BiGRU + attention | 0.80 | 0.84 | 0.23 | 0.71 |
| CNN only | 0.83 | 0.86 | 0.24 | 0.73 |
| BiGRU only | 0.76 | 0.85 | 0.28 | 0.74 |
| Tiny separable CNN | 0.79 | 0.81 | 0.22 | 0.72 |

On three-class, amusement F1 rises to 0.36-0.45, still below the gate.

## Claims from the paper draft, tested

| Claim | Result |
| --- | --- |
| GRU is lighter than LSTM at equal quality | Lighter: 809k vs 957k parameters, same time per epoch. Quality: no reliable difference; BiLSTM is within seed noise on all three tasks |
| CNN-BiGRU beats conventional CNN-LSTM | Not supported: within 1 point on binary, +1.3 three-class, -0.7 four-class, all within seed noise |
| Attention helps | Not supported: removing it gives 0.739 vs 0.734 four-class |
| Recurrent layers are needed | Not supported: CNN only is the best four- and three-class model |

## Design check: 16× downsampling (four-class, 10 s, 5 seeds)

| Front-end | Accuracy | Macro-F1 | s/epoch | Peak GPU |
| --- | --- | --- | --- | --- |
| No downsampling | 0.726 ± 0.015 | 0.629 | 1.96 | 697 MB |
| 16× downsampling | 0.719 ± 0.019 | 0.630 | 0.53 | 148 MB |

Over 5 seeds the accuracy difference shrinks to 0.7 points, within seed
noise, while training is 3.7× faster and uses 4.7× less memory.

## Window length (proposed model)

| Task | 10 s | 30 s | 60 s |
| --- | --- | --- | --- |
| Binary | 0.914 ± 0.016 | 0.929 ± 0.011 | 0.934 ± 0.014 |
| Three-class | 0.781 ± 0.016 | 0.777 ± 0.016 | 0.784 ± 0.039 |
| Four-class | 0.719 ± 0.019 | 0.740 ± 0.027 | 0.734 ± 0.012 |

Longer windows help binary by 2 points and the multi-class tasks little. The
features in M2 gained 4-5 points from the same change.

## Ablations (proposed model, four-class, 60 s)

| Variant | Parameters | Accuracy | Macro-F1 |
| --- | --- | --- | --- |
| Full model (hidden 128, 2 layers, attention, all signals) | 809k | 0.734 ± 0.012 | 0.646 |
| Hidden 64 | 274k | **0.758 ± 0.016** | **0.662** |
| Hidden 32 | 125k | 0.732 ± 0.014 | 0.642 |
| One recurrent layer | 513k | 0.746 ± 0.025 | 0.654 |
| No attention | 546k | 0.739 ± 0.018 | 0.637 |
| Chest signals only | 807k | 0.727 ± 0.026 | 0.618 |
| Wrist signals only | 806k | 0.609 ± 0.017 | 0.546 |

The model is over-sized: a third of the parameters does as well or better.
Wrist-only loses 12.5 points, matching the chest/wrist gap seen in M2.

## What this means for the paper

1. The defensible contribution is not an architecture. Small models
   (71k-parameter CNN, even the 5k tiny CNN on binary) match the larger
   recurrent ones, which supports a cost-versus-accuracy study for M4.
2. Hand-crafted features remain far ahead on the multi-class tasks, so the
   paper must report them alongside the deep models.
3. Amusement is the open problem for every method. A dedicated analysis
   (confusions with baseline; self-reports showed amusement had only a small
   effect in the dataset paper) belongs in M6.

## Caveats

- 5 seeds; no significance tests yet (M6).
- One training recipe for all models, not tuned per model; a tuned model
  could do better, and the paper should say so.
- Subject-wise validation leaves 12 training subjects per fold, while the M2
  baselines train on 14.
