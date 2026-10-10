# M6 results: statistics and the amusement analysis

Computed 2026-10-09 by `run_pipeline.py stats` from saved M3 and M5 results.
Every result is reduced to one value per test subject (deep models: mean over
5 seeds), so all tests pair subjects. 95 % confidence intervals are bootstrap
intervals over subjects (10,000 resamples). Paired comparisons use the
Wilcoxon signed-rank test with Holm correction within each family. Full
tables: `Code/runs/m6/report.md`, `summary.csv`, `comparisons.csv`.

## Findings that hold up

| Question | Result | Holm-adjusted p |
| --- | --- | --- |
| Do classical features beat the proposed deep model under deployable normalisation? | Yes, for every task and classifier. Four-class: RF +0.183 (95 % CI +0.146 to +0.221); binary: RF +0.139 (+0.086 to +0.191) | 0.0003-0.02 |
| Does any deep architecture beat the proposed one? | No. All 16 comparisons across tasks and settings are non-significant (largest difference 0.029) | 0.39-1.0 |
| Does whole-recording normalisation inflate accuracy? | Yes, for the CNN in every variant (four-class: -0.125 to -0.365 without it) and for RF in all but one (binary with 5-min rest: -0.038, p = 0.065) | 0.0002-0.025 |
| Does per-user calibration help on WESAD four-class? | Yes, at every amount, for both CNN (+0.036 to +0.086) and RF (+0.038 to +0.101) | 0.0002-0.002 |
| Does it help on WESAD binary? | Not reliably: only RF at 1 minute (+0.007) is significant | 0.039 (RF, 1 min) |
| Does it help on Stress-Predict? | No: the CNN gets significantly worse at 2 and 4 minutes (-0.023, -0.048) | 0.021, 0.004 |
| Does excluding hyperventilation change Stress-Predict results? | Slightly for AdaBoost (+0.020); not significant for the CNN | 0.0001 / 0.25 |
| Does training on WESAD transfer to Stress-Predict? | Classical RF and AdaBoost lose about 10 points against training on Stress-Predict itself (LDA: -0.016, n.s.); deep models are mostly no worse than their already poor within-dataset results, except the BiLSTM model (-0.042) | 0.0001 (RF, AB) / 0.037 (BiLSTM) |

## Headline numbers with 95 % confidence intervals

| Result | Accuracy | 95 % CI |
| --- | --- | --- |
| WESAD four-class, proposed model, deployable | 0.635 | 0.591-0.679 |
| WESAD four-class, RF, deployable | 0.818 | 0.775-0.857 |
| WESAD four-class, LDA, deployable | 0.750 | 0.714-0.787 |
| WESAD binary, proposed model, deployable | 0.806 | 0.756-0.852 |
| WESAD binary, RF, deployable | 0.945 | 0.900-0.983 |
| Stress-Predict, proposed model | 0.634 | 0.607-0.660 |
| Stress-Predict, AdaBoost | 0.742 | 0.724-0.762 |
| WESAD → Stress-Predict, CNN | 0.637 | 0.608-0.664 |
| WESAD → Stress-Predict, LDA | 0.685 | 0.659-0.707 |

Seed-to-seed standard deviation of the deep models is 0.007-0.054; the spread
between subjects is several times larger, which is why intervals are wide.

## Amusement

Self-reports (SAM scale, amusement minus baseline, 15 subjects): valence
+0.80 and arousal +0.47 on average, and 6 of 15 subjects reported no gain in
valence at all. The amusement condition changed how many participants felt
only a little, which matches the dataset paper's own remark.

Where amusement windows go (four-class, pooled):

| Model | Predicted as baseline | as stress | correctly | as meditation |
| --- | --- | --- | --- | --- |
| Proposed model, whole-recording normalisation (M3) | 0.26 | 0.18 | 0.32 | 0.24 |
| CNN, deployable | 0.26 | 0.28 | 0.27 | 0.18 |
| RF, deployable | 0.41 | 0.17 | 0.27 | 0.15 |

Amusement is spread across all other classes rather than mistaken for one,
consistent with a weak, inconsistent physiological response. Per-subject
amusement recall is not explained by self-reported valence change (|rho| <=
0.28, all p > 0.3). It correlates negatively with arousal change for RF (rho
= -0.72, p = 0.003; uncorrected, one of six correlations, n = 15): subjects
who felt more aroused during the funny clips were more often classified as
stressed or meditating. Treat this as a lead, not a finding.

## Caveats

- 15 WESAD and 34 Stress-Predict subjects: confidence intervals are wide and
  small differences cannot be resolved.
- Classical results are single runs (deterministic); deep results average 5
  seeds, so their per-subject values are less noisy.
- Families are corrected separately; across the whole report about 70 tests
  were run, so isolated p-values near 0.05 should be read with care.
