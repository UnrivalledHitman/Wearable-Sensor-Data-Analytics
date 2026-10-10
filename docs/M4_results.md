# M4 results: measured efficiency

Measured 2026-10-09 on the development laptop (AMD Ryzen 7 8845HS, 8 cores /
16 threads; PyTorch 2.13, CPU only). One prediction = one 60 s window
(1,920 samples x 14 channels for deep models), batch size 1; median of 300
timed runs after 50 warm-up runs. Sources: `Code/runs/m4/`.

## Summary

- **Every model is far below any real-time limit.** The slowest whole
  pipeline takes about 41 ms per window, against a new window every 10 s
  (0.4 % of the time). Latency does not decide which model to deploy.
- **Size and accuracy do.** The tiny CNN is 31 KB; LDA on wrist features is
  7 KB; the proposed CNN + BiGRU + attention model is 3.2 MB.
- **Classical models are more accurate at every size.** On binary, LDA on
  wrist features (7 KB, 6.2 ms with feature extraction) reaches 0.864 against
  0.807 for the tiny CNN (31 KB, 1.7 ms) and 0.806 for the proposed model.
- **Int8 conversion costs no accuracy** (99.9 % identical predictions) but
  on this CPU it makes recurrent models slower, not faster.

## Static cost

| Model | Parameters | Multiply-accumulates per window | fp32 size | fp16 size | int8 size |
| --- | --- | --- | --- | --- | --- |
| CNN + BiGRU + attention (proposed) | 808,964 | 135.0 M | 3,178 KB | 1,597 KB | 1,787 KB |
| CNN + BiLSTM + attention | 957,444 | 152.7 M | 3,758 KB | 1,887 KB | 1,935 KB |
| CNN-LSTM | 315,140 | 70.6 M | 1,245 KB | 629 KB | 526 KB |
| CNN only | 70,980 | 43.0 M | 289 KB | 149 KB | 278 KB |
| BiGRU only | 704,132 | 87.3 M | 2,758 KB | 1,383 KB | 1,480 KB |
| Tiny separable CNN | 4,918 | 2.1 M | 31 KB | 21 KB | 29 KB |

Int8 here is PyTorch dynamic quantisation: only linear and recurrent weights
are converted, so convolution-heavy models barely shrink.

## Latency per window (median, ms)

| Model | fp32, 1 thread | fp32, 4 threads | int8, 1 thread | int8, 4 threads |
| --- | --- | --- | --- | --- |
| CNN + BiGRU + attention | 12.90 | 14.26 | 25.75 | 31.87 |
| CNN + BiLSTM + attention | 8.98 | 5.25 | 22.09 | 30.96 |
| CNN-LSTM | 5.35 | 2.95 | 13.35 | 16.10 |
| CNN only | 3.42 | 1.53 | 3.69 | 1.77 |
| BiGRU only | 14.91 | 13.51 | 19.94 | 29.20 |
| Tiny separable CNN | 1.66 | 1.22 | 1.89 | 1.43 |

95th percentiles are within about 25 % of the medians (`latency.csv`).
Recurrent layers process time steps one after another, so extra threads help
little; dynamic int8 adds conversion work at every step and is slower on this
CPU.

## Classical pipeline

| Step | Signals / model | Median time | Size |
| --- | --- | --- | --- |
| Feature extraction | chest + wrist | 34.5 ms | - |
| Feature extraction | wrist only | 5.9 ms | - |
| Classifier | RF, all physio | 7.0 ms | 1,814 KB |
| Classifier | AdaBoost, all physio | 5.0 ms | 3,239 KB |
| Classifier | LDA, all physio | 0.34 ms | 16 KB |
| Classifier | RF, wrist physio | 7.3 ms | 2,822 KB |
| Classifier | AdaBoost, wrist physio | 6.5 ms | 6,308 KB |
| Classifier | LDA, wrist physio | 0.33 ms | 7 KB |

Feature extraction dominates: chest ECG and EMG processing at 700 Hz takes
most of the 34.5 ms.

## Int8 accuracy (LOSO, mean over 5 seeds)

| Config | fp32 accuracy | int8 accuracy | Identical predictions |
| --- | --- | --- | --- |
| Binary, CNN + BiGRU + attention | 0.8059 | 0.8058 | 99.99 % |
| Binary, CNN | 0.8178 | 0.8180 | 99.97 % |
| Binary, tiny CNN | 0.8066 | 0.8063 | 99.97 % |
| Four-class, CNN + BiGRU + attention | 0.6351 | 0.6350 | 99.91 % |
| Four-class, CNN | 0.6347 | 0.6346 | 99.95 % |
| Four-class, tiny CNN | 0.6301 | 0.6303 | 99.90 % |

The fp32 numbers reproduce the M5 results exactly, which checks that the
saved checkpoints and the re-evaluation agree.

## Accuracy against cost

`Code/runs/m4/accuracy_vs_cost.png` (points in `accuracy_vs_cost.csv`):
deployable-normalisation accuracy against total single-thread latency
(feature extraction + classifier for classical models).

| Task | Most accurate | Cheapest | Best small classical |
| --- | --- | --- | --- |
| Binary | RF, all physio: 0.945, 41 ms, 1.8 MB | Tiny CNN: 0.807, 1.7 ms, 31 KB | LDA, wrist: 0.864, 6.2 ms, 7 KB |
| Four-class | RF / AdaBoost, all physio: 0.818, 40 ms, 1.8-3.2 MB | Tiny CNN: 0.630, 1.7 ms, 31 KB | AdaBoost, wrist: 0.740, 12 ms, 6.3 MB |

## Limits

- Laptop CPU, not a wearable. All times would be several times longer on a
  smartwatch-class core, but would still sit far below the 10 s budget.
- Energy was not measured.
- Feature-extraction time is for one subject's window, single-threaded.
- Only the three deep models run under deployable normalisation in M5 appear
  on the accuracy-versus-cost plot.
