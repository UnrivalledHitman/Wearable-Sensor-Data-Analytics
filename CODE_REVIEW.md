# Comprehensive Code Review — Wearable Sensor Data Analytics (WESAD)

## Scope Reviewed
- Project documentation and structure.
- End-to-end pipeline notebook (`wesad_full_pipeline_code.ipynb`) covering preprocessing, EDA, model definition, training, and evaluation.
- Auxiliary validation notebook (`wesad_test.ipynb`).
- Existing saved evaluation artifacts in `Code/results_training_hybrid` and `Code/results_evaluation`.

## Executive Summary
The project has a good practical baseline: a complete LOSO pipeline, a reasonable hybrid model (CNN + BiGRU + attention), and paper-friendly result exports. However, there are **several methodological and reproducibility issues** that can materially affect scientific validity.

Top priorities to fix before publication:
1. **Label downsampling is mathematically incorrect** (mean + cast to int), which can distort class labels.
2. **Sampling-rate handling is inconsistent** (upsample wrist to chest, then downsample all), introducing avoidable signal distortion.
3. **Baseline branch is not reproducible from the main notebook as written** (expects files not generated in that branch).
4. **Validation split can leak temporally-near windows across train/val.**
5. **Documentation is too sparse for reproducibility**.

---

## Major Findings

### 1) Critical: Label downsampling can silently relabel data
In preprocessing, labels are reduced via `block_downsample_1d(...).astype(np.int32)`. Because `block_downsample_1d` computes a mean, label windows with mixed classes are converted by truncation instead of majority/argmax semantics.

**Risk:** invalid labels and biased class assignment in training/evaluation.

**Recommendation:** for labels, replace mean downsampling with mode/majority vote only (or derive labels purely at window stage without numeric averaging).

### 2) Critical: Signal resampling strategy likely degrades fidelity
Pipeline upsamples wrist signals to chest length and then downsamples all channels using a fixed factor from chest frequency (`int(700/32)`), which is not exact resampling and can smear high-frequency/transition information.

**Risk:** feature distortion and reduced physiological validity.

**Recommendation:** resample each modality directly from native rate to target rate with explicit anti-alias filtering and proper resampling ratio.

### 3) High: Baseline evaluation path is disconnected from baseline training path
Cell 6a/6c expects `results_training_hybrid/results_*.json`, while baseline training cell focuses on model checkpoints and does not show corresponding baseline result-file generation in the same flow.

**Risk:** reproducibility gap; notebook reruns may fail to recreate reported baseline comparisons.

**Recommendation:** unify into one deterministic runner that always emits baseline and balanced metrics with identical schema.

### 4) High: Validation split method can inflate validation signal
For each LOSO fold, validation samples are random windows from pooled training subjects/windows. With overlapping windows, highly similar neighboring segments can land in both train and val.

**Risk:** optimistic early stopping/model selection.

**Recommendation:** split by contiguous blocks or subject-session grouping before windowing (or at least by non-overlapping time blocks).

### 5) Medium: EDA raw snapshot indexing appears incorrect for 3D tensor
`X` is windowed (`n_windows x time x channels`), but plotting uses `X[:win_len, i]`, which indexes windows/channels rather than time-series of a selected window.

**Risk:** misleading visualization in reports.

**Recommendation:** use `X[window_idx, :win_len, i]`.

### 6) Medium: API mismatch in test notebook
`wesad_test.ipynb` calls `build_model(..., num_classes=NUM_CLASSES)` while `build_model` in the main notebook does not accept `num_classes`.

**Risk:** auxiliary notebook failure/confusion.

**Recommendation:** align signatures or update test call.

### 7) Medium: Documentation/reproducibility debt
README is only two lines and does not document dependencies, data layout, execution order, expected runtime, or exact experiment protocol.

**Risk:** hard for reviewers/supervisors to reproduce results.

**Recommendation:** add full reproducibility README with environment, commands, and experiment manifest.

---

## Strengths
- Clean modular architecture definition and helper utilities.
- Balanced training/evaluation branch stores confusion matrices and per-class reports.
- Result artifacts are exported in CSV/JSON/plots suitable for research reporting.

## Suggested Next Milestones
1. Fix preprocessing/resampling + label handling and regenerate datasets.
2. Refactor notebook into scripts (`preprocess.py`, `train.py`, `evaluate.py`) with CLI args.
3. Add deterministic experiment tracking (seed log, config JSON per run, software versions).
4. Add ablations: baseline CNN/GRU only, no-attention, class-weighted CE vs sampler.
5. Add confidence intervals (bootstrap across subjects) for key metrics.

## Quick Risk Rating
- **Scientific validity risk:** High (until preprocessing/labeling is corrected).
- **Engineering maintainability risk:** Medium.
- **Reproducibility risk:** High.
