"""Tests for hand-crafted features and the classical baselines (no dataset needed)."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.baselines import (
    BaselineConfig, evaluate, guesser_reference, standardise_per_subject,
)
from pipeline.features import (
    UNLABELLED, beat_intervals, detect_r_peaks, hrv_features, subject_features, window_grid,
)
from tests.test_smoke import synthetic_subject


def synthetic_ecg(beat_times: np.ndarray, duration: float, fs: int = 700) -> np.ndarray:
    """Narrow Gaussian R-waves at the given times, plus a little noise."""
    t = np.arange(int(duration * fs)) / fs
    ecg = np.zeros_like(t)
    for bt in beat_times:
        ecg += np.exp(-0.5 * ((t - bt) / 0.012) ** 2)
    return ecg + np.random.default_rng(0).normal(scale=0.02, size=len(t))


class BeatDetectionTest(unittest.TestCase):
    def test_r_peaks_recovered_to_within_10_ms(self):
        rng = np.random.default_rng(1)
        beats = np.cumsum(0.8 + rng.normal(scale=0.03, size=140)) + 1.0
        found = detect_r_peaks(synthetic_ecg(beats, beats[-1] + 2))
        self.assertEqual(len(found), len(beats))
        self.assertLess(np.abs(found - beats).max(), 0.010)

    def test_hrv_of_a_steady_rhythm(self):
        peaks = np.arange(0, 120, 0.8)
        out = hrv_features(*beat_intervals(peaks), 10.0, 70.0, "x_")
        self.assertAlmostEqual(out["x_hr_mean"], 75.0, places=6)
        self.assertAlmostEqual(out["x_sdnn"], 0.0, places=9)
        self.assertEqual(out["x_nn50"], 0)

    def test_implausible_intervals_are_excluded(self):
        peaks = np.delete(np.arange(0, 120, 0.8), 50)  # one missed beat -> a 1.6 s interval
        ibi, _, valid = beat_intervals(peaks)
        self.assertEqual(int((~valid).sum()), 1)
        self.assertAlmostEqual(ibi[~valid][0], 1.6)

    def test_too_few_beats_gives_nan(self):
        out = hrv_features(*beat_intervals(np.arange(0, 5, 0.8)), 0.0, 60.0, "x_")
        self.assertTrue(all(np.isnan(v) for v in out.values()))


class WindowAndReferenceTest(unittest.TestCase):
    def test_window_grid_marks_mixed_windows_unlabelled(self):
        lab = np.concatenate([np.zeros(700 * 10), np.ones(700 * 20), np.full(700 * 10, 5)]).astype(int)
        starts, labels = window_grid(lab, window_sec=10, stride_sec=5, min_purity=1.0)
        self.assertEqual(starts.tolist(), [0, 5, 10, 15, 20, 25, 30])
        self.assertEqual(labels.tolist(), [UNLABELLED, UNLABELLED, 1, 1, 1, UNLABELLED, UNLABELLED])

    def test_guessers(self):
        ref = guesser_reference(np.array([0] * 5 + [1] * 3 + [2] * 2), 3)
        self.assertAlmostEqual(ref["majority_accuracy"], 0.5)
        self.assertAlmostEqual(ref["majority_f1_macro"], (2 * 0.5 / 1.5) / 3)
        self.assertAlmostEqual(ref["random_accuracy"], 1 / 3)

    def test_per_subject_standardisation(self):
        table = pd.DataFrame({
            "subject": ["S2"] * 4 + ["S3"] * 4, "label": [1, 2, -1, 1] * 2, "start_sec": range(8),
            "f": [1.0, 2.0, 3.0, 4.0, 101.0, 102.0, 103.0, 104.0], "const": [7.0] * 8,
        })
        out = standardise_per_subject(table)
        for _, group in out.groupby("subject"):
            self.assertAlmostEqual(group["f"].mean(), 0.0)
            self.assertAlmostEqual(group["f"].std(), 1.0)
        self.assertTrue((out["const"] == 0).all())  # a constant feature must not become NaN
        self.assertEqual(out["label"].tolist(), table["label"].tolist())


class BaselinePipelineTest(unittest.TestCase):
    def test_features_and_resumable_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "WESAD").mkdir()
            rng = np.random.default_rng(0)
            for subject in ("S2", "S3", "S10"):
                with open(root / "WESAD" / f"{subject}.pkl", "wb") as fh:
                    pickle.dump(synthetic_subject(rng), fh)

            table = subject_features(synthetic_subject(rng), "S2", window_sec=10, stride_sec=5)
            self.assertEqual(set(table["label"]), {UNLABELLED, 1, 2, 3, 4})
            prefixes = {c.rsplit("_", 1)[0].split("_")[0] + "_" + c.split("_")[1] for c in table.columns[3:]}
            self.assertEqual(prefixes, {"chest_ecg", "chest_eda", "chest_emg", "chest_resp", "chest_temp",
                                        "chest_acc", "wrist_bvp", "wrist_eda", "wrist_temp", "wrist_acc"})

            cfg = BaselineConfig(
                name="t", seed=0, raw_dir=root / "WESAD", features_dir=root / "features", out_dir=root / "out",
                window_sec=[10], stride_sec=5, min_label_purity=1.0,
                tasks=["binary", "four_class"], feature_sets=["chest_physio"], classifiers=["DT", "LDA"],
                subject_normalisation=[False, True],
            )
            cfg.out_dir.mkdir()
            self.assertEqual(len(evaluate(cfg, n_jobs=1, max_minutes=0)), 0)  # budget spent: nothing starts

            summary = evaluate(cfg, n_jobs=1)
            self.assertEqual(len(summary), 2 * 1 * 2 * 2)
            self.assertEqual(len(list((cfg.out_dir / "combinations").glob("*.json"))), 8)
            self.assertTrue(summary["accuracy_mean"].between(0, 1).all())
            # Binary leaves meditation out, so it sees fewer windows than four-class.
            n = summary.groupby("task")["n_windows"].first()
            self.assertLess(n["binary"], n["four_class"])

            again = evaluate(cfg, n_jobs=1)  # everything cached
            pd.testing.assert_frame_equal(summary, again)


if __name__ == "__main__":
    unittest.main()
