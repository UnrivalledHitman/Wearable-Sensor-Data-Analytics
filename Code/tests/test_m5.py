"""Tests for M5: Stress-Predict loading, deployable normalisation, calibration, cross-dataset."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline import datasets, stresspredict
from pipeline.baselines import BaselineConfig, calibration_split, evaluate, standardise_per_subject
from pipeline.config import CleanPreprocessConfig, load_config, load_cross_config, upgrade_saved_config
from pipeline.preprocess import preprocess_all
from pipeline.preprocess_clean import normalisation_reference
from pipeline.train import run_cross_dataset, run_loso
from tests.test_smoke import synthetic_subject

WRIST = ["wr_bvp", "wr_eda", "wr_temp", "wr_acc_x", "wr_acc_y", "wr_acc_z"]
# (label, seconds): rest, Stroop, rest, interview, rest, hyperventilation, rest
SP_PROTOCOL = [(0, 180), (1, 120), (0, 120), (1, 180), (0, 120), (1, 60), (0, 180)]


def write_stress_predict(root: Path, participants=(2, 3, 4), seed=0) -> Path:
    """A miniature Stress-Predict folder with the real file layout."""
    rng = np.random.default_rng(seed)
    labels = []
    for p in participants:
        folder = root / "Raw_data" / f"S{p:02d}"
        folder.mkdir(parents=True)
        t0 = 1_644_000_000 + 10_000 * p
        seconds = sum(sec for _, sec in SP_PROTOCOL) + 20
        for name, fs, cols in (("ACC", 32, 3), ("BVP", 64, 1), ("EDA", 4, 1), ("TEMP", 4, 1)):
            data = rng.normal(size=(seconds * fs, cols)) + (33 if name == "TEMP" else 0)
            header = pd.DataFrame([[float(t0)] * cols, [float(fs)] * cols])
            pd.concat([header, pd.DataFrame(data)]).to_csv(folder / f"{name}.csv", header=False, index=False)
        second = t0 + 12
        for lab, sec in SP_PROTOCOL:
            for _ in range(sec):
                labels.append({"Participant": p, "HR": 80, "respr": 12, "Time(sec)": second, "Label": lab})
                second += 1
    (root / "Processed_data").mkdir()
    pd.DataFrame(labels).to_csv(root / stresspredict.LABELS_FILE, index=False)
    return root


class StressPredictTest(unittest.TestCase):
    def test_adapter_labels_and_hyperventilation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_stress_predict(Path(tmp))
            files = datasets.subject_files("stress_predict", root)
            self.assertEqual([f.name for f in files], ["S02", "S03", "S04"])
            data = datasets.load_subject("stress_predict", files[0])
            self.assertEqual(set(data["signal"]), {"wrist"})
            counts = {k: v / 700 for k, v in zip(*np.unique(data["label"], return_counts=True))}
            self.assertAlmostEqual(counts[2], 360, delta=1)   # 120 + 180 + 60 s of stress
            self.assertAlmostEqual(counts[1], 600, delta=1)   # rest
            no_hv = datasets.load_subject("stress_predict", files[0], exclude_hyperventilation=True)
            counts = {k: v / 700 for k, v in zip(*np.unique(no_hv["label"], return_counts=True))}
            self.assertAlmostEqual(counts[stresspredict.LABEL_IGNORED], 60, delta=1)
            self.assertAlmostEqual(counts[2], 300, delta=1)

    def test_unlabelled_participant_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_stress_predict(Path(tmp))
            (root / "Raw_data" / "S01").mkdir()
            self.assertNotIn("S01", [f.name for f in datasets.subject_files("stress_predict", root)])

    def test_chest_channels_rejected_for_stress_predict(self):
        with self.assertRaises(ValueError):
            CleanPreprocessConfig(target_rate=32, window_sec=10, stride_sec=5, dataset="stress_predict")


class NormalisationTest(unittest.TestCase):
    def test_reference_samples(self):
        signals = np.arange(100, dtype=float)[:, None]
        lab = np.array([0] * 20 + [1] * 50 + [2] * 30)
        cfg = CleanPreprocessConfig(target_rate=1, window_sec=10, stride_sec=5,
                                    normalisation_source="first_minutes", normalisation_minutes=0.5)
        self.assertEqual(normalisation_reference(signals, lab, cfg)[:, 0].tolist(), list(range(30)))
        cfg = CleanPreprocessConfig(target_rate=1, window_sec=10, stride_sec=5,
                                    normalisation_source="rest_minutes", normalisation_minutes=0.5)
        self.assertEqual(normalisation_reference(signals, lab, cfg)[:, 0].tolist(), list(range(20, 50)))

    def test_feature_standardisation_variants(self):
        rows = []
        for subject, offset in (("S2", 0.0), ("S3", 100.0)):
            for i in range(40):
                rows.append({"subject": subject, "label": 1 if 10 <= i < 30 else 2, "start_sec": i * 5.0,
                             "f": offset + i})
        table = pd.DataFrame(rows)
        first = standardise_per_subject(table, "first_minutes:1", window_sec=10, stride_sec=5)
        rest = standardise_per_subject(table, "rest_minutes:1", window_sec=10, stride_sec=5)
        for out in (first, rest):
            # Each subject's own offset is removed, so the two subjects line up.
            np.testing.assert_allclose(out[out.subject == "S2"]["f"].to_numpy(),
                                       out[out.subject == "S3"]["f"].to_numpy())
        self.assertAlmostEqual(first[first.subject == "S2"]["f"].iloc[:11].mean(), 0.0)  # first 60 s of windows

    def test_old_saved_settings_still_match(self):
        cfg = CleanPreprocessConfig(target_rate=32, window_sec=10, stride_sec=5)
        from dataclasses import asdict
        saved = asdict(cfg)
        for key in ("normalisation_source", "normalisation_minutes", "dataset", "exclude_hyperventilation"):
            del saved[key]
        self.assertEqual(upgrade_saved_config(saved), asdict(cfg))


class CalibrationSplitTest(unittest.TestCase):
    def test_first_minutes_of_each_class(self):
        starts = np.arange(0, 600, 10.0)
        y = np.array([0] * 20 + [1] * 20 + [0] * 20)  # class 0 has two blocks
        calib, test = calibration_split(starts, y, minutes=1, window_sec=30, stride_sec=10)
        self.assertEqual(starts[calib].tolist(), [0.0, 10.0, 20.0, 30.0, 200.0, 210.0, 220.0, 230.0])
        self.assertFalse((calib & test).any())
        # Windows overlapping a calibration minute are not tested on.
        self.assertFalse(test[starts < 60].any())
        self.assertTrue(test[starts >= 400].all())  # the second class-0 block is all test


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        (cls.root / "WESAD").mkdir()
        rng = np.random.default_rng(0)
        for subject in ("S2", "S3", "S10"):
            with open(cls.root / "WESAD" / f"{subject}.pkl", "wb") as fh:
                pickle.dump(synthetic_subject(rng), fh)
        write_stress_predict(cls.root / "SP", participants=(2, 3, 4, 5))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _config(self, name, dataset="wesad", calibration="[]", norm="rest_minutes", task="binary"):
        raw = "SP" if dataset == "stress_predict" else "WESAD"
        path = self.root / f"{name}.toml"
        path.write_text(f"""
[run]
name = "{name}"
seed = 0
[paths]
raw_dir = "{(self.root / raw).as_posix()}"
data_dir = "{(self.root / f'data_{dataset}_{norm}').as_posix()}"
out_dir = "{(self.root / 'runs' / name).as_posix()}"
[preprocess]
mode = "clean"
dataset = "{dataset}"
target_rate = 32
window_sec = 10
stride_sec = 5
normalisation_source = "{norm}"
normalisation_minutes = 1.0
channels = {json.dumps(WRIST)}
[model]
arch = "seq"
conv_channels = [8, 8]
downsample = [2, 2]
rnn = "gru"
rnn_hidden = 8
attn_heads = 2
[train]
task = "{task}"
balancing = "weighted_sampler"
batch_size = 16
epochs = 2
lr = 1e-3
patience = 3
grad_clip = 5.0
val_fraction = 0.1
validation = "subjects"
val_subjects = 1
calibration_minutes = {calibration}
calibration_epochs = 2
""")
        return path

    def test_deep_calibration(self):
        cfg = load_config(self._config("cal", calibration="[0.5, 1]"))
        preprocess_all(cfg)
        results = run_loso(cfg, device=torch.device("cpu"))
        for r in results:
            self.assertEqual(set(r["calibration"]), {"0.5", "1"})
            for k, c in r["calibration"].items():
                self.assertGreater(c["n_calibration"], 0)
                self.assertLess(c["n_test"], np.array(r["confusion_matrix"]).sum())
                self.assertIn("accuracy", c["before"])
            self.assertGreater(r["calibration"]["1"]["n_calibration"], r["calibration"]["0.5"]["n_calibration"])

    def test_stress_predict_loso_and_cross_dataset(self):
        sp = load_config(self._config("sp", dataset="stress_predict"))
        preprocess_all(sp)
        self.assertEqual(len(run_loso(sp, device=torch.device("cpu"))), 4)

        cross_path = self.root / "cross.toml"
        cross_path.write_text(f"""
[cross]
name = "cross"
train_config = "{self._config('wtrain').as_posix()}"
test_config = "{self._config('sptest', dataset='stress_predict').as_posix()}"
out_dir = "{(self.root / 'runs' / 'cross').as_posix()}"
""")
        train, test = load_cross_config(cross_path)
        self.assertEqual(train.name, "cross")
        preprocess_all(train)
        preprocess_all(test)
        results = run_cross_dataset(train, test, train.out_dir, device=torch.device("cpu"))
        self.assertEqual([r["subject"] for r in results], ["S02", "S03", "S04", "S05"])
        again = run_cross_dataset(train, test, train.out_dir, device=torch.device("cpu"), resume=True)
        self.assertEqual(results, again)  # resumed, not retrained

    def test_classical_calibration_and_cross(self):
        common = dict(seed=0, features_dir=self.root / "features", window_sec=[10], stride_sec=5,
                      min_label_purity=1.0, tasks=["binary"], feature_sets=["wrist_physio"],
                      classifiers=["DT", "LDA"], normalisation=["rest_minutes:1"])
        cal = BaselineConfig(name="c", raw_dir=self.root / "WESAD", out_dir=self.root / "bcal",
                             calibration_minutes=[1], **common)
        summary = evaluate(cal, n_jobs=1)
        self.assertEqual(len(summary), 4)  # 2 classifiers x (plain + 1 calibration amount)
        self.assertTrue(summary["accuracy_before_mean"].notna().sum() == 2)

        cross = BaselineConfig(name="x", raw_dir=self.root / "WESAD", out_dir=self.root / "bcross",
                               test_dataset="stress_predict", test_raw_dir=self.root / "SP", **common)
        summary = evaluate(cross, n_jobs=1)
        self.assertEqual(len(summary), 2)
        self.assertTrue((summary["test_dataset"] == "stress_predict").all())


if __name__ == "__main__":
    unittest.main()
