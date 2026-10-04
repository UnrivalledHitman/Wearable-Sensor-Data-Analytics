"""End-to-end smoke test on synthetic WESAD-shaped data (no dataset needed).

    python -m unittest discover -s Code/tests -t Code
"""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pipeline.config import CLASS_NAMES, CLEAN_CHANNELS, CleanPreprocessConfig, load_config
from pipeline.model import build_model, count_parameters
from pipeline.preprocess import preprocess_all, preprocessed_files
from pipeline.preprocess_clean import resample_recording, window_subject_clean
from pipeline.report import load_folds, metrics_from_cm, report
from pipeline.train import run_loso
from pipeline.wesad import pure_windows

CODE_DIR = Path(__file__).resolve().parents[1]

# (WESAD label id, seconds). 0 = undefined, 5-7 = ignore.
PROTOCOL = [(0, 30), (1, 120), (0, 20), (2, 60), (5, 10), (4, 40), (3, 40), (0, 20), (4, 40), (6, 10)]


def synthetic_subject(rng: np.random.Generator) -> dict:
    labels = np.concatenate([np.full(sec * 700, lab, dtype=np.int32) for lab, sec in PROTOCOL])
    n = len(labels)
    seconds = n / 700
    n64, n32, n4 = int(seconds * 64), int(seconds * 32), int(seconds * 4)
    return {
        "subject": "synthetic",
        "label": labels,
        "signal": {
            "chest": {
                "ACC": rng.normal(size=(n, 3)),
                "ECG": rng.normal(size=(n, 1)),
                "EDA": rng.normal(size=(n, 1)) + labels[:, None],
                "EMG": rng.normal(size=(n, 1)),
                "Resp": rng.normal(size=(n, 1)),
                "Temp": rng.normal(size=(n, 1)) + 34,
            },
            "wrist": {
                "ACC": rng.normal(size=(n32, 3)),
                "BVP": rng.normal(size=(n64, 1)),
                "EDA": rng.normal(size=(n4, 1)),
                "TEMP": rng.normal(size=(n4, 1)) + 33,
            },
        },
    }


def write_config(path: Path, root: Path, name: str, balancing: str, window_sec: int = 10) -> Path:
    path.write_text(f"""
[run]
name = "{name}"
seed = 0

[paths]
raw_dir = "{(root / 'WESAD').as_posix()}"
data_dir = "{(root / 'data').as_posix()}"
out_dir = "{(root / 'runs' / name).as_posix()}"

[preprocess]
target_rate = 32
window_sec = {window_sec}
overlap = 0.5
majority_threshold = 0.6

[model]
cnn_channels = 8
gru_hidden = 8
gru_layers = 2
attn_heads = 2
dropout = 0.3

[train]
balancing = "{balancing}"
batch_size = 16
epochs = 2
lr = 1e-3
patience = 6
grad_clip = 5.0
val_fraction = 0.1
lr_scheduler = "{'plateau' if balancing == 'weighted_loss' else 'none'}"
""")
    return path


class PipelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        raw = cls.root / "WESAD"
        raw.mkdir()
        rng = np.random.default_rng(0)
        for subject in ("S2", "S3", "S10"):
            with open(raw / f"{subject}.pkl", "wb") as fh:
                pickle.dump(synthetic_subject(rng), fh)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_1_preprocess_shapes_and_legacy_labelling(self):
        cfg = load_config(write_config(self.root / "a.toml", self.root, "loss", "weighted_loss"))
        manifest = json.loads(preprocess_all(cfg).read_text())
        files = preprocessed_files(cfg)
        self.assertEqual([f.stem for f in files], ["S10_combined", "S2_combined", "S3_combined"])

        with np.load(files[0]) as arr:
            X, y = arr["X"], arr["y"]
        self.assertEqual(X.shape[1:], (320, 8))
        self.assertEqual(len(X), len(y))
        self.assertEqual(set(np.unique(y)), {0, 1, 2, 3})

        # Legacy behaviour: undefined/ignored samples end up in "baseline", so that
        # class is far larger than the 120 s of real baseline in the protocol.
        counts = manifest["subjects"]["S2"]["window_counts"]
        real_baseline_share = 120 / sum(sec for _, sec in PROTOCOL)
        self.assertGreater(counts["baseline"] / sum(counts.values()), real_baseline_share + 0.1)

    def test_2_stale_data_is_refused(self):
        cfg60 = load_config(write_config(self.root / "b.toml", self.root, "w60", "none", window_sec=60))
        with self.assertRaisesRegex(ValueError, "was preprocessed with"):
            preprocessed_files(cfg60)

    def test_3_train_and_report(self):
        device = torch.device("cpu")
        cfgs = [
            load_config(write_config(self.root / f"{name}.toml", self.root, name, balancing))
            for name, balancing in (("loss", "weighted_loss"), ("sampler", "weighted_sampler"), ("plain", "none"))
        ]
        for cfg in cfgs:
            results = run_loso(cfg, device=device)
            self.assertEqual([r["subject"] for r in results], ["S10", "S2", "S3"])
            for r in results:
                self.assertEqual(np.array(r["confusion_matrix"]).shape, (4, 4))
                self.assertTrue((cfg.out_dir / "checkpoints" / f"best_{r['subject']}.pt").exists())
            self.assertTrue((cfg.out_dir / "run_manifest.json").exists())

        out = self.root / "report"
        report({cfg.name: cfg.out_dir for cfg in cfgs}, out)
        summary = json.loads((out / "loss_summary.json").read_text())
        self.assertEqual(summary["n_folds"], 3)
        tests = json.loads((out / "comparison_tests.json").read_text())
        self.assertEqual(set(tests), {"sampler vs loss", "plain vs loss"})
        for name in ("loss_confusion_matrix.png", "compare_f1_macro.png", "compare_per_class_f1.png"):
            self.assertTrue((out / name).exists())

    def test_4_same_seed_same_result(self):
        cfg = load_config(write_config(self.root / "det.toml", self.root, "det", "weighted_sampler"))
        first = run_loso(cfg, device=torch.device("cpu"))
        second = run_loso(cfg, device=torch.device("cpu"))
        self.assertEqual([r["confusion_matrix"] for r in first], [r["confusion_matrix"] for r in second])

        # Resuming after losing one fold retrains only that fold, with the same result.
        (cfg.out_dir / "fold_S2.json").unlink()
        stamp = (cfg.out_dir / "fold_S10.json").stat().st_mtime_ns
        resumed = run_loso(cfg, device=torch.device("cpu"), resume=True)
        self.assertEqual((cfg.out_dir / "fold_S10.json").stat().st_mtime_ns, stamp)
        self.assertEqual([r["confusion_matrix"] for r in resumed], [r["confusion_matrix"] for r in first])


class CleanPreprocessTest(unittest.TestCase):
    """The corrected pipeline keeps only windows inside one condition."""

    def setUp(self):
        self.data = synthetic_subject(np.random.default_rng(1))

    def test_only_condition_windows_survive(self):
        cfg = CleanPreprocessConfig(target_rate=32, window_sec=10, stride_sec=10)
        X, y, _ = window_subject_clean(self.data, cfg)
        self.assertEqual(X.shape[1:], (320, len(CLEAN_CHANNELS)))

        # Non-overlapping 10 s windows: each condition yields floor(duration / 10)
        # per block at most, and nothing comes from labels 0 or 5-7.
        seconds = {1: 120, 2: 60, 3: 40, 4: 80}
        counts = np.bincount(y, minlength=4)
        for label, sec in seconds.items():
            self.assertLessEqual(counts[label - 1], sec // 10)
            self.assertGreaterEqual(counts[label - 1], sec // 10 - 2)
        self.assertEqual(len(y), counts.sum())

    def test_subject_normalisation_and_channel_subset(self):
        cfg = CleanPreprocessConfig(target_rate=64, window_sec=10, stride_sec=5, channels=["ch_temp", "wr_bvp"])
        signals, lab = resample_recording(self.data, cfg)
        self.assertEqual(signals.shape[1], 2)
        self.assertEqual(len(signals), len(lab))
        self.assertAlmostEqual(len(signals) / 64, len(self.data["label"]) / 700, delta=1.0)
        # Temperature sits near 34 before normalisation; resampling must not distort its level.
        self.assertAlmostEqual(signals[:, 0].mean(), 34.0, delta=0.1)

        X, _, _ = window_subject_clean(self.data, cfg)
        self.assertLess(abs(X[..., 0].mean()), 1.0)  # standardised, no longer near 34

    def test_pure_windows_reject_mixed_labels(self):
        lab = np.array([0] * 10 + [1] * 30 + [5] * 5 + [2] * 20)
        starts, labels = pure_windows(lab, rate=1, window_sec=10, stride_sec=5)
        self.assertEqual(starts.tolist(), [10.0, 15.0, 20.0, 25.0, 30.0, 45.0, 50.0, 55.0])
        self.assertEqual(labels.tolist(), [1, 1, 1, 1, 1, 2, 2, 2])
        # A lower purity threshold admits the three windows that are half one condition.
        starts50, _ = pure_windows(lab, rate=1, window_sec=10, stride_sec=5, min_purity=0.5)
        self.assertEqual(len(starts50), len(starts) + 3)

    def test_unknown_channel_is_rejected(self):
        with self.assertRaises(ValueError):
            CleanPreprocessConfig(target_rate=32, window_sec=10, stride_sec=5, channels=["ch_bvp"])


class PaperNumbersTest(unittest.TestCase):
    """The committed fold results must reproduce the numbers in the draft."""

    def test_parameter_count(self):
        cfg = load_config(CODE_DIR / "configs" / "legacy_weighted_sampler.toml")
        self.assertEqual(count_parameters(build_model(8, cfg.model)), 756_484)

    def test_committed_results(self):
        expected = {
            "results_training_hybrid": (0.6327, 0.2845),
            "results_evaluation": (0.6443, 0.2962),
        }
        for folder, (acc, f1) in expected.items():
            folds = load_folds(CODE_DIR / folder)
            self.assertEqual(len(folds), 15)
            rows = [metrics_from_cm(cm) for cm in folds.values()]
            self.assertAlmostEqual(np.mean([r["accuracy"] for r in rows]), acc, places=4)
            self.assertAlmostEqual(np.mean([r["f1_macro"] for r in rows]), f1, places=4)
            self.assertEqual(len(CLASS_NAMES), 4)


if __name__ == "__main__":
    unittest.main()
