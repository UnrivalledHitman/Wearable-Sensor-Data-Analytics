"""Tests for M4 (efficiency) and M6 (statistics). No dataset needed."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pipeline import efficiency, stats
from pipeline.config import load_config
from pipeline.features import subject_features
from pipeline.preprocess import preprocess_all
from pipeline.train import run_loso
from tests.test_smoke import synthetic_subject

QUEST = """# Subj;S2;;;
# ORDER;Base;TSST;Medi 1;Fun;Medi 2;sRead;fRead;;
# START;7.08;39.55;70.19;81.25;93.38;54.42;89.51;;
# END;26.32;50.3;77.1;87.47;100.15;56.07;91.15;;
;;;;
# DIM;5;2;;
# DIM;3;7;;
# DIM;6;2;;
# DIM;8;4;;
# DIM;6;1;;
"""


def tiny_config(root: Path, name: str = "tiny") -> Path:
    path = root / f"{name}.toml"
    path.write_text(f"""
[run]
name = "{name}"
seed = 0
[paths]
raw_dir = "{(root / 'WESAD').as_posix()}"
data_dir = "{(root / 'data').as_posix()}"
out_dir = "{(root / 'runs' / name).as_posix()}"
[preprocess]
mode = "clean"
target_rate = 32
window_sec = 10
stride_sec = 5
[model]
arch = "seq"
conv_channels = [8, 8]
downsample = [2, 2]
rnn = "gru"
rnn_hidden = 8
attn_heads = 2
[train]
task = "four_class"
balancing = "weighted_sampler"
batch_size = 16
epochs = 2
lr = 1e-3
patience = 3
grad_clip = 5.0
val_fraction = 0.1
validation = "subjects"
val_subjects = 1
""")
    return path


class MacCountTest(unittest.TestCase):
    def test_layer_formulas(self):
        self.assertEqual(efficiency.count_macs(nn.Linear(10, 5), torch.randn(1, 3, 10)), 3 * 10 * 5)

        class Conv(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv1d(2, 4, 3, padding=1)

            def forward(self, x):
                return self.c(x)

        self.assertEqual(efficiency.count_macs(Conv(), torch.randn(1, 2, 8)), 4 * 8 * 2 * 3)

        class Rnn(nn.Module):
            def __init__(self):
                super().__init__()
                self.g = nn.GRU(4, 3, batch_first=True)

            def forward(self, x):
                return self.g(x)[0]

        self.assertEqual(efficiency.count_macs(Rnn(), torch.randn(1, 5, 4)), 5 * 3 * (4 * 3 + 3 * 3))

    def test_quantisation_keeps_outputs_and_shrinks_recurrent_models(self):
        model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 4)).eval()
        q = efficiency.quantise(model)
        x = torch.randn(2, 64)
        self.assertEqual(q(x).shape, (2, 4))
        self.assertLess(efficiency.state_size_bytes(q), efficiency.state_size_bytes(model))


class EfficiencyPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        (cls.root / "WESAD").mkdir()
        rng = np.random.default_rng(0)
        for subject in ("S2", "S3", "S10"):
            with open(cls.root / "WESAD" / f"{subject}.pkl", "wb") as fh:
                pickle.dump(synthetic_subject(rng), fh)
        cls.cfg_path = tiny_config(cls.root)
        cfg = load_config(cls.cfg_path).with_seed(0)
        preprocess_all(cfg)
        run_loso(cfg, device=torch.device("cpu"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_static_and_latency(self):
        static = efficiency.static_costs([str(self.cfg_path)])
        self.assertGreater(static.loc[0, "macs_per_window"], 0)
        self.assertEqual(static.loc[0, "input"], "320 x 14")
        lat = efficiency.latency([str(self.cfg_path)], threads=[1], warmup=1, runs=3, stride_sec=5)
        self.assertEqual(set(lat.precision), {"fp32", "int8"})
        self.assertTrue((lat.median_ms > 0).all())

    def test_quantised_accuracy_matches_saved_runs(self):
        out = self.root / "quantised"
        efficiency.quantised_accuracy([(str(self.cfg_path), 0)], self.root / "runs", out, threads=1)
        result = json.loads((out / "tiny_seed0.json").read_text())
        self.assertEqual(len(result["folds"]), 3)
        # float32 re-evaluation reproduces the accuracy saved during training
        saved = {json.loads(p.read_text())["subject"]: json.loads(p.read_text())["accuracy"]
                 for p in (self.root / "runs" / "tiny_seed0").glob("fold_*.json")}
        for fold in result["folds"]:
            self.assertAlmostEqual(fold["fp32"]["accuracy"], saved[fold["subject"]], places=6)
            self.assertGreaterEqual(fold["agreement"], 0.0)
        efficiency.quantised_accuracy([(str(self.cfg_path), 0)], self.root / "runs", out)  # resumes, no error

    def test_classical_pipeline_timing(self):
        rng = np.random.default_rng(3)
        tables = [subject_features(synthetic_subject(rng), s, 60, 5) for s in ("S2", "S3", "S10")]
        table_path = self.root / "features.csv.gz"
        pd.concat(tables, ignore_index=True).to_csv(table_path, index=False)
        df = efficiency.classical_pipeline({
            "raw_dir": str(self.root / "WESAD"), "subject": "S2", "window_sec": 60, "warmup": 1, "runs": 2,
            "features_table": str(table_path), "normalisation": "rest_minutes:2", "task": "four_class",
            "feature_sets": ["wrist_physio"], "classifiers": ["LDA"],
        })
        self.assertEqual(set(df.stage), {"features", "classifier"})
        self.assertEqual(set(df[df.stage == "features"].signals), {"chest + wrist", "wrist only"})
        self.assertGreater(df[df.stage == "classifier"].size_kb.iloc[0], 0)


class StatsTest(unittest.TestCase):
    def test_holm(self):
        np.testing.assert_allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_bootstrap_interval_contains_mean(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.8, 0.05, size=15)
        lo, hi = stats.bootstrap_ci(values, 2000, rng)
        self.assertLess(lo, values.mean())
        self.assertGreater(hi, values.mean())

    def test_paired_test_detects_a_consistent_gain(self):
        rng = np.random.default_rng(1)
        subjects = [f"S{i}" for i in range(2, 17)]
        a = pd.Series(rng.normal(0.7, 0.05, 15), index=subjects)
        res = stats.paired_test(a, a + 0.05 + rng.normal(0, 0.005, 15), 2000, rng)
        self.assertLess(res["wilcoxon_p"], 0.001)
        self.assertGreater(res["diff_ci_low"], 0)
        self.assertAlmostEqual(res["rank_biserial"], 1.0)

    def test_self_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "S2_quest.csv"
            path.write_text(QUEST)
            sam = stats.read_self_reports(path)
        self.assertEqual(list(sam.index), ["baseline", "stress", "meditation 1", "amusement", "meditation 2"])
        self.assertEqual(sam.loc["amusement", "valence"], 8)
        self.assertEqual(sam.loc["stress", "arousal"], 7)

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subjects = [f"S{i}" for i in range(2, 9)]
            rng = np.random.default_rng(2)
            for seed in (1, 2):
                run = root / "runs" / f"model_seed{seed}"
                run.mkdir(parents=True)
                for s in subjects:
                    cm = rng.integers(1, 20, size=(4, 4)).tolist()
                    acc = float(np.trace(cm) / np.sum(cm))
                    (run / f"fold_{s}.json").write_text(json.dumps(
                        {"subject": s, "accuracy": acc, "f1_macro": acc - 0.1, "confusion_matrix": cm,
                         "class_names": stats.FOUR_CLASS}))
            per_subject = {s: {"accuracy": 0.9, "f1_macro": 0.8,
                               "confusion_matrix": rng.integers(1, 20, size=(4, 4)).tolist()} for s in subjects}
            (root / "classical.json").write_text(json.dumps({"per_subject": per_subject}))
            for s in subjects:
                (root / "WESAD" / s).mkdir(parents=True)
                (root / "WESAD" / s / f"{s}_quest.csv").write_text(QUEST)
            cfg = root / "stats.toml"
            cfg.write_text(f"""
[run]
out_dir = "{(root / 'out').as_posix()}"
bootstrap = 500
seeds = [1, 2]
[sources]
"deep" = {{ kind = "deep", runs = "{(root / 'runs').as_posix()}", config = "model" }}
"classical" = {{ kind = "classical", file = "{(root / 'classical.json').as_posix()}" }}
[[families]]
name = "deep vs classical"
labels = ["deep", "classical"]
[amusement]
raw_dir = "{(root / 'WESAD').as_posix()}"
labels = ["deep", "classical"]
""")
            stats.run(cfg)
            summary = pd.read_csv(root / "out" / "summary.csv")
            self.assertEqual(set(summary.label), {"deep", "classical"})
            self.assertTrue((summary.ci_low <= summary["mean"]).all())
            comp = pd.read_csv(root / "out" / "comparisons.csv")
            self.assertEqual(len(comp), 1)
            self.assertTrue((root / "out" / "report.md").exists())
            self.assertTrue((root / "out" / "amusement_confusion.png").exists())


if __name__ == "__main__":
    unittest.main()
