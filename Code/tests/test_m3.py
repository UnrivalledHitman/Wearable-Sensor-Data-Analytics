"""Tests for the M3 additions: SeqNet models, tasks, subject validation, seeds."""

import json
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from pipeline.config import TASKS, SeqModelConfig, load_config
from pipeline.model import SeqNet, count_parameters
from pipeline.preprocess import preprocess_all
from pipeline.report import report
from pipeline.train import _can_resume, load_normalised, run_loso, split_validation, write_run_manifest
from tests.test_smoke import synthetic_subject

CODE_DIR = Path(__file__).resolve().parents[1]


def seq_config(root: Path, name: str, task: str, model: str, validation: str = "subjects") -> Path:
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
{model}

[train]
task = "{task}"
balancing = "weighted_sampler"
batch_size = 16
epochs = 2
lr = 1e-3
patience = 3
grad_clip = 5.0
val_fraction = 0.1
validation = "{validation}"
val_subjects = 1
""")
    return path


BIGRU = 'conv_channels = [8, 8]\ndownsample = [2, 2]\nrnn = "gru"\nrnn_hidden = 8\nattn_heads = 2'


class SeqNetTest(unittest.TestCase):
    def test_downsampling_shortens_the_sequence(self):
        cfg = SeqModelConfig(conv_channels=[8, 8, 8, 8], downsample=[2, 2, 2, 2], rnn_hidden=8, attn_heads=2)
        model = SeqNet(14, cfg, num_classes=3)
        x = torch.randn(2, 320, 14)
        self.assertEqual(model.front(x.permute(0, 2, 1)).shape[-1], 20)
        self.assertEqual(model(x).shape, (2, 3))

    def test_every_variant_runs(self):
        variants = {
            "lstm": SeqModelConfig([8], [4], rnn="lstm", rnn_hidden=8, attn_heads=2),
            "unidirectional": SeqModelConfig([8], [4], bidirectional=False, attention=False, rnn_hidden=8),
            "cnn only": SeqModelConfig([8, 8], [2, 2], rnn="none", attention=False),
            "rnn only": SeqModelConfig([], [4], rnn_hidden=8, attn_heads=2),
            "separable": SeqModelConfig([8, 8], [2, 2], separable=True, rnn="none", attention=False),
        }
        for name, cfg in variants.items():
            with self.subTest(name):
                self.assertEqual(SeqNet(6, cfg, num_classes=4)(torch.randn(3, 64, 6)).shape, (3, 4))

    def test_lstm_costs_more_than_gru(self):
        gru = SeqNet(14, SeqModelConfig([64], [2]), 4)
        lstm = SeqNet(14, SeqModelConfig([64], [2], rnn="lstm"), 4)
        self.assertGreater(count_parameters(lstm), count_parameters(gru))

    def test_invalid_configs_are_rejected(self):
        with self.assertRaises(ValueError):
            SeqModelConfig([8, 8], [2])
        with self.assertRaises(ValueError):
            SeqModelConfig([], [], rnn="none")
        with self.assertRaises(ValueError):
            SeqModelConfig([8], [2], rnn="transformer")

    def test_generated_m3_configs_load_and_build(self):
        configs = sorted((CODE_DIR / "configs" / "m3").glob("*.toml"))
        self.assertEqual(len(configs), 32)
        for path in configs:
            cfg = load_config(path)
            self.assertIsInstance(cfg.model, SeqModelConfig)
            self.assertEqual(cfg.train.validation, "subjects")
            n_classes = len(TASKS[cfg.train.task]["classes"])
            SeqNet(len(cfg.preprocess.channels), cfg.model, n_classes)


class TrainingOptionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        (cls.root / "WESAD").mkdir()
        rng = np.random.default_rng(0)
        for subject in ("S2", "S3", "S10"):
            with open(cls.root / "WESAD" / f"{subject}.pkl", "wb") as fh:
                pickle.dump(synthetic_subject(rng), fh)
        preprocess_all(load_config(seq_config(cls.root, "prep", "four_class", BIGRU)))
        cls.files = sorted((cls.root / "data").glob("*_combined.npz"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _load(self, task):
        n_ch = np.load(self.files[0])["X"].shape[-1]
        return load_normalised(self.files, np.zeros(n_ch, np.float32), np.ones(n_ch, np.float32), task)

    def test_task_mapping(self):
        _, y4, _ = self._load("four_class")
        _, y3, _ = self._load("three_class")
        _, y2, g2 = self._load("binary")
        counts = np.bincount(y4.numpy(), minlength=4)
        self.assertEqual(len(y3), counts[:3].sum())            # meditation dropped
        self.assertEqual(len(y2), counts[:3].sum())            # meditation dropped
        self.assertEqual(int((y2 == 1).sum()), counts[1])      # stress stays stress
        self.assertEqual(int((y2 == 0).sum()), counts[0] + counts[2])  # baseline + amusement
        self.assertEqual(len(g2), len(y2))

    def test_subject_validation_holds_out_whole_subjects(self):
        cfg = load_config(seq_config(self.root, "v", "four_class", BIGRU))
        cfg = replace(cfg, train=replace(cfg.train, val_subjects=1))
        _, y, groups = self._load("four_class")
        train_idx, val_idx = split_validation(cfg, len(y), groups, torch.Generator().manual_seed(0))
        val_groups, train_groups = set(groups[val_idx].tolist()), set(groups[train_idx].tolist())
        self.assertEqual(len(val_groups), 1)
        self.assertFalse(val_groups & train_groups)
        self.assertEqual(len(train_idx) + len(val_idx), len(y))

    def test_end_to_end_three_class_with_seeds(self):
        cfg = load_config(seq_config(self.root, "e2e", "three_class", BIGRU)).with_seed(7)
        self.assertEqual(cfg.name, "e2e_seed7")
        self.assertTrue(cfg.out_dir.name.endswith("e2e_seed7"))
        results = run_loso(cfg, device=torch.device("cpu"))
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r["class_names"], ["baseline", "stress", "amusement"])
            self.assertEqual(np.array(r["confusion_matrix"]).shape, (3, 3))
            self.assertGreater(r["seconds_per_epoch"], 0)
        report({cfg.name: cfg.out_dir}, cfg.out_dir / "report")
        summary = json.loads((cfg.out_dir / "report" / f"{cfg.name}_summary.json").read_text())
        self.assertEqual(summary["class_names"], ["baseline", "stress", "amusement"])
        self.assertIn("f1_amusement", summary["mean"])
        self.assertIn("seconds_per_epoch", summary["cost"])

    def test_resume_accepts_runs_saved_before_new_options(self):
        cfg = load_config(seq_config(self.root, "old", "four_class", BIGRU, validation="random_windows"))
        cfg = replace(cfg, train=replace(cfg.train, val_subjects=2))  # an old config could only have the default
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        write_run_manifest(cfg, torch.device("cpu"))
        manifest_path = cfg.out_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for key in ("task", "validation", "val_subjects"):
            del manifest["config"]["train"][key]  # as written before these options existed
        manifest_path.write_text(json.dumps(manifest))
        self.assertTrue(_can_resume(cfg))

        changed = replace(cfg, train=replace(cfg.train, task="binary"))
        self.assertFalse(_can_resume(changed))


if __name__ == "__main__":
    unittest.main()
