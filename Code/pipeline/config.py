"""Run configuration, loaded from a TOML file.

Every value that affects a result lives in the config file, so the settings a
run was produced with can never drift from the settings written in the paper.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

# Relative paths in a config file are resolved against the Code/ directory.
CODE_DIR = Path(__file__).resolve().parents[1]

# WESAD label ids -> class index. Ids 0 (undefined/transient) and 5-7 (ignore)
# are not in this map; how they are handled is up to the preprocessing step.
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}
CLASS_NAMES = ["baseline", "stress", "amusement", "meditation"]
NUM_CLASSES = len(CLASS_NAMES)

CHANNEL_NAMES = [
    "wr_acc_mag", "wr_eda", "wr_temp",
    "ch_ecg", "ch_resp", "ch_acc_mag", "ch_eda", "ch_temp",
]

# Channels the clean pipeline can produce; a config may select a subset.
CLEAN_CHANNELS = [
    "ch_ecg", "ch_eda", "ch_emg", "ch_resp", "ch_temp", "ch_acc_x", "ch_acc_y", "ch_acc_z",
    "wr_bvp", "wr_eda", "wr_temp", "wr_acc_x", "wr_acc_y", "wr_acc_z",
]

WRIST_CHANNELS = [c for c in CLEAN_CHANNELS if c.startswith("wr_")]
NORMALISATION_SOURCES = ("recording", "first_minutes", "rest_minutes")
DATASETS = ("wesad", "stress_predict")

BALANCING_MODES = ("none", "weighted_loss", "weighted_sampler")
LR_SCHEDULERS = ("none", "plateau")


@dataclass(frozen=True)
class PreprocessConfig:
    target_rate: int
    window_sec: int
    overlap: float
    majority_threshold: float
    chest_rate: int = 700


@dataclass(frozen=True)
class CleanPreprocessConfig:
    """Settings for the corrected pipeline (``mode = "clean"`` in the config)."""

    target_rate: int
    window_sec: int
    stride_sec: float
    min_label_purity: float = 1.0
    subject_normalisation: bool = True
    channels: list[str] = field(default_factory=lambda: list(CLEAN_CHANNELS))
    mode: str = "clean"
    # Which part of a subject's recording the per-subject statistics come from:
    #   "recording"     the whole recording (needs it all in advance; not deployable)
    #   "first_minutes" the first normalisation_minutes of the recording
    #   "rest_minutes"  the first normalisation_minutes of the first rest/baseline block
    normalisation_source: str = "recording"
    normalisation_minutes: float = 5.0
    dataset: str = "wesad"
    # Stress-Predict only: drop the hyperventilation task instead of calling it stress.
    exclude_hyperventilation: bool = False

    def __post_init__(self):
        unknown = [c for c in self.channels if c not in CLEAN_CHANNELS]
        if unknown:
            raise ValueError(f"Unknown preprocess.channels {unknown}; choose from {CLEAN_CHANNELS}")
        if self.normalisation_source not in NORMALISATION_SOURCES:
            raise ValueError(f"preprocess.normalisation_source must be one of {NORMALISATION_SOURCES}")
        if self.dataset not in DATASETS:
            raise ValueError(f"preprocess.dataset must be one of {DATASETS}, got {self.dataset!r}")
        if self.dataset == "stress_predict" and any(not c.startswith("wr_") for c in self.channels):
            raise ValueError("Stress-Predict has wrist signals only; set preprocess.channels to wr_* channels")
        if self.exclude_hyperventilation and self.dataset != "stress_predict":
            raise ValueError("preprocess.exclude_hyperventilation applies to Stress-Predict only")


@dataclass(frozen=True)
class ModelConfig:
    """The original CNN -> BiGRU -> attention model, exactly as in the paper draft."""

    cnn_channels: int
    gru_hidden: int
    gru_layers: int
    attn_heads: int
    dropout: float


RNN_TYPES = ("gru", "lstm", "none")


@dataclass(frozen=True)
class SeqModelConfig:
    """Configurable model family for M3 (``arch = "seq"`` in the config).

    A stack of conv blocks, each optionally followed by max-pooling by the
    matching ``downsample`` factor, then an optional recurrent layer, optional
    self-attention, mean pooling over time and a small classifier. With no
    conv blocks, ``downsample`` may hold one factor for average-pooling the
    raw input instead.
    """

    conv_channels: list[int]
    downsample: list[int]
    kernel_size: int = 5
    separable: bool = False
    rnn: str = "gru"
    bidirectional: bool = True
    rnn_hidden: int = 128
    rnn_layers: int = 2
    attention: bool = True
    attn_heads: int = 4
    dropout: float = 0.3
    arch: str = "seq"

    def __post_init__(self):
        if self.rnn not in RNN_TYPES:
            raise ValueError(f"model.rnn must be one of {RNN_TYPES}, got {self.rnn!r}")
        if self.conv_channels and len(self.downsample) != len(self.conv_channels):
            raise ValueError("model.downsample needs one factor per entry of model.conv_channels")
        if not self.conv_channels and len(self.downsample) > 1:
            raise ValueError("without conv blocks, model.downsample may hold at most one factor")
        if not self.conv_channels and self.rnn == "none":
            raise ValueError("a model needs conv blocks, a recurrent layer, or both")


# Tasks map the four condition classes (baseline, stress, amusement,
# meditation = 0..3) to the task's classes; unmapped classes are dropped.
# Binary follows the dataset paper: stress against baseline + amusement.
TASKS = {
    "four_class": {"classes": ["baseline", "stress", "amusement", "meditation"], "map": {0: 0, 1: 1, 2: 2, 3: 3}},
    "three_class": {"classes": ["baseline", "stress", "amusement"], "map": {0: 0, 1: 1, 2: 2}},
    "binary": {"classes": ["non-stress", "stress"], "map": {0: 0, 2: 0, 1: 1}},
}
VALIDATION_MODES = ("random_windows", "subjects")


@dataclass(frozen=True)
class TrainConfig:
    balancing: str
    batch_size: int
    epochs: int
    lr: float
    patience: int
    grad_clip: float
    val_fraction: float
    lr_scheduler: str = "none"
    min_delta: float = 0.0
    test_batch_size: int | None = None
    task: str = "four_class"
    # "random_windows" draws val_fraction of the training windows (the legacy
    # behaviour; overlapping windows leak into validation). "subjects" holds
    # out val_subjects whole training subjects instead.
    validation: str = "random_windows"
    val_subjects: int = 2
    # Per-user calibration: after training, fine-tune on the first N minutes of
    # each condition of the test subject (one entry per N to try) and test on
    # the rest of that subject's recording.
    calibration_minutes: list[float] = field(default_factory=list)
    calibration_epochs: int = 10
    calibration_lr: float = 1e-4
    # "head" fine-tunes only the final classifier layers (the usual few-shot
    # choice; a pilot that tuned every layer on a few windows overfitted),
    # "all" fine-tunes the whole network.
    calibration_layers: str = "head"

    def __post_init__(self):
        if self.balancing not in BALANCING_MODES:
            raise ValueError(f"train.balancing must be one of {BALANCING_MODES}, got {self.balancing!r}")
        if self.lr_scheduler not in LR_SCHEDULERS:
            raise ValueError(f"train.lr_scheduler must be one of {LR_SCHEDULERS}, got {self.lr_scheduler!r}")
        if self.task not in TASKS:
            raise ValueError(f"train.task must be one of {list(TASKS)}, got {self.task!r}")
        if self.calibration_layers not in ("head", "all"):
            raise ValueError(f"train.calibration_layers must be 'head' or 'all', got {self.calibration_layers!r}")
        if self.validation not in VALIDATION_MODES:
            raise ValueError(f"train.validation must be one of {VALIDATION_MODES}, got {self.validation!r}")


# Options added after runs were already saved. A saved run without them was
# made with these values, which reproduce the earlier behaviour.
ADDED_TRAIN_DEFAULTS = {
    "task": "four_class", "validation": "random_windows", "val_subjects": 2,
    "calibration_minutes": [], "calibration_epochs": 10, "calibration_lr": 1e-4, "calibration_layers": "head",
}
ADDED_CLEAN_DEFAULTS = {
    "normalisation_source": "recording", "normalisation_minutes": 5.0,
    "dataset": "wesad", "exclude_hyperventilation": False,
}


def upgrade_saved_config(saved: dict) -> dict:
    """Fill options added later into a config dict saved before they existed.

    Works on a whole run config (``preprocess``/``train`` keys) or on the
    preprocessing settings alone (``mode`` key).
    """
    saved = dict(saved)
    if "preprocess" in saved:
        saved["preprocess"] = upgrade_saved_config(saved["preprocess"])
        saved["train"] = {**ADDED_TRAIN_DEFAULTS, **saved["train"]}
    elif saved.get("mode") == "clean":
        saved = {**ADDED_CLEAN_DEFAULTS, **saved}
    return saved


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    raw_dir: Path
    data_dir: Path
    out_dir: Path
    preprocess: PreprocessConfig | CleanPreprocessConfig
    model: ModelConfig | SeqModelConfig
    train: TrainConfig

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("raw_dir", "data_dir", "out_dir"):
            d[key] = str(d[key])
        return d

    def with_seed(self, seed: int) -> "RunConfig":
        """The same run under another seed, with its own name and output folder."""
        return replace(
            self, seed=seed, name=f"{self.name}_seed{seed}",
            out_dir=self.out_dir.parent / f"{self.out_dir.name}_seed{seed}",
        )


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else CODE_DIR / p


def load_config(path: str | Path) -> RunConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    run, paths = raw["run"], raw["paths"]
    pre = dict(raw["preprocess"])
    mode = pre.pop("mode", "legacy")
    if mode not in ("legacy", "clean"):
        raise ValueError(f"preprocess.mode must be 'legacy' or 'clean', got {mode!r}")
    model = dict(raw["model"])
    arch = model.pop("arch", "legacy")
    if arch not in ("legacy", "seq"):
        raise ValueError(f"model.arch must be 'legacy' or 'seq', got {arch!r}")
    return RunConfig(
        name=run["name"],
        seed=int(run["seed"]),
        raw_dir=_resolve(paths["raw_dir"]),
        data_dir=_resolve(paths["data_dir"]),
        out_dir=_resolve(paths["out_dir"]),
        preprocess=CleanPreprocessConfig(**pre) if mode == "clean" else PreprocessConfig(**pre),
        model=SeqModelConfig(**model) if arch == "seq" else ModelConfig(**model),
        train=TrainConfig(**raw["train"]),
    )


def load_cross_config(path: str | Path) -> tuple[RunConfig, RunConfig]:
    """A cross-dataset run: ``[cross]`` names a training config and a test config.

    Returns (training config renamed to this run, test config). Model and
    training settings come from the training config; the test config only
    supplies its dataset and preprocessing.
    """
    with open(path, "rb") as fh:
        cross = tomllib.load(fh)["cross"]
    train = load_config(_resolve(cross["train_config"]))
    test = load_config(_resolve(cross["test_config"]))
    return replace(train, name=cross["name"], out_dir=_resolve(cross["out_dir"])), test
