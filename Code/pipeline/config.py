"""Run configuration, loaded from a TOML file.

Every value that affects a result lives in the config file, so the settings a
run was produced with can never drift from the settings written in the paper.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
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

    def __post_init__(self):
        unknown = [c for c in self.channels if c not in CLEAN_CHANNELS]
        if unknown:
            raise ValueError(f"Unknown preprocess.channels {unknown}; choose from {CLEAN_CHANNELS}")


@dataclass(frozen=True)
class ModelConfig:
    cnn_channels: int
    gru_hidden: int
    gru_layers: int
    attn_heads: int
    dropout: float


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

    def __post_init__(self):
        if self.balancing not in BALANCING_MODES:
            raise ValueError(f"train.balancing must be one of {BALANCING_MODES}, got {self.balancing!r}")
        if self.lr_scheduler not in LR_SCHEDULERS:
            raise ValueError(f"train.lr_scheduler must be one of {LR_SCHEDULERS}, got {self.lr_scheduler!r}")


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    raw_dir: Path
    data_dir: Path
    out_dir: Path
    preprocess: PreprocessConfig | CleanPreprocessConfig
    model: ModelConfig
    train: TrainConfig

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("raw_dir", "data_dir", "out_dir"):
            d[key] = str(d[key])
        return d


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
    return RunConfig(
        name=run["name"],
        seed=int(run["seed"]),
        raw_dir=_resolve(paths["raw_dir"]),
        data_dir=_resolve(paths["data_dir"]),
        out_dir=_resolve(paths["out_dir"]),
        preprocess=CleanPreprocessConfig(**pre) if mode == "clean" else PreprocessConfig(**pre),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
    )
