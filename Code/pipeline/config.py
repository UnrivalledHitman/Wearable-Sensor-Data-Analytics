"""Run configuration, loaded from a TOML file.

Every value that affects a result lives in the config file, so the settings a
run was produced with can never drift from the settings written in the paper.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
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
    preprocess: PreprocessConfig
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
    return RunConfig(
        name=run["name"],
        seed=int(run["seed"]),
        raw_dir=_resolve(paths["raw_dir"]),
        data_dir=_resolve(paths["data_dir"]),
        out_dir=_resolve(paths["out_dir"]),
        preprocess=PreprocessConfig(**raw["preprocess"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
    )
