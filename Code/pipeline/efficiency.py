"""M4: measured efficiency of every model, on the CPU.

Stages (run separately or together, see ``run_pipeline.py efficiency``):

  static     parameters, multiply-accumulates per prediction, file size in
             float32 / float16 / int8.
  latency    time to classify one 60 s window, batch size 1, 1 and N CPU
             threads, float32 and int8; median and 95th percentile.
             Needs an otherwise idle machine.
  classical  the hand-crafted feature pipeline: time to compute features for
             one window from raw signals, plus classifier time and file size.
  quantised  accuracy after int8 conversion, re-testing every saved LOSO
             checkpoint on its own held-out subject (float32 and int8 on the
             same windows). Shardable across processes.
  plot       accuracy against latency and size, from the other stages.

Int8 here is PyTorch dynamic quantisation: weights of linear and recurrent
layers become int8; convolutions stay float32. Energy is not measured.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import platform
import time
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from threadpoolctl import threadpool_limits

from .config import CODE_DIR, TASKS, load_config
from .model import build_model, count_parameters
from .preprocess import preprocessed_files
from .train import channel_mean_std, fold_metrics, load_normalised, subject_of


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else CODE_DIR / p


def _model_name(config_path: str) -> str:
    """Short architecture name from a config file name, e.g. bigru_attn."""
    stem = Path(config_path).stem
    for name in ("bigru_attn", "bilstm_attn", "cnn_lstm", "bigru_only", "tiny_dscnn", "cnn"):
        if f"_{name}_" in f"_{stem}_":
            return name
    return stem


# --------------------------------------------------------------------------- static cost

def count_macs(model: nn.Module, x: torch.Tensor) -> int:
    """Multiply-accumulates of one forward pass, counted per layer type.

    Covers convolutions, linear layers, GRU/LSTM and multi-head attention,
    which hold essentially all of the arithmetic; activations, normalisation
    and pooling are not counted.
    """
    total = 0
    hooks = []

    def conv(m: nn.Conv1d, inp, out):
        nonlocal total
        total += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0]

    def linear(m: nn.Linear, inp, out):
        nonlocal total
        total += out.numel() // m.out_features * m.in_features * m.out_features

    def rnn(m: nn.RNNBase, inp, out):
        nonlocal total
        gates = 3 if isinstance(m, nn.GRU) else 4
        batch, steps = inp[0].shape[0], inp[0].shape[1]
        dirs = 2 if m.bidirectional else 1
        for layer in range(m.num_layers):
            in_size = m.input_size if layer == 0 else m.hidden_size * dirs
            total += batch * steps * dirs * gates * (in_size * m.hidden_size + m.hidden_size * m.hidden_size)

    def attention(m: nn.MultiheadAttention, inp, out):
        nonlocal total
        q = inp[0]
        batch, steps, dim = q.shape
        total += batch * (3 * steps * dim * dim + 2 * steps * steps * dim + steps * dim * dim)

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv))
        elif isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(attention))
        elif isinstance(module, nn.Linear) and not _inside_attention(model, module):
            hooks.append(module.register_forward_hook(linear))
        elif isinstance(module, (nn.GRU, nn.LSTM)):
            hooks.append(module.register_forward_hook(rnn))
    try:
        with torch.inference_mode():
            model(x)
    finally:
        for h in hooks:
            h.remove()
    return int(total)


def _inside_attention(model: nn.Module, module: nn.Module) -> bool:
    return any(module is m.out_proj for m in model.modules() if isinstance(m, nn.MultiheadAttention))


def quantise(model: nn.Module) -> nn.Module:
    """Int8 dynamic quantisation of linear and recurrent layers."""
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear, nn.GRU, nn.LSTM}, dtype=torch.qint8)


def state_size_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def build_from_config(config_path: str) -> tuple[nn.Module, int, int, str]:
    cfg = load_config(_resolve(config_path))
    n_channels = len(cfg.preprocess.channels)
    steps = cfg.preprocess.window_sec * cfg.preprocess.target_rate
    model = build_model(n_channels, cfg.model, len(TASKS[cfg.train.task]["classes"])).eval()
    return model, n_channels, steps, cfg.train.task


def static_costs(config_paths: list[str]) -> pd.DataFrame:
    rows = []
    for path in config_paths:
        model, n_channels, steps, task = build_from_config(path)
        x = torch.randn(1, steps, n_channels)
        half = build_from_config(path)[0].half()
        rows.append({
            "model": _model_name(path), "config": Path(path).stem, "task": task,
            "parameters": count_parameters(model), "macs_per_window": count_macs(model, x),
            "size_fp32_kb": state_size_bytes(model) / 1024,
            "size_fp16_kb": state_size_bytes(half) / 1024,
            "size_int8_kb": state_size_bytes(quantise(model)) / 1024,
            "input": f"{steps} x {n_channels}",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- latency

def time_call(fn, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) / 1e6)
    times = np.array(times)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "mean_ms": float(times.mean())}


def latency(config_paths: list[str], threads: list[int], warmup: int, runs: int, stride_sec: float) -> pd.DataFrame:
    rows = []
    before = torch.get_num_threads()
    try:
        for path in config_paths:
            model, n_channels, steps, task = build_from_config(path)
            x = torch.randn(1, steps, n_channels)
            for variant, net in (("fp32", model), ("int8", quantise(model))):
                for n in threads:
                    torch.set_num_threads(n)
                    with torch.inference_mode():
                        t = time_call(lambda: net(x), warmup, runs)
                    rows.append({"model": _model_name(path), "config": Path(path).stem, "task": task,
                                 "precision": variant, "threads": n, **t,
                                 "duty_cycle": t["median_ms"] / (stride_sec * 1000)})
                    print(f"  {_model_name(path):<11s} {variant} {n} thread(s): median {t['median_ms']:.2f} ms")
    finally:
        torch.set_num_threads(before)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- classical pipeline

def _crop(data: dict, t0: float, seconds: float) -> dict:
    """A copy of a raw recording holding only [t0, t0 + seconds)."""
    from .wesad import CHEST_RATE, WRIST_RATES

    out = {"subject": data.get("subject"), "signal": {}}
    a, b = int(t0 * CHEST_RATE), int((t0 + seconds) * CHEST_RATE)
    out["label"] = np.asarray(data["label"])[a:b]
    if "chest" in data["signal"]:
        out["signal"]["chest"] = {k: np.asarray(v)[a:b] for k, v in data["signal"]["chest"].items()}
    out["signal"]["wrist"] = {
        k: np.asarray(v)[int(t0 * WRIST_RATES[k]):int((t0 + seconds) * WRIST_RATES[k])]
        for k, v in data["signal"]["wrist"].items()
    }
    return out


def classical_pipeline(cfg: dict) -> pd.DataFrame:
    """Per-window cost of the hand-crafted route: features from raw signals, then a classifier."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .baselines import FEATURE_SETS, TASKS as BASE_TASKS, make_classifier, standardise_per_subject
    from .datasets import load_subject, subject_files
    from .features import subject_features

    raw_dir = _resolve(cfg["raw_dir"])
    files = {p.stem: p for p in subject_files("wesad", raw_dir)}
    data = load_subject("wesad", files[cfg.get("subject", "S2")])
    lab = np.asarray(data["label"])
    t0 = float(np.flatnonzero(lab == 1)[0] / 700 + 30)  # inside the baseline block
    window = float(cfg.get("window_sec", 60))
    one_window = _crop(data, t0, window)
    wrist_only = {**one_window, "signal": {"wrist": one_window["signal"]["wrist"]}}
    warmup, runs = int(cfg.get("warmup", 3)), int(cfg.get("runs", 30))

    rows = []
    with threadpool_limits(1):
        for name, rec in (("chest + wrist", one_window), ("wrist only", wrist_only)):
            t = time_call(lambda: subject_features(rec, "S", window, window), warmup, runs)
            rows.append({"stage": "features", "signals": name, **t})
            print(f"  features ({name}): median {t['median_ms']:.1f} ms per {window:g} s window")

        table = pd.read_csv(_resolve(cfg["features_table"]))
        table = standardise_per_subject(table, cfg.get("normalisation", "rest_minutes:5"), window, 5)
        spec = BASE_TASKS[cfg.get("task", "four_class")]
        table = table[table["label"].isin(list(spec["map"]))]
        y = table["label"].map(spec["map"]).to_numpy()
        for feature_set in cfg.get("feature_sets", ["all_physio", "wrist_physio"]):
            columns = [c for c in table.columns if c.startswith(tuple(FEATURE_SETS[feature_set]))]
            X = table[columns].to_numpy(dtype=np.float64)
            for clf in cfg.get("classifiers", ["RF", "AB", "LDA"]):
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), make_classifier(clf, 42))
                model.fit(X, y)
                row = X[:1]
                t = time_call(lambda: model.predict(row), 20, 300)
                rows.append({"stage": "classifier", "signals": feature_set, "classifier": clf,
                             "size_kb": len(pickle.dumps(model)) / 1024, **t})
                print(f"  {clf} on {feature_set}: median {t['median_ms']:.2f} ms, {rows[-1]['size_kb']:.0f} KB")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- quantised accuracy

def quantised_accuracy(items: list[tuple[str, int]], runs_dir: Path, out_dir: Path, threads: int | None = None) -> None:
    """Float32 and int8 accuracy of every saved fold model, on the same test windows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if threads:
        torch.set_num_threads(threads)  # keep parallel shards from oversubscribing the CPU
    device = torch.device("cpu")
    for config_path, seed in items:
        cfg = load_config(_resolve(config_path)).with_seed(seed)
        out = out_dir / f"{cfg.name}.json"
        if out.exists():
            print(f"  {cfg.name}: already done")
            continue
        run_dir = runs_dir / cfg.name
        files = preprocessed_files(cfg)
        class_names = TASKS[cfg.train.task]["classes"]
        folds = []
        for test_file in files:
            subject = subject_of(test_file)
            ckpt = run_dir / "checkpoints" / f"best_{subject}.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing checkpoint {ckpt}; this run's checkpoints are needed")
            mean, std = channel_mean_std([f for f in files if f != test_file])
            X, y, _ = load_normalised([test_file], mean, std, cfg.train.task)
            model = build_model(X.shape[2], cfg.model, len(class_names))
            model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
            model.eval()
            with torch.inference_mode():
                fp32 = torch.cat([model(xb).argmax(1) for xb in torch.split(X, 64)]).numpy()
                q = quantise(model)
                int8 = torch.cat([q(xb).argmax(1) for xb in torch.split(X, 64)]).numpy()
            yt = y.numpy()
            folds.append({"subject": subject,
                          "fp32": {k: v for k, v in fold_metrics(yt, fp32, class_names).items() if k != "classification_report"},
                          "int8": {k: v for k, v in fold_metrics(yt, int8, class_names).items() if k != "classification_report"},
                          "agreement": float((fp32 == int8).mean())})
        result = {"config": cfg.name, "folds": folds,
                  "accuracy_fp32": float(np.mean([f["fp32"]["accuracy"] for f in folds])),
                  "accuracy_int8": float(np.mean([f["int8"]["accuracy"] for f in folds])),
                  "f1_fp32": float(np.mean([f["fp32"]["f1_macro"] for f in folds])),
                  "f1_int8": float(np.mean([f["int8"]["f1_macro"] for f in folds])),
                  "agreement": float(np.mean([f["agreement"] for f in folds]))}
        out.write_text(json.dumps(result, indent=2))
        print(f"  {cfg.name}: acc fp32 {result['accuracy_fp32']:.4f} -> int8 {result['accuracy_int8']:.4f}, "
              f"agreement {result['agreement']:.4f}")


def summarise_quantised(out_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(out_dir.glob("*.json")):
        r = json.loads(path.read_text())
        rows.append({"run": r["config"], "config": r["config"].rsplit("_seed", 1)[0],
                     **{k: r[k] for k in ("accuracy_fp32", "accuracy_int8", "f1_fp32", "f1_int8", "agreement")}})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby("config")[["accuracy_fp32", "accuracy_int8", "f1_fp32", "f1_int8", "agreement"]].agg(
        ["mean", "std"]).round(4)


# --------------------------------------------------------------------------- plot

def plot(out_dir: Path, accuracy_sources: dict) -> None:
    """Accuracy against single-thread latency, deep and classical, one panel per task."""
    lat = pd.read_csv(out_dir / "latency.csv")
    stat = pd.read_csv(out_dir / "static.csv")
    classical = pd.read_csv(out_dir / "classical.csv")
    deep_acc = pd.read_csv(_resolve(accuracy_sources["deep_summary"]))
    classical_acc = pd.read_csv(_resolve(accuracy_sources["classical_summary"]))

    feat = classical[classical.stage == "features"].set_index("signals")["median_ms"]
    rows = []
    for task in accuracy_sources.get("tasks", ["binary", "four_class"]):
        for model in lat.model.unique():
            cfg = f"e_{task}_{model}_rest5_w60"
            acc = deep_acc[deep_acc.config == cfg]
            ms = lat[(lat.model == model) & (lat.precision == "fp32") & (lat.threads == 1)]["median_ms"]
            size = stat[stat.model == model]["size_fp32_kb"]
            if acc.empty or ms.empty:
                continue
            rows.append({"task": task, "kind": "deep", "name": model, "accuracy": acc.accuracy_mean.iloc[0],
                         "latency_ms": float(ms.iloc[0]), "size_kb": float(size.iloc[0])})
        for _, r in classical[classical.stage == "classifier"].iterrows():
            m = classical_acc[(classical_acc.task == task) & (classical_acc.feature_set == r.signals)
                              & (classical_acc.classifier == r.classifier)
                              & (classical_acc.normalisation == "rest_minutes:5")]
            if m.empty:
                continue
            signals = "chest + wrist" if r.signals == "all_physio" else "wrist only"
            rows.append({"task": task, "kind": "classical", "name": f"{r.classifier} ({r.signals})",
                         "accuracy": m.accuracy_mean.iloc[0],
                         "latency_ms": float(feat.get(signals, np.nan) + r.median_ms), "size_kb": r.size_kb})
    points = pd.DataFrame(rows)
    points.to_csv(out_dir / "accuracy_vs_cost.csv", index=False)

    tasks = list(points.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.5 * len(tasks), 4.8), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = points[points.task == task]
        for kind, marker, color in (("deep", "o", "#3b6fb6"), ("classical", "s", "#c0504d")):
            k = sub[sub.kind == kind]
            ax.scatter(k.latency_ms, k.accuracy, marker=marker, color=color, label=kind, s=40)
            for _, r in k.iterrows():
                ax.annotate(r["name"], (r.latency_ms, r.accuracy), fontsize=7, xytext=(4, 2), textcoords="offset points")
        ax.set_xscale("log")
        ax.set(title=f"WESAD {task.replace('_', '-')}, deployable normalisation",
               xlabel="Latency per 60 s window, 1 CPU thread (ms, log scale)", ylabel="LOSO accuracy")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_cost.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- driver

def load_efficiency_config(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def quantised_items(cfg: dict) -> list[tuple[str, int]]:
    q = cfg["quantised"]
    return [(c, s) for c in q["configs"] for s in q.get("seeds", [42, 43, 44, 45, 46])]


def run(config_path: str | Path, stages: list[str], shard: str | None = None) -> None:
    cfg = load_efficiency_config(config_path)
    out_dir = _resolve(cfg["run"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["run"].get("seed", 42))
    (out_dir / "machine.json").write_text(json.dumps({
        "processor": platform.processor(), "machine": platform.machine(), "python": platform.python_version(),
        "torch": torch.__version__, "logical_cpus": os.cpu_count(),
        "torch_default_threads": torch.get_num_threads(),
    }, indent=2))

    if "static" in stages:
        df = static_costs(cfg["models"]["configs"])
        df.to_csv(out_dir / "static.csv", index=False)
        print(df.to_string(index=False))
    if "latency" in stages:
        lc = cfg["latency"]
        print("Latency (keep the machine otherwise idle):")
        latency(cfg["models"]["configs"], lc.get("threads", [1, 4]), lc.get("warmup", 50), lc.get("runs", 300),
                lc.get("stride_sec", 10)).to_csv(out_dir / "latency.csv", index=False)
    if "classical" in stages:
        print("Classical pipeline:")
        classical_pipeline(cfg["classical"]).to_csv(out_dir / "classical.csv", index=False)
    if "quantised" in stages:
        items = quantised_items(cfg)
        if shard:
            index, count = (int(v) for v in shard.split("/"))
            items = items[index::count]
        quantised_accuracy(items, _resolve(cfg["quantised"]["runs_dir"]), out_dir / "quantised",
                           cfg["quantised"].get("threads"))
        summary = summarise_quantised(out_dir / "quantised")
        if not summary.empty:
            summary.to_csv(out_dir / "quantised_summary.csv")
    if "plot" in stages:
        plot(out_dir, cfg["plot"])
        print(f"Plot written to {out_dir / 'accuracy_vs_cost.png'}")
