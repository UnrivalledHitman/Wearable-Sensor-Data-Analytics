"""Write configs/m6/stats.toml: which results M6 analyses and which comparisons it tests.

    python Code/configs/m6/make_config.py

Each comparison family is Holm-corrected on its own, so families are kept to
one question each.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = ("binary", "three_class", "four_class")
MAIN_DEEP = ("bigru_attn", "cnn", "tiny_dscnn")
ALL_DEEP = ("bigru_attn", "bilstm_attn", "cnn_lstm", "cnn", "bigru_only", "tiny_dscnn")
CLASSIFIERS = ("RF", "AB", "LDA")
NORM_TAGS = {"recording": "subjnorm", "none": "raw", "first5": "first5", "rest5": "rest5"}


def deep(runs: str, config: str, **extra) -> dict:
    return {"kind": "deep", "runs": runs, "config": config, **extra}


def classical(path: str, **extra) -> dict:
    return {"kind": "classical", "file": path, **extra}


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    return str(v)


def main() -> None:
    sources: dict[str, dict] = {}
    families: list[dict] = []

    # A. Headline: WESAD under deployable normalisation (5-minute rest).
    for task in TASKS:
        labels = []
        for m in MAIN_DEEP:
            label = f"wesad_{task}_deep_{m}_rest5"
            sources[label] = deep("runs/m5", f"e_{task}_{m}_rest5_w60")
            labels.append(label)
        for clf in CLASSIFIERS:
            label = f"wesad_{task}_classical_{clf}_rest5"
            sources[label] = classical(
                f"runs/m5/baselines_normalisation/combinations/w60_{task}_all_physio_rest5_{clf}.json")
            labels.append(label)
        families.append({"name": f"WESAD {task}, deployable: each model vs the proposed model", "labels": labels})

    # B. Effect of normalisation (reference: whole recording, as used in M1-M3).
    for task in ("binary", "four_class"):
        d = {"recording": deep("runs/m3", f"b_{task}_cnn_w60")}
        d.update({n: deep("runs/m5", f"e_{task}_cnn_{n}_w60") for n in ("none", "first5", "rest5")})
        labels = []
        for n, spec in d.items():
            label = f"wesad_{task}_deep_cnn_{n}"
            sources[label] = spec
            labels.append(label)
        families.append({"name": f"WESAD {task}, CNN: normalisation vs whole recording", "labels": labels})

        labels = []
        for n, tag in NORM_TAGS.items():
            label = f"wesad_{task}_classical_RF_{n}"
            sources[label] = classical(
                f"runs/m5/baselines_normalisation/combinations/w60_{task}_all_physio_{tag}_RF.json")
            labels.append(label)
        families.append({"name": f"WESAD {task}, RF: normalisation vs whole recording", "labels": labels})

    # C. M3 architecture comparison (whole-recording normalisation), four-class.
    labels = []
    for m in ALL_DEEP:
        label = f"wesad_four_class_deep_{m}_recording"
        sources[label] = deep("runs/m3", f"b_four_class_{m}_w60")
        labels.append(label)
    families.append({"name": "WESAD four-class (M3 setting): architectures vs the proposed model", "labels": labels})

    # D. Per-user calibration: before vs after on the same windows.
    for task in ("binary", "four_class"):
        for minutes in (1, 2, 4):
            pair = []
            for which in ("before", "after"):
                label = f"wesad_{task}_deep_cnn_rest5_cal{minutes}_{which}"
                sources[label] = deep("runs/m5", f"e_{task}_cnn_rest5_w60", calibration=minutes, which=which)
                pair.append(label)
            families.append({"name": f"Calibration, WESAD {task}, CNN, {minutes} min", "labels": pair})
            pair = []
            for which in ("before", "after"):
                label = f"wesad_{task}_classical_RF_rest5_cal{minutes}_{which}"
                sources[label] = classical(
                    f"runs/m5/baselines_calibration/combinations/w60_{task}_all_physio_rest5_RF_cal{minutes}.json",
                    which=which)
                pair.append(label)
            families.append({"name": f"Calibration, WESAD {task}, RF, {minutes} min", "labels": pair})
    for minutes in (1, 2, 4):
        pair = []
        for which in ("before", "after"):
            label = f"sp_deep_cnn_hv_cal{minutes}_{which}"
            sources[label] = deep("runs/m5", "h_binary_cnn_hv_w60", calibration=minutes, which=which)
            pair.append(label)
        families.append({"name": f"Calibration, Stress-Predict, CNN, {minutes} min", "labels": pair})

    # E. Stress-Predict LOSO (hyperventilation counted as stress).
    labels = []
    for m in ALL_DEEP:
        label = f"sp_deep_{m}_hv"
        sources[label] = deep("runs/m5", f"h_binary_{m}_hv_w60")
        labels.append(label)
    for clf in CLASSIFIERS:
        label = f"sp_classical_{clf}_hv"
        sources[label] = classical(
            f"runs/m5/baselines_stress_predict_hv/combinations/w60_binary_wrist_all_rest5_{clf}.json")
        labels.append(label)
    families.append({"name": "Stress-Predict: each model vs the proposed model", "labels": labels})

    # F. Hyperventilation counted as stress vs excluded.
    for name, hv, nohv in (
        ("CNN", deep("runs/m5", "h_binary_cnn_hv_w60"), deep("runs/m5", "h_binary_cnn_nohv_w60")),
        ("AdaBoost", classical("runs/m5/baselines_stress_predict_hv/combinations/w60_binary_wrist_all_rest5_AB.json"),
         classical("runs/m5/baselines_stress_predict_nohv/combinations/w60_binary_wrist_all_rest5_AB.json")),
    ):
        a, b = f"sp_{name}_hv_for_hvtest", f"sp_{name}_nohv_for_hvtest"
        sources[a], sources[b] = hv, nohv
        families.append({"name": f"Stress-Predict, {name}: hyperventilation excluded vs counted", "labels": [a, b]})

    # G. Cross-dataset (train WESAD, test Stress-Predict) vs training within Stress-Predict.
    for m in ALL_DEEP:
        a, b = f"sp_deep_{m}_hv", f"cross_deep_{m}_hv"
        sources[b] = deep("runs/m5", f"i_binary_{m}_wesad_to_sp_hv")
        families.append({"name": f"Cross-dataset vs within Stress-Predict, {m}", "labels": [a, b]})
    for clf in CLASSIFIERS:
        a, b = f"sp_classical_{clf}_hv", f"cross_classical_{clf}_hv"
        sources[b] = classical(
            f"runs/m5/baselines_cross_hv/combinations/cross_stress_predict_w60_binary_wrist_all_rest5_{clf}.json")
        families.append({"name": f"Cross-dataset vs within Stress-Predict, {clf}", "labels": [a, b]})

    lines = [
        "# Generated by make_config.py; edit that script, not this file.",
        "",
        "[run]",
        'out_dir = "runs/m6"',
        "seed = 42",
        "bootstrap = 10000",
        "seeds = [42, 43, 44, 45, 46]",
        "",
        "[sources]",
    ]
    for label, spec in sources.items():
        inner = ", ".join(f"{k} = {toml_value(v)}" for k, v in spec.items())
        lines.append(f'"{label}" = {{ {inner} }}')
    for fam in families:
        lines += ["", "[[families]]", f'name = "{fam["name"]}"', f"labels = {toml_value(fam['labels'])}",
                  'metric = "accuracy"', 'mode = "vs_first"']
    lines += [
        "", "[amusement]", 'raw_dir = "WESAD"',
        "labels = " + toml_value(["wesad_four_class_deep_bigru_attn_recording", "wesad_four_class_deep_cnn_rest5",
                                  "wesad_four_class_classical_RF_rest5"]),
        "",
    ]
    (HERE / "stats.toml").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Wrote {len(sources)} sources and {len(families)} comparison families to {HERE / 'stats.toml'}")


if __name__ == "__main__":
    main()
