"""Hand-crafted features per window, after Schmidt et al. (ICMI 2018, Table 1).

Features are computed from the raw signals at their native rates (chest
700 Hz, wrist BVP 64 Hz), not from the resampled deep-learning input, because
heart-rate variability needs millisecond beat timing.

Deviations from the dataset paper, kept deliberately simple:
  * one window length for every modality (the paper uses 5 s for ACC);
  * SCL/SCR split by a 0.05 Hz low-pass instead of the cvxEDA-style model;
  * R-peaks from a band-pass + energy-envelope detector written here.
Every window of the recording gets a row, labelled or not, so per-subject
feature standardisation can be done without using labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d
from scipy.signal import butter, find_peaks, peak_widths, periodogram, sosfiltfilt, welch

from .preprocess_clean import to_rate
from .wesad import CHEST_RATE, CONDITION_LABELS, WRIST_RATES, chest, labels, wrist

UNLABELLED = -1  # window does not lie inside a single condition

HRV_BANDS = {"ulf": (0.01, 0.04), "lf": (0.04, 0.15), "hf": (0.15, 0.4), "uhf": (0.4, 1.0)}
EMG_BANDS = np.linspace(0, 350, 8)  # seven equal bands up to Nyquist


def _filter(x: np.ndarray, fs: float, low: float | None = None, high: float | None = None, order: int = 3):
    if low and high:
        sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    elif low:
        sos = butter(order, low, btype="highpass", fs=fs, output="sos")
    else:
        sos = butter(order, high, btype="lowpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def _slope(x: np.ndarray, fs: float) -> float:
    """Least-squares slope in units per second."""
    if len(x) < 2:
        return np.nan
    t = np.arange(len(x)) / fs
    t = t - t.mean()
    return float((t * (x - x.mean())).sum() / (t ** 2).sum())


def _basic(x: np.ndarray, fs: float, prefix: str) -> dict:
    return {
        f"{prefix}mean": x.mean(), f"{prefix}std": x.std(), f"{prefix}min": x.min(),
        f"{prefix}max": x.max(), f"{prefix}range": x.max() - x.min(), f"{prefix}slope": _slope(x, fs),
    }


# --------------------------------------------------------------------------- beats

def detect_r_peaks(ecg: np.ndarray, fs: int = CHEST_RATE) -> np.ndarray:
    """R-peak times in seconds."""
    qrs = _filter(ecg, fs, 5, 20)
    envelope = uniform_filter1d(qrs ** 2, int(0.12 * fs))
    # Typical local peak height: rolling maximum over a beat, smoothed over ~8 s.
    reference = uniform_filter1d(maximum_filter1d(envelope, int(1.5 * fs)), int(8 * fs))
    candidates, _ = find_peaks(envelope, distance=int(0.3 * fs))
    candidates = candidates[envelope[candidates] > 0.3 * reference[candidates]]

    # Move each detection to the true R maximum of the lightly filtered ECG.
    clean = _filter(ecg, fs, 0.5, 40)
    half = int(0.06 * fs)
    peaks = []
    for c in candidates:
        lo, hi = max(0, c - half), min(len(clean), c + half + 1)
        peaks.append(lo + int(np.argmax(clean[lo:hi])))
    return np.unique(peaks) / fs


def detect_pulse_peaks(bvp: np.ndarray, fs: int = WRIST_RATES["BVP"]) -> np.ndarray:
    """Systolic peak times in seconds, with sub-sample (parabolic) refinement."""
    x = _filter(bvp, fs, 0.7, 3.5)
    local_std = np.sqrt(uniform_filter1d(x ** 2, int(5 * fs)))
    peaks, _ = find_peaks(x, distance=int(0.33 * fs), prominence=0.5 * local_std)
    peaks = peaks[(peaks > 0) & (peaks < len(x) - 1)]
    a, b, c = x[peaks - 1], x[peaks], x[peaks + 1]
    denom = a - 2 * b + c
    offset = np.divide(0.5 * (a - c), denom, out=np.zeros_like(b), where=np.abs(denom) > 1e-12)
    return (peaks + np.clip(offset, -0.5, 0.5)) / fs


def beat_intervals(peak_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inter-beat intervals, the time each one ends, and a validity mask."""
    ibi = np.diff(peak_times)
    local = median_filter(ibi, size=9, mode="nearest")
    valid = (ibi > 0.3) & (ibi < 2.0) & (np.abs(ibi - local) < 0.3 * local)
    return ibi, peak_times[1:], valid


def hrv_features(ibi: np.ndarray, t: np.ndarray, valid: np.ndarray, t0: float, t1: float, prefix: str) -> dict:
    names = ["hr_mean", "hr_std", "nn_mean", "sdnn", "rmssd", "nn50", "pnn50", "tri_index",
             *[f"{b}_power" for b in HRV_BANDS], *[f"{b}_rel" for b in HRV_BANDS],
             "total_power", "lf_hf", "lf_norm", "hf_norm"]
    out = {f"{prefix}{n}": np.nan for n in names}

    lo, hi = np.searchsorted(t, [t0, t1])
    ok = valid[lo:hi]
    rr, rr_t = ibi[lo:hi][ok], t[lo:hi][ok]
    if len(rr) < 8:
        return out

    # Successive differences only between intervals that were adjacent beats.
    adjacent = ok[1:] & ok[:-1]
    diffs = np.diff(ibi[lo:hi])[adjacent]
    hr = 60.0 / rr
    out.update({
        f"{prefix}hr_mean": hr.mean(), f"{prefix}hr_std": hr.std(),
        f"{prefix}nn_mean": rr.mean(), f"{prefix}sdnn": rr.std(),
    })
    if len(diffs):
        nn50 = int((np.abs(diffs) > 0.05).sum())
        out.update({
            f"{prefix}rmssd": np.sqrt((diffs ** 2).mean()),
            f"{prefix}nn50": nn50, f"{prefix}pnn50": nn50 / len(diffs),
        })
    hist, _ = np.histogram(rr, bins=np.arange(rr.min(), rr.max() + 1 / 64, 1 / 128))
    out[f"{prefix}tri_index"] = len(rr) / max(hist.max(), 1)

    # Spectrum of the evenly resampled tachogram (4 Hz).
    grid = np.arange(rr_t[0], rr_t[-1], 0.25)
    if len(grid) >= 16:
        tach = np.interp(grid, rr_t, rr)
        f, pxx = periodogram(tach - tach.mean(), fs=4.0, window="hann")
        df = f[1] - f[0]
        power = {b: pxx[(f >= lo_f) & (f < hi_f)].sum() * df for b, (lo_f, hi_f) in HRV_BANDS.items()}
        total = sum(power.values())
        for b, p in power.items():
            out[f"{prefix}{b}_power"] = p
            out[f"{prefix}{b}_rel"] = p / total if total > 0 else np.nan
        out[f"{prefix}total_power"] = total
        lf, hf = power["lf"], power["hf"]
        out[f"{prefix}lf_hf"] = lf / hf if hf > 0 else np.nan
        out[f"{prefix}lf_norm"] = lf / (lf + hf) if lf + hf > 0 else np.nan
        out[f"{prefix}hf_norm"] = hf / (lf + hf) if lf + hf > 0 else np.nan
    return out


# --------------------------------------------------------------------------- per-signal state

class _Eda:
    def __init__(self, eda: np.ndarray, fs: float):
        self.fs = fs
        self.eda = eda
        self.scl = _filter(eda, fs, high=0.05, order=2)
        self.scr = eda - self.scl
        peaks, props = find_peaks(self.scr, prominence=0.01, distance=max(1, int(fs)))
        self.peak_t = peaks / fs
        self.peak_amp = props["prominences"]
        self.peak_dur = peak_widths(self.scr, peaks, rel_height=0.5)[0] / fs

    def features(self, t0: float, t1: float, prefix: str) -> dict:
        a, b = int(t0 * self.fs), int(t1 * self.fs)
        eda, scl, scr = self.eda[a:b], self.scl[a:b], self.scr[a:b]
        lo, hi = np.searchsorted(self.peak_t, [t0, t1])
        time = np.arange(len(scl))
        return {
            **_basic(eda, self.fs, prefix),
            f"{prefix}scl_mean": scl.mean(), f"{prefix}scl_std": scl.std(),
            f"{prefix}scl_time_corr": np.corrcoef(time, scl)[0, 1] if scl.std() > 0 else 0.0,
            f"{prefix}scr_mean": scr.mean(), f"{prefix}scr_std": scr.std(),
            f"{prefix}scr_n": hi - lo,
            f"{prefix}scr_amp_sum": self.peak_amp[lo:hi].sum(),
            f"{prefix}scr_dur_sum": self.peak_dur[lo:hi].sum(),
            f"{prefix}scr_area": np.clip(scr, 0, None).sum() / self.fs,
        }


class _Resp:
    def __init__(self, resp: np.ndarray, fs: int):
        self.fs = 16
        self.x = to_rate(_filter(resp, fs, 0.1, 0.35), fs, self.fs)
        prominence = 0.3 * self.x.std()
        self.peaks = find_peaks(self.x, distance=int(1.5 * self.fs), prominence=prominence)[0]
        self.troughs = find_peaks(-self.x, distance=int(1.5 * self.fs), prominence=prominence)[0]

    def features(self, t0: float, t1: float, prefix: str) -> dict:
        a, b = int(t0 * self.fs), int(t1 * self.fs)
        peaks = self.peaks[(self.peaks >= a) & (self.peaks < b)]
        troughs = self.troughs[(self.troughs >= a) & (self.troughs < b)]
        seg = self.x[a:b]

        inhale, exhale, depth = [], [], []
        for p in peaks:
            before = troughs[troughs < p]
            after = troughs[troughs > p]
            if len(before):
                inhale.append((p - before[-1]) / self.fs)
                depth.append(self.x[p] - self.x[before[-1]])
            if len(after):
                exhale.append((after[0] - p) / self.fs)

        def stat(v, fn):
            return float(fn(v)) if len(v) else np.nan

        cycle = np.diff(peaks) / self.fs
        return {
            f"{prefix}inhale_mean": stat(inhale, np.mean), f"{prefix}inhale_std": stat(inhale, np.std),
            f"{prefix}exhale_mean": stat(exhale, np.mean), f"{prefix}exhale_std": stat(exhale, np.std),
            f"{prefix}ie_ratio": stat(inhale, np.mean) / stat(exhale, np.mean) if inhale and exhale else np.nan,
            f"{prefix}stretch": seg.max() - seg.min(),
            f"{prefix}depth_mean": stat(depth, np.mean),
            f"{prefix}rate": len(peaks) / (t1 - t0) * 60.0,
            f"{prefix}cycle_mean": stat(cycle, np.mean),
        }


class _Emg:
    def __init__(self, emg: np.ndarray, fs: int):
        self.fs = fs
        self.x = _filter(emg, fs, low=1.0)  # remove the DC component
        smooth = _filter(self.x, fs, high=50.0)
        mad = np.median(np.abs(smooth - np.median(smooth)))
        self.peaks, props = find_peaks(smooth, height=3 * 1.4826 * mad, distance=int(0.05 * fs))
        self.peak_amp = props["peak_heights"]

    def features(self, t0: float, t1: float, prefix: str) -> dict:
        a, b = int(t0 * self.fs), int(t1 * self.fs)
        x = self.x[a:b]
        f, pxx = welch(x, fs=self.fs, nperseg=min(2048, len(x)))
        cum = np.cumsum(pxx)
        lo, hi = np.searchsorted(self.peaks, [a, b])
        amp = self.peak_amp[lo:hi]
        out = {
            f"{prefix}mean": x.mean(), f"{prefix}std": x.std(), f"{prefix}range": x.max() - x.min(),
            f"{prefix}abs_integral": np.abs(x).sum() / self.fs, f"{prefix}median": np.median(x),
            f"{prefix}p10": np.percentile(x, 10), f"{prefix}p90": np.percentile(x, 90),
            f"{prefix}freq_mean": (f * pxx).sum() / pxx.sum(),
            f"{prefix}freq_median": f[np.searchsorted(cum, cum[-1] / 2)],
            f"{prefix}freq_peak": f[np.argmax(pxx)],
            f"{prefix}peaks_n": hi - lo,
            f"{prefix}peaks_amp_mean": amp.mean() if len(amp) else 0.0,
            f"{prefix}peaks_amp_std": amp.std() if len(amp) else 0.0,
            f"{prefix}peaks_amp_sum": amp.sum(),
        }
        for i in range(len(EMG_BANDS) - 1):
            band = (f >= EMG_BANDS[i]) & (f < EMG_BANDS[i + 1])
            out[f"{prefix}band{i + 1}"] = pxx[band].sum() / pxx.sum()
        return out


class _Acc:
    def __init__(self, acc: np.ndarray, fs: int):
        self.fs = 32
        xyz = np.stack([to_rate(acc[:, i], fs, self.fs) for i in range(3)], axis=1)
        self.axes = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "mag": np.linalg.norm(xyz, axis=1)}

    def features(self, t0: float, t1: float, prefix: str) -> dict:
        a, b = int(t0 * self.fs), int(t1 * self.fs)
        out = {}
        for name, sig in self.axes.items():
            x = sig[a:b]
            out[f"{prefix}{name}_mean"] = x.mean()
            out[f"{prefix}{name}_std"] = x.std()
            out[f"{prefix}{name}_abs_integral"] = np.abs(x).sum() / self.fs
            if name != "mag":
                f, pxx = periodogram(x - x.mean(), fs=self.fs)
                out[f"{prefix}{name}_freq_peak"] = f[np.argmax(pxx)]
        return out


# --------------------------------------------------------------------------- per subject

def window_grid(lab: np.ndarray, window_sec: float, stride_sec: float, min_purity: float):
    """Start time and label of every window; label is UNLABELLED unless one condition dominates."""
    win, step = int(window_sec * CHEST_RATE), int(stride_sec * CHEST_RATE)
    starts, window_labels = [], []
    for start in range(0, len(lab) - win + 1, step):
        counts = np.bincount(lab[start:start + win], minlength=8)
        best = max(CONDITION_LABELS, key=lambda c: counts[c])
        starts.append(start / CHEST_RATE)
        window_labels.append(best if counts[best] / win >= min_purity else UNLABELLED)
    return np.array(starts), np.array(window_labels)


def subject_features(data: dict, subject: str, window_sec: float, stride_sec: float,
                     min_purity: float = 1.0) -> pd.DataFrame:
    """One row per window of the recording: subject, label, start_sec, then features."""
    ecg_ibi = beat_intervals(detect_r_peaks(chest(data, "ECG")))
    bvp_ibi = beat_intervals(detect_pulse_peaks(wrist(data, "BVP")))

    chest_eda = _Eda(to_rate(_filter(chest(data, "EDA"), CHEST_RATE, high=5.0), CHEST_RATE, 16), 16)
    wrist_eda = _Eda(wrist(data, "EDA"), WRIST_RATES["EDA"])
    chest_temp = to_rate(chest(data, "Temp"), CHEST_RATE, 4)
    wrist_temp = wrist(data, "TEMP")
    emg = _Emg(chest(data, "EMG"), CHEST_RATE)
    resp = _Resp(chest(data, "Resp"), CHEST_RATE)
    chest_acc = _Acc(chest(data, "ACC"), CHEST_RATE)
    wrist_acc = _Acc(wrist(data, "ACC"), WRIST_RATES["ACC"])

    starts, window_labels = window_grid(labels(data), window_sec, stride_sec, min_purity)
    rows = []
    for t0, label in zip(starts, window_labels):
        t1 = t0 + window_sec
        a4, b4 = int(t0 * 4), int(t1 * 4)
        rows.append({
            "subject": subject, "label": int(label), "start_sec": float(t0),
            **hrv_features(*ecg_ibi, t0, t1, "chest_ecg_"),
            **chest_eda.features(t0, t1, "chest_eda_"),
            **emg.features(t0, t1, "chest_emg_"),
            **resp.features(t0, t1, "chest_resp_"),
            **_basic(chest_temp[a4:b4], 4, "chest_temp_"),
            **chest_acc.features(t0, t1, "chest_acc_"),
            **hrv_features(*bvp_ibi, t0, t1, "wrist_bvp_"),
            **wrist_eda.features(t0, t1, "wrist_eda_"),
            **_basic(wrist_temp[a4:b4], 4, "wrist_temp_"),
            **wrist_acc.features(t0, t1, "wrist_acc_"),
        })
    return pd.DataFrame(rows)
