"""ECG 特征提取

HRV时域(4) + 频域(3) + Poincaré(3) + DFA(2) = 12维
"""

import numpy as np
from scipy import signal as sp_signal

FS = 250


def detect_r_peaks(ecg, fs=FS):
    import neurokit2 as nk
    try:
        _, info = nk.ecg_peaks(ecg, sampling_rate=fs)
        rp = info['ECG_R_Peaks']
        return rp, np.diff(rp) / fs
    except Exception:
        peaks, _ = sp_signal.find_peaks(ecg, distance=int(0.4 * fs), height=np.std(ecg))
        if len(peaks) < 2:
            return np.array([0]), np.array([0.75])
        return peaks, np.diff(peaks) / fs


def _hrv_time(rr):
    if len(rr) < 2:
        return {'sdnn': 0, 'rmssd': 0, 'pnn50': 0, 'mean_hr': 0}
    rr_ms = rr * 1000
    diffs = np.diff(rr_ms)
    return {'sdnn': np.std(rr_ms, ddof=1), 'rmssd': np.sqrt(np.mean(diffs**2)),
            'pnn50': np.sum(np.abs(diffs) > 50) / max(len(diffs), 1) * 100,
            'mean_hr': 60.0 / np.mean(rr)}


def _hrv_freq(rr):
    if len(rr) < 4:
        return {'lf': 0, 'hf': 0, 'lf_hf': 1}
    t = np.cumsum(rr) - rr[0]
    freqs = np.linspace(0.01, 0.5, 500)
    p = sp_signal.lombscargle(t, rr, freqs, normalize=True)
    lf = np.trapz(p[np.where((freqs >= 0.04) & (freqs <= 0.15))])
    hf = np.trapz(p[np.where((freqs >= 0.15) & (freqs <= 0.4))])
    return {'lf': lf, 'hf': hf, 'lf_hf': lf / (hf + 1e-10)}


def _poincare(rr):
    if len(rr) < 3:
        return {'sd1': 0, 'sd2': 0, 'sd1_sd2': 1}
    x, y = rr[:-1] * 1000, rr[1:] * 1000
    sd1 = np.std((x - y) / np.sqrt(2), ddof=1)
    sd2 = np.std((x + y - 2 * np.mean(rr * 1000)) / np.sqrt(2), ddof=1)
    return {'sd1': sd1, 'sd2': sd2, 'sd1_sd2': sd1 / (sd2 + 1e-10)}


def _dfa(x, scales):
    y = np.cumsum(x - np.mean(x))
    fluct = []
    for s in scales:
        if s > len(y) // 2:
            continue
        n = len(y) // s
        rms = [np.sqrt(np.mean((y[i*s:(i+1)*s] - np.polyval(np.polyfit(np.arange(s), y[i*s:(i+1)*s], 1), np.arange(s)))**2)) for i in range(n)]
        fluct.append(np.mean(rms))
    if len(fluct) < 2:
        return 0.5
    return np.polyfit(np.log(scales[:len(fluct)]), np.log(fluct), 1)[0]


def _dfa_features(rr):
    if len(rr) < 12:
        return {'dfa_a1': 0.5, 'dfa_a2': 0.5}
    return {'dfa_a1': _dfa(rr, np.array([4, 6, 8, 10, 12, 14, 16])),
            'dfa_a2': _dfa(rr, np.array([16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64]))}


def extract_window(ecg_window, fs=FS):
    _, rr = detect_r_peaks(ecg_window, fs)
    feats = {}
    feats.update(_hrv_time(rr))
    feats.update(_hrv_freq(rr))
    feats.update(_poincare(rr))
    feats.update(_dfa_features(rr))
    keys = ['sdnn', 'rmssd', 'pnn50', 'mean_hr', 'lf', 'hf', 'lf_hf', 'sd1', 'sd2', 'sd1_sd2', 'dfa_a1', 'dfa_a2']
    return np.array([feats[k] for k in keys])


def extract_batch(ecg_windows, fs=FS):
    n, d = len(ecg_windows), 12
    out = np.zeros((n, d))
    for i in range(n):
        out[i] = extract_window(ecg_windows[i], fs)
    return out
