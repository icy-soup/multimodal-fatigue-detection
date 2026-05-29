"""EDA 特征提取

SCL(3) + SCR(3) = 6维
"""

import numpy as np


def _scl_feats(scl):
    if len(scl) < 2:
        return {'scl_mean': 0, 'scl_std': 0, 'scl_slope': 0}
    return {'scl_mean': np.mean(scl), 'scl_std': np.std(scl, ddof=1),
            'scl_slope': (scl[-1] - scl[0]) / len(scl)}


def _scr_feats(scr, peaks):
    if len(peaks) == 0:
        return {'scr_mean_amp': 0, 'scr_freq': 0, 'scr_sum_amp': 0}
    amps = scr[peaks]
    return {'scr_mean_amp': np.mean(amps), 'scr_freq': len(peaks), 'scr_sum_amp': np.sum(amps)}


def extract_window(eda_window, scl_window=None, scr_window=None, scr_peaks=None):
    if scl_window is None:
        scl_window = eda_window
    if scr_window is None:
        scr_window = np.zeros_like(eda_window)
    if scr_peaks is None:
        scr_peaks = np.array([], dtype=int)

    feats = {}
    feats.update(_scl_feats(scl_window))
    feats.update(_scr_feats(scr_window, scr_peaks))
    keys = ['scl_mean', 'scl_std', 'scl_slope', 'scr_mean_amp', 'scr_freq', 'scr_sum_amp']
    return np.array([feats[k] for k in keys])


def extract_batch(eda_windows, scl_windows=None, scr_windows=None, scr_peaks_list=None):
    n = len(eda_windows)
    out = np.zeros((n, 6))
    for i in range(n):
        scl = scl_windows[i] if scl_windows is not None else None
        scr = scr_windows[i] if scr_windows is not None else None
        peaks = scr_peaks_list[i] if scr_peaks_list is not None else None
        out[i] = extract_window(eda_windows[i], scl, scr, peaks)
    return out
