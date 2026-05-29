"""EEG 特征提取

频带功率(32维) + 熵(12维) + 功能连接(30维) = 74维
"""

import numpy as np
from scipy import signal
from antropy import perm_entropy

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']
CH_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
FS = 256


def _sampen(x, order=2, r_factor=0.2):
    from scipy.spatial.distance import pdist
    n = len(x)
    if n < order + 2:
        return 0.0
    r = r_factor * np.std(x, ddof=1)
    if r == 0:
        return 0.0

    def _cnt(m):
        tmpl = np.array([x[i:i + m] for i in range(n - m)])
        dists = pdist(tmpl, metric='chebyshev')
        return np.sum(dists < r)

    A, B = _cnt(order + 1), _cnt(order)
    return 0.0 if A == 0 or B == 0 else -np.log(A / B)


def extract_band(eeg):
    feats = {}
    for ch in range(eeg.shape[1]):
        f, psd = signal.welch(eeg[:, ch], fs=FS, nperseg=min(256, len(eeg)))
        pw = {n: np.mean(psd[np.where((f >= b[0]) & (f <= b[1]))]) for n, b in BANDS.items()}
        for n in BAND_NAMES:
            feats[f'ch{ch}_{n}'] = pw[n]
        feats[f'ch{ch}_tb'] = pw['theta'] / (pw['beta'] + 1e-10)
        feats[f'ch{ch}_ab'] = pw['alpha'] / (pw['beta'] + 1e-10)
        feats[f'ch{ch}_tab'] = (pw['theta'] + pw['alpha']) / (pw['beta'] + 1e-10)
    return feats


def extract_entropy(eeg):
    feats = {}
    for ch in range(eeg.shape[1]):
        x = eeg[:, ch]
        feats[f'ch{ch}_sampen'] = _sampen(x)
        feats[f'ch{ch}_permen'] = perm_entropy(x, normalize=True)
        mse = []
        for s in [1, 2, 5, 10]:
            n = len(x) // s
            coarse = np.mean(x[:n * s].reshape(n, s), axis=1)
            mse.append(_sampen(coarse))
        feats[f'ch{ch}_mse'] = np.mean(mse)
    return feats


def extract_connectivity(eeg):
    feats = {}
    for a, b in CH_PAIRS:
        f, Cxy = signal.coherence(eeg[:, a], eeg[:, b], fs=FS, nperseg=min(256, len(eeg) // 2))
        for name, (lo, hi) in BANDS.items():
            idx = np.where((f >= lo) & (f <= hi))
            feats[f'coh_{a}{b}_{name}'] = np.mean(Cxy[idx]) if idx[0].size > 0 else 0.0
    return feats


def extract_window(eeg_window):
    feats = {}
    feats.update(extract_band(eeg_window))
    feats.update(extract_entropy(eeg_window))
    feats.update(extract_connectivity(eeg_window))
    return np.array([feats[k] for k in sorted(feats.keys())])


def extract_batch(eeg_windows):
    n, d = len(eeg_windows), len(extract_window(eeg_windows[0]))
    out = np.zeros((n, d))
    for i in range(n):
        out[i] = extract_window(eeg_windows[i])
    return out
