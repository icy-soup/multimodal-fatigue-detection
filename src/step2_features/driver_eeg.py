"""Driver EEG 特征提取

基于多通道EEG功率谱特征。
每通道提取 delta/theta/alpha/beta/gamma 频带绝对+相对功率 = 10维
总计: 34通道 × 10 = 340维 (若选择子集则更少)
"""

import numpy as np
from scipy import signal

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']


def extract_features(windows, fs=1000):
    """提取多通道EEG频带功率特征

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_samples, n_channels)
    fs : int

    Returns
    -------
    features : np.ndarray, shape (n_windows, n_channels * len(BANDS) * 2)
        每通道: 5个绝对功率 + 5个相对功率 = 10维
    """
    n_wins, n_samples, n_ch = windows.shape
    n_features = n_ch * len(BANDS) * 2
    features = np.zeros((n_wins, n_features), dtype=np.float64)

    for i in range(n_wins):
        idx = 0
        for ch in range(n_ch):
            sig = windows[i, :, ch]
            # 计算频谱
            f, Pxx = signal.welch(sig, fs, nperseg=min(fs*2, n_samples))
            total_power = Pxx.sum()

            for band_name in BAND_NAMES:
                lo, hi = BANDS[band_name]
                mask = (f >= lo) & (f < hi)
                bp = Pxx[mask].sum()
                features[i, idx] = bp          # 绝对功率
                features[i, idx + 1] = bp / (total_power + 1e-10)  # 相对功率
                idx += 2

    return features


def extract_dataset(all_data, fs=1000):
    """聚合所有 session 为训练格式"""
    X_list, y_list, subj_list = [], [], []

    for d in all_data:
        X = extract_features(d['windows'], d.get('fs', fs))
        X_list.append(X)
        y_list.append(np.full(len(X), d['label'], dtype=int))
        subj_list.extend([d['subject']] * len(X))

    from itertools import chain

    return {
        'X': np.vstack(X_list),
        'y': np.concatenate(y_list),
        'modality_dims': np.array([X_list[0].shape[1]]),
        'subject_ids': list(chain.from_iterable([d['subject']] * X.shape[0] for d, X in zip(all_data, X_list))),
    }
