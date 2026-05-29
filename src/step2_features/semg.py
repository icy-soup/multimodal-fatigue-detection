"""sEMG 特征提取

时域 + 频域特征，每窗口输出固定维度的特征向量。

时域(10维): MAV, RMS, ZC, SSC, WL, VAR, IEMG, SSI, SKEW, KURT
频域(10维): MNF, MDF, PKF, FR, TTP, BP1~BP5 (5个频段能量)
总计: 20维
"""

import numpy as np
from scipy import signal

FREQ_BANDS = {'low': (20, 60), 'mid_low': (60, 120), 'mid': (120, 200),
              'mid_high': (200, 300), 'high': (300, 450)}
BAND_NAMES = ['low', 'mid_low', 'mid', 'mid_high', 'high']


# ─── 时域特征 ─────────────────────────────────────────────

def _mav(x):
    return np.mean(np.abs(x))


def _rms(x):
    return np.sqrt(np.mean(x ** 2))


def _zc(x, threshold=1e-6):
    """过零率"""
    diff = x[:-1] * x[1:]
    return np.sum(diff < 0) / len(x)


def _ssc(x, threshold=1e-6):
    """斜率符号变化"""
    diff = np.diff(x)
    ssc_sum = 0
    for i in range(1, len(diff)):
        if diff[i] * diff[i - 1] < 0 and (np.abs(diff[i]) > threshold or np.abs(diff[i - 1]) > threshold):
            ssc_sum += 1
    return ssc_sum / len(x)


def _wl(x):
    """波形长度"""
    return np.sum(np.abs(np.diff(x))) / len(x)


def _var(x):
    return np.var(x, ddof=1)


def _iemg(x):
    """积分肌电"""
    return np.sum(np.abs(x)) / len(x)


def _ssi(x):
    """简单平方积分"""
    return np.sum(x ** 2) / len(x)


def _skew(x):
    return float(pd_safe(x, lambda v: np.mean(((v - np.mean(v)) / (np.std(v, ddof=1) + 1e-10)) ** 3)))


def _kurt(x):
    return float(pd_safe(x, lambda v: np.mean(((v - np.mean(v)) / (np.std(v, ddof=1) + 1e-10)) ** 4)))


def pd_safe(x, func):
    if len(x) < 3 or np.std(x, ddof=1) < 1e-10:
        return 0.0
    return func(x)


def extract_time_features(window):
    """从单个窗口提取时域特征 (10维)"""
    return np.array([
        _mav(window), _rms(window), _zc(window), _ssc(window), _wl(window),
        _var(window), _iemg(window), _ssi(window), _skew(window), _kurt(window),
    ])


# ─── 频域特征 ─────────────────────────────────────────────

def _get_spectrum(window, fs):
    n = len(window)
    if n < 4:
        return np.array([0.0]), np.array([0.0])
    windowed = window * np.hanning(n)
    fft_vals = np.fft.rfft(windowed)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, power


def _mnf(power, freqs):
    total = np.sum(power)
    return np.sum(freqs * power) / total if total > 0 else 0.0


def _mdf(power, freqs):
    cum_power = np.cumsum(power)
    total = cum_power[-1]
    if total <= 0:
        return 0.0
    idx = np.searchsorted(cum_power, total / 2)
    return freqs[min(idx, len(freqs) - 1)]


def _pkf(power, freqs):
    return freqs[np.argmax(power)]


def _fr(power, freqs):
    """频带比: 高频/低频 能量比 (以120Hz为界)"""
    low = np.sum(power[freqs <= 120])
    high = np.sum(power[freqs > 120])
    return high / (low + 1e-10)


def _ttp(power):
    return np.sum(power)


def _band_power(power, freqs, bands):
    bp = []
    for bname, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        bp.append(np.sum(power[mask]))
    return np.array(bp)


def extract_freq_features(window, fs):
    """从单个窗口提取频域特征 (10维)"""
    freqs, power = _get_spectrum(window, fs)
    if len(freqs) < 4:
        return np.zeros(10)

    return np.array([
        _mnf(power, freqs), _mdf(power, freqs), _pkf(power, freqs),
        _fr(power, freqs), _ttp(power),
        *_band_power(power, freqs, FREQ_BANDS),
    ])


# ─── 批量接口 ─────────────────────────────────────────────

def extract_batch(windows, fs=1259):
    """从批量窗口提取特征

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_samples) or (n_windows, n_samples, n_channels)
        如果是3D，取所有通道平均
    fs : int

    Returns
    -------
    features : np.ndarray, shape (n_windows, 20)
    """
    if windows.ndim == 3:
        windows = windows.mean(axis=2)  # 多通道平均

    n = len(windows)
    features = np.zeros((n, 20), dtype=np.float64)

    for i in range(n):
        features[i, :10] = extract_time_features(windows[i])
        features[i, 10:] = extract_freq_features(windows[i], fs)

    return features


def extract_dataset(all_data):
    """聚合所有 trial 数据为训练格式

    Parameters
    ----------
    all_data : list of dict (from semg_loader.load)

    Returns
    -------
    dict with X, y, modality_dims, subject_ids
    """
    X_list, y_list, subj_list = [], [], []

    for d in all_data:
        X = extract_batch(d['windows'], d.get('fs', 1259))
        X_list.append(X)
        y_list.append(d['labels'])
        subj_list.extend([d['subject']] * len(d['labels']))

    from itertools import chain

    return {
        'X': np.vstack(X_list),
        'y': np.concatenate(y_list),
        'modality_dims': np.array([X_list[0].shape[1]]),
        'subject_ids': list(chain.from_iterable([d['subject']] * len(d['labels']) for d in all_data)),
    }
