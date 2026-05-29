"""通用信号处理工具函数：滤波器设计、滑动窗口、时间对齐"""

import numpy as np
from scipy import signal


# ─── 滤波器设计 ───────────────────────────────────────────────

def butter_bandpass(lowcut, highcut, fs, order=4):
    """设计 Butterworth 带通滤波器系数"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    return b, a


def butter_lowpass(cutoff, fs, order=4):
    """设计 Butterworth 低通滤波器系数"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a


def butter_highpass(cutoff, fs, order=4):
    """设计 Butterworth 高通滤波器系数"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='highpass')
    return b, a


def notch_filter(freq, fs, quality=30):
    """设计陷波滤波器（用于去除工频干扰 50Hz）"""
    b, a = signal.iirnotch(freq, quality, fs)
    return b, a


def apply_filter(b, a, data, axis=-1):
    """零相位滤波（filtfilt），不造成波形偏移

    Parameters
    ----------
    b, a : ndarray
        滤波器系数
    data : ndarray
    axis : int
        滤波轴向，多通道信号按列滤波 axis=0
    """
    return signal.filtfilt(b, a, data, axis=axis, padlen=None)


# ─── 滑动窗口 ─────────────────────────────────────────────────

def sliding_window_indices(n_samples, window_len, step_len):
    """生成滑动窗口的起止索引

    Parameters
    ----------
    n_samples : int
        信号总长度
    window_len : int
        每个窗口的样本数
    step_len : int
        窗口步长（样本数）

    Returns
    -------
    list of (start, end) tuples
    """
    indices = []
    for start in range(0, n_samples - window_len + 1, step_len):
        indices.append((start, start + window_len))
    return indices


def sliding_windows(data, window_len, step_len):
    """对数组进行滑动窗口切分

    Parameters
    ----------
    data : ndarray, shape=(n_samples,) or (n_samples, n_channels)
    window_len : int
    step_len : int

    Returns
    -------
    windows : ndarray, shape=(n_windows, window_len) or (n_windows, window_len, n_channels)
    """
    is_1d = data.ndim == 1
    if is_1d:
        data = data.reshape(-1, 1)

    n_samples = data.shape[0]
    indices = sliding_window_indices(n_samples, window_len, step_len)
    n_channels = data.shape[1]

    result = np.zeros((len(indices), window_len, n_channels), dtype=data.dtype)
    for i, (s, e) in enumerate(indices):
        result[i] = data[s:e]

    if is_1d:
        result = result.squeeze(-1)
    return result


def time_based_windows(timestamps, window_duration, step_duration):
    """基于时间（而非样本数）的滑动窗口切分

    通过时间戳对齐不同采样率的信号，保证各模态窗口覆盖同一时间段。

    Parameters
    ----------
    timestamps : ndarray, shape=(n_samples,)
        时间戳（秒，单调递增）
    window_duration : float
        窗口时长（秒）
    step_duration : float
        步长（秒）

    Returns
    -------
    list of (start_idx, end_idx, t_start, t_end)
    """
    windows = []
    t_start = timestamps[0]
    t_end = timestamps[-1]
    cur = t_start

    while cur + window_duration <= t_end:
        ts = cur
        te = cur + window_duration
        start_idx = int(np.searchsorted(timestamps, ts))
        end_idx = int(np.searchsorted(timestamps, te))
        if end_idx - start_idx > 0:
            windows.append((start_idx, end_idx, ts, te))
        cur += step_duration

    return windows


# ─── 时间戳处理 ───────────────────────────────────────────────

def estimate_fs(timestamps):
    """估算信号采样率（基于时间戳中位数间隔）"""
    diffs = np.diff(timestamps)
    # 过滤重复时间戳和负间隔
    valid = diffs[diffs > 0]
    if len(valid) == 0:
        return 1
    median_diff = np.median(valid)
    return int(round(1.0 / median_diff))


def timestamps_to_relative(timestamps, t0=None):
    """将绝对毫秒时间戳转换为相对秒数

    Parameters
    ----------
    timestamps : ndarray
        毫秒级绝对时间戳
    t0 : float or None
        参考零点（毫秒），None 则用第一个时间戳

    Returns
    -------
    relative_seconds : ndarray
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    if t0 is None:
        t0 = ts[0]
    return (ts - t0) / 1000.0


def align_to_common_time(eeg_ts, ecg_ts, eda_ts,
                         window_duration=4.0, step_duration=2.0):
    """找到 EEG/ECG/EDA 三模态的公共时间窗口

    Parameters
    ----------
    eeg_ts : ndarray
    ecg_ts : ndarray
    eda_ts : ndarray
        均为相对秒数
    window_duration, step_duration : float

    Returns
    -------
    common_windows : list of (t_start, t_end)
        三个信号都覆盖的时间段
    """
    t_max = min(eeg_ts[-1], ecg_ts[-1], eda_ts[-1])
    t_min = max(eeg_ts[0], ecg_ts[0], eda_ts[0])

    common = []
    cur = t_min
    while cur + window_duration <= t_max:
        common.append((cur, cur + window_duration))
        cur += step_duration
    return common
