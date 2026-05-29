"""EDA 信号预处理

处理 FatigueSet 的 wrist_eda.csv（4Hz，单通道）：
  1. 加载 CSV
  2. 低通滤波 (<1Hz)
  3. SCL/SCR 分解
  4. 滑动窗口切分
"""

import numpy as np
import pandas as pd
from scipy import signal
from . import utils

EDA_LOWPASS_CUTOFF = 1.0
KNOWN_FS = 4
SCR_MIN_HEIGHT = 0.01
SCR_MIN_DISTANCE = 10
WINDOW_SEC = 4.0
STEP_SEC = 2.0


def load_raw(csv_path):
    df = pd.read_csv(csv_path)
    if 'eda' not in df.columns:
        raise ValueError(f"EDA CSV 缺少 'eda' 列")
    timestamps = utils.timestamps_to_relative(df['timestamp'].values)
    data = df['eda'].values.astype(np.float64)
    return timestamps, data


def filter_signal(data, fs=KNOWN_FS):
    b, a = utils.butter_lowpass(EDA_LOWPASS_CUTOFF, fs, order=2)
    return utils.apply_filter(b, a, data)


def decompose(eda_signal, fs=KNOWN_FS):
    b_scl, a_scl = utils.butter_lowpass(0.05, fs, order=2)
    scl = utils.apply_filter(b_scl, a_scl, eda_signal)
    scr = np.clip(eda_signal - scl, 0, None)
    scr_peaks, _ = signal.find_peaks(scr, height=SCR_MIN_HEIGHT, distance=int(SCR_MIN_DISTANCE * fs))
    return scl, scr, scr_peaks


def _slice_windows(data, window_ts, target_samples):
    wins, ts_list = [], []
    for start_idx, end_idx, t_start, t_end in window_ts:
        seg = data[start_idx:end_idx]
        if len(seg) < target_samples * 0.5:
            continue
        if len(seg) >= target_samples:
            wins.append(seg[:target_samples])
        else:
            wins.append(np.pad(seg, (0, target_samples - len(seg))))
        ts_list.append((t_start, t_end))
    return np.array(wins), np.array(ts_list)


def preprocess(csv_path, fs=KNOWN_FS, window_sec=WINDOW_SEC, step_sec=STEP_SEC):
    timestamps, data = load_raw(csv_path)
    filtered = filter_signal(data, fs)
    scl, scr, scr_peaks = decompose(filtered, fs)

    target_samples = int(round(window_sec * fs))
    window_ts = utils.time_based_windows(timestamps, window_sec, step_sec)

    windows, ts_arr = _slice_windows(filtered, window_ts, target_samples)
    scl_wins, _ = _slice_windows(scl, window_ts, target_samples)
    scr_wins, _ = _slice_windows(scr, window_ts, target_samples)

    win_peaks = []
    for start_idx, end_idx, _, _ in window_ts:
        in_win = (scr_peaks >= start_idx) & (scr_peaks < end_idx)
        win_peaks.append(scr_peaks[in_win] - start_idx)

    return {
        'windows': windows,
        'timestamps': ts_arr,
        'scl_windows': scl_wins,
        'scr_windows': scr_wins,
        'scr_peaks': win_peaks,
        'fs': fs,
    }
