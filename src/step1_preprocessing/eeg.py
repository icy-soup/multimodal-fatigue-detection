"""EEG 信号预处理

处理 FatigueSet 的 forehead_eeg_raw.csv（4通道，256Hz）：
  1. 加载 CSV
  2. 0.5-45Hz 带通滤波
  3. 50Hz 陷波（去工频干扰）
  4. 滑动窗口切分
"""

import numpy as np
import pandas as pd
from . import utils

EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
EEG_BANDPASS_LOW = 0.5
EEG_BANDPASS_HIGH = 45
NOTCH_FREQ = 50.0
KNOWN_FS = 256
WINDOW_SEC = 4.0
STEP_SEC = 2.0


def load_raw(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    expected_cols = ['timestamp'] + EEG_CHANNELS
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"EEG CSV 缺少列: {missing}")
    timestamps = utils.timestamps_to_relative(df['timestamp'].values)
    data = np.zeros((len(df), len(EEG_CHANNELS)), dtype=np.float64)
    for i, ch in enumerate(EEG_CHANNELS):
        col = pd.to_numeric(df[ch], errors='coerce')
        data[:, i] = col.interpolate(method='linear').fillna(0).values
    return timestamps, data


def filter_signal(data, fs=KNOWN_FS):
    b, a = utils.butter_bandpass(EEG_BANDPASS_LOW, EEG_BANDPASS_HIGH, fs)
    filtered = utils.apply_filter(b, a, data, axis=0)
    b_notch, a_notch = utils.notch_filter(NOTCH_FREQ, fs)
    filtered = utils.apply_filter(b_notch, a_notch, filtered, axis=0)
    return filtered


def preprocess(csv_path, fs=KNOWN_FS, window_sec=WINDOW_SEC, step_sec=STEP_SEC):
    timestamps, data = load_raw(csv_path)
    filtered = filter_signal(data, fs)
    target_samples = int(round(window_sec * fs))
    window_ts = utils.time_based_windows(timestamps, window_sec, step_sec)

    windows_list, ts_list = [], []
    for start_idx, end_idx, t_start, t_end in window_ts:
        seg = filtered[start_idx:end_idx]
        if len(seg) < target_samples * 0.5:
            continue
        if len(seg) >= target_samples:
            windows_list.append(seg[:target_samples])
        else:
            windows_list.append(np.pad(seg, ((0, target_samples - len(seg)), (0, 0))))
        ts_list.append((t_start, t_end))

    return {
        'windows': np.array(windows_list),
        'timestamps': np.array(ts_list),
        'fs': fs,
        'n_channels': len(EEG_CHANNELS),
    }
