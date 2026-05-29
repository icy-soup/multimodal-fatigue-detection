"""ECG 信号预处理

处理 FatigueSet 的 chest_raw_ecg.csv（单通道，250Hz）：
  1. 加载 CSV
  2. 0.5-45Hz 带通滤波
  3. 滑动窗口切分
"""

import numpy as np
import pandas as pd
from . import utils

ECG_BANDPASS_LOW = 0.5
ECG_BANDPASS_HIGH = 45
KNOWN_FS = 250
WINDOW_SEC = 4.0
STEP_SEC = 2.0


def load_raw(csv_path):
    df = pd.read_csv(csv_path)
    if 'ecg_waveform' not in df.columns:
        raise ValueError(f"ECG CSV 缺少 'ecg_waveform' 列")
    timestamps = utils.timestamps_to_relative(df['timestamp'].values)
    data = df['ecg_waveform'].values.astype(np.float64)
    return timestamps, data


def filter_signal(data, fs=KNOWN_FS):
    b, a = utils.butter_bandpass(ECG_BANDPASS_LOW, ECG_BANDPASS_HIGH, fs)
    return utils.apply_filter(b, a, data)


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
            windows_list.append(np.pad(seg, (0, target_samples - len(seg))))
        ts_list.append((t_start, t_end))

    return {
        'windows': np.array(windows_list),
        'timestamps': np.array(ts_list),
        'fs': fs,
    }
