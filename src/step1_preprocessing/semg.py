"""sEMG 信号预处理（对照实验用）
"""

import numpy as np
import pandas as pd
from . import utils

SEMG_BANDPASS_LOW = 20
SEMG_BANDPASS_HIGH = 450
WINDOW_SEC = 4.0
STEP_SEC = 2.0


def load_raw(csv_path):
    df = pd.read_csv(csv_path)
    emg_cols = [c for c in df.columns if 'EMG' in c]
    if not emg_cols:
        raise ValueError(f"未找到 EMG 列")
    time_col = [c for c in df.columns if 'X [' in c or 'time' in c.lower()]
    timestamps = df[time_col[0]].values.astype(np.float64) if time_col else np.arange(len(df)) / 1259.0
    data = df[emg_cols].values.astype(np.float64)
    return timestamps, data, emg_cols


def filter_signal(data, fs):
    b, a = utils.butter_bandpass(SEMG_BANDPASS_LOW, SEMG_BANDPASS_HIGH, fs)
    return utils.apply_filter(b, a, data, axis=0)


def preprocess(csv_path, window_sec=WINDOW_SEC, step_sec=STEP_SEC):
    timestamps, data, ch_names = load_raw(csv_path)
    fs = utils.estimate_fs(timestamps)
    filtered = filter_signal(data, fs)

    target_samples = int(round(window_sec * fs))
    window_ts = utils.time_based_windows(timestamps, window_sec, step_sec)

    wins, ts_list = [], []
    for start_idx, end_idx, t_start, t_end in window_ts:
        seg = filtered[start_idx:end_idx]
        if len(seg) < target_samples * 0.5:
            continue
        wins.append(seg[:target_samples] if len(seg) >= target_samples else np.pad(seg, ((0, target_samples - len(seg)), (0, 0))))
        ts_list.append((t_start, t_end))

    return {
        'windows': np.array(wins),
        'timestamps': np.array(ts_list),
        'fs': fs,
        'channel_names': ch_names,
    }
