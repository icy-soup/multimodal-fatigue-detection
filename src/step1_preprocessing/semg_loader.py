"""sEMG 数据集加载器

从嵌套 zip 中读取数据，支持按 subject 加载。
数据来源: data/semg/sEMG_data.zip + data/semg/self_perceived_fatigue_index.zip
"""

import numpy as np
import pandas as pd
import zipfile
import io
import re
from pathlib import Path
from . import semg, utils

N_SAMPLES_THRESHOLD = 0.5  # 窗口有效采样比例低于此值则丢弃

# 每个 trial 对应的主要肌肉 (来自 code.ipynb)
TRIAL_TO_MUSCLE = {
    1: "R_DELTOID_ANTERIOR", 2: "L_DELTOID_ANTERIOR",
    3: "R_DELTOID_POSTERIOR", 4: "L_DELTOID_POSTERIOR",
    5: "R_BICEPS_BRACHII", 6: "L_BICEPS_BRACHII",
    7: "R_DELTOID_MEDIUS", 8: "L_DELTOID_MEDIUS",
    9: "R_DELTOID_ANTERIOR_C", 10: "L_DELTOID_ANTERIOR_C",
    11: "R_DELTOID_POSTERIOR_C", 12: "L_DELTOID_POSTERIOR_C",
}

# CSV列布局: 每个肌肉通道为(时间, EMG)一对，共4对
# EMG_index / time_index 分别对应4对的EMG列和时间的列号
_EMG_IDX = [1, 3, 5, 7]
_TIME_IDX = [0, 2, 4, 6]
# get_prime_mover 返回用哪一对 (0-3)
_PRIME_MOVER = {1: 1, 2: 0, 3: 3, 4: 2, 5: 0, 6: 1, 7: 2, 8: 3, 9: 1, 10: 0, 11: 3, 12: 2}


def _get_emg_and_time(trial_data, trial_num):
    """提取 trial 的主肌肉 EMG 信号和时间戳"""
    idx = _PRIME_MOVER[trial_num]
    emg_col = trial_data.columns[_EMG_IDX[idx]]
    time_col = trial_data.columns[_TIME_IDX[idx]]
    return trial_data[emg_col].values.astype(np.float64), trial_data[time_col].values.astype(np.float64)


def load_subject_emg(subj_num, data_root, window_sec=4.0, step_sec=2.0):
    """加载单个 subject 的所有 trial，返回处理后的窗口数据

    Parameters
    ----------
    subj_num : int (1-13)
    data_root : Path
    window_sec, step_sec : float

    Returns
    -------
    list of dict, each with windows, timestamps, labels, subject, trial, muscle
    """
    data_root = Path(data_root)
    emg_zip = data_root / 'sEMG_data.zip'
    label_zip = data_root / 'self_perceived_fatigue_index.zip'

    # 从 sEMG_data.zip 读取 subject 的 trial CSV
    subj_key = f'subject_{subj_num}'
    with zipfile.ZipFile(str(emg_zip)) as outer:
        inner_bytes = outer.read(f'sEMG_data/{subj_key}.zip')

    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        trial_files = sorted(
            [n for n in inner.namelist() if n.endswith('.csv') and 'trial' in n.lower()],
            key=lambda x: int(re.findall(r'\d+', x)[-1])
        )

        results = []
        for tf in trial_files:
            trial_num = int(re.findall(r'\d+', tf)[-1])
            muscle = TRIAL_TO_MUSCLE[trial_num]
            df = pd.read_csv(io.BytesIO(inner.read(tf)))

            emg_signal, time_stamps = _get_emg_and_time(df, trial_num)

            # 滤波 (已知采样率1259Hz, 直接从diff中位数估算)
            diffs = np.diff(time_stamps)
            diffs = diffs[diffs > 0]
            fs = int(round(1.0 / np.median(diffs))) if len(diffs) > 0 else 1259
            b, a = utils.butter_bandpass(20, 450, fs)
            filtered = utils.apply_filter(b, a, emg_signal, axis=0)

            # 滑窗
            target_samples = int(round(window_sec * fs))
            window_ts = utils.time_based_windows(time_stamps, window_sec, step_sec)

            wins, ts_list = [], []
            for start_idx, end_idx, t_start, t_end in window_ts:
                seg = filtered[start_idx:end_idx]
                if len(seg) < target_samples * N_SAMPLES_THRESHOLD:
                    continue
                if len(seg) >= target_samples:
                    wins.append(seg[:target_samples])
                else:
                    wins.append(np.pad(seg, ((0, target_samples - len(seg)),)))
                ts_list.append((t_start, t_end))

            results.append({
                'windows': np.array(wins),
                'timestamps': np.array(ts_list),
                'subject': f'{subj_num:02d}',
                'trial': trial_num,
                'muscle': muscle,
                'fs': fs,
            })

        return results


def load_subject_labels(subj_num, data_root):
    """加载单个 subject 的标签（每个 trial 的时间序列标签）"""
    data_root = Path(data_root)
    label_zip = data_root / 'self_perceived_fatigue_index.zip'
    subj_key = f'subject_{subj_num}'

    with zipfile.ZipFile(str(label_zip)) as outer:
        inner_bytes = outer.read(f'self_perceived_fatigue_index/{subj_key}.zip')

    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        label_files = sorted(
            [n for n in inner.namelist() if n.endswith('.csv')],
            key=lambda x: int(re.findall(r'\d+', x)[-1])
        )

        labels = {}
        for lf in label_files:
            trial_num = int(re.findall(r'\d+', lf)[-1])
            df = pd.read_csv(io.BytesIO(inner.read(lf)))
            labels[trial_num] = {'time': df.iloc[:, 0].values, 'label': df.iloc[:, 1].values.astype(int)}

        return labels


def assign_labels(trial_windows, trial_labels):
    """给窗口分配标签：取窗口内标签的众数"""
    win_labels = np.full(len(trial_windows['windows']), -1, dtype=int)
    timestamps = trial_windows['timestamps']

    for i, (t_start, t_end) in enumerate(timestamps):
        mask = (trial_labels['time'] >= t_start) & (trial_labels['time'] < t_end)
        if mask.sum() > 0:
            vals = trial_labels['label'][mask]
            win_labels[i] = 1 if vals.mean() > 0.5 else 0
        else:
            win_labels[i] = 0

    return win_labels


def load(data_root, subjects=None, window_sec=4.0, step_sec=2.0):
    """加载 sEMG 数据集

    Parameters
    ----------
    data_root : str or Path
    subjects : list of int, default all (1-13)
    window_sec, step_sec : float

    Returns
    -------
    all_data : list of dict
    """
    data_root = Path(data_root)
    if subjects is None:
        subjects = list(range(1, 14))

    all_data = []
    for subj in subjects:
        print(f"\nsEMG Subject {subj:02d}")
        trial_data = load_subject_emg(subj, data_root, window_sec, step_sec)
        trial_labels = load_subject_labels(subj, data_root)

        for td in trial_data:
            trial_num = td['trial']
            if trial_num not in trial_labels:
                continue
            labels = assign_labels(td, trial_labels[trial_num])
            valid = labels != -1
            if valid.sum() == 0:
                continue
            td['windows'] = td['windows'][valid]
            td['timestamps'] = td['timestamps'][valid]
            td['labels'] = labels[valid]
            all_data.append(td)

            print(f"  Trial {trial_num:2d} ({td['muscle']:25s}): {len(td['windows']):4d} windows, "
                  f"疲劳={(labels[valid]==1).sum()}, 非疲劳={(labels[valid]==0).sum()}")

    print(f"\n完成！共 {len(all_data)} 个 trial")
    return all_data


def aggregate(all_data):
    """聚合所有 trial 数据为训练格式"""
    X_list, y_list, subj_list, muscle_list = [], [], [], []

    for d in all_data:
        X_list.append(d['windows'])
        y_list.append(d['labels'])
        subj_list.extend([d['subject']] * len(d['labels']))
        muscle_list.extend([d['muscle']] * len(d['labels']))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return {
        'X': X,
        'y': y,
        'subject_ids': subj_list,
        'muscle_names': muscle_list,
    }
