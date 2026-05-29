"""FatigueSet 数据集主加载器
"""

import numpy as np
import pandas as pd
from pathlib import Path
from . import eeg, ecg, eda

FATIGUE_THRESHOLD = 30
WINDOW_SEC = 4.0
STEP_SEC = 2.0


def load_labels(csv_path, threshold=FATIGUE_THRESHOLD):
    df = pd.read_csv(csv_path)
    labels = []
    for _, row in df.iterrows():
        binary = 1 if row['mentalFatigueScore'] > threshold else 0
        labels.append({'time_sec': row['fatigueSurveySubmissionTime'],
                       'score': row['mentalFatigueScore'], 'binary': binary})
    return labels


def assign_labels_to_windows(window_timestamps, labels):
    n = len(window_timestamps)
    win_labels = np.full(n, -1, dtype=int)
    win_scores = np.full(n, np.nan, dtype=float)
    label_times = np.array([l['time_sec'] for l in labels])
    for i in range(n):
        win_end = window_timestamps[i, 1]
        future = np.where(label_times >= win_end)[0]
        idx = future[0] if len(future) > 0 else -1
        win_labels[i] = labels[idx]['binary']
        win_scores[i] = labels[idx]['score']
    return win_labels, win_scores


def _resolve_eeg_path(session_dir):
    """有些 session 的 forehead_eeg_raw.csv 为空（仅表头），实际数据在 _raw2.csv"""
    primary = session_dir / 'forehead_eeg_raw.csv'
    if primary.exists():
        df_check = pd.read_csv(primary, nrows=1)
        if len(df_check) > 0:
            return primary
        fallback = session_dir / 'forehead_eeg_raw2.csv'
        if fallback.exists():
            print(f"    [FALLBACK] {primary.name} 为空, 使用 {fallback.name}")
            return fallback
    return primary


def process_session(session_dir, window_sec=WINDOW_SEC, step_sec=STEP_SEC):
    session_dir = Path(session_dir)
    eeg_path = _resolve_eeg_path(session_dir)
    ecg_path = session_dir / 'chest_raw_ecg.csv'
    eda_path = session_dir / 'wrist_eda.csv'
    fatigue_path = session_dir / 'exp_fatigue.csv'

    missing = [str(p) for p in [eeg_path, ecg_path, eda_path, fatigue_path] if not p.exists()]
    if missing:
        print(f"  [WARN] 缺少文件: {missing}")
        return None

    eeg_out = eeg.preprocess(str(eeg_path), window_sec=window_sec, step_sec=step_sec)
    ecg_out = ecg.preprocess(str(ecg_path), window_sec=window_sec, step_sec=step_sec)
    eda_out = eda.preprocess(str(eda_path), window_sec=window_sec, step_sec=step_sec)

    for name, mod in [('EEG', eeg_out), ('ECG', ecg_out), ('EDA', eda_out)]:
        if len(mod['windows']) == 0:
            print(f"  [WARN] {name} 无有效窗口")
            return None

    t_max = min(eeg_out['timestamps'][-1, 1], ecg_out['timestamps'][-1, 1], eda_out['timestamps'][-1, 1])
    t_min = max(eeg_out['timestamps'][0, 0], ecg_out['timestamps'][0, 0], eda_out['timestamps'][0, 0])

    def in_range(ts): return (ts[:, 0] >= t_min) & (ts[:, 1] <= t_max)

    eeg_wins = eeg_out['windows'][in_range(eeg_out['timestamps'])]
    ecg_wins = ecg_out['windows'][in_range(ecg_out['timestamps'])]
    eda_wins = eda_out['windows'][in_range(eda_out['timestamps'])]
    common_ts = eeg_out['timestamps'][in_range(eeg_out['timestamps'])]

    n_wins = min(len(eeg_wins), len(ecg_wins), len(eda_wins))
    eeg_wins, ecg_wins, eda_wins = eeg_wins[:n_wins], ecg_wins[:n_wins], eda_wins[:n_wins]
    common_ts = common_ts[:n_wins]

    if n_wins == 0:
        print("  [WARN] 无有效窗口")
        return None

    labels = load_labels(str(fatigue_path))
    win_labels, win_scores = assign_labels_to_windows(common_ts, labels)

    eda_mask = in_range(eda_out['timestamps'])
    print(f"  OK: {n_wins} windows, 疲劳={(win_labels==1).sum()}, 非疲劳={(win_labels==0).sum()}")

    return {
        'subject': session_dir.parent.name, 'session': session_dir.name,
        'eeg': eeg_wins, 'ecg': ecg_wins, 'eda': eda_wins,
        'scl': eda_out['scl_windows'][eda_mask][:n_wins],
        'scr': eda_out['scr_windows'][eda_mask][:n_wins],
        'scr_peaks': eda_out['scr_peaks'][:n_wins],
        'timestamps': common_ts, 'labels': win_labels, 'scores': win_scores,
        'eeg_fs': eeg_out['fs'], 'ecg_fs': ecg_out['fs'], 'eda_fs': eda_out['fs'],
        'n_windows': n_wins,
    }


def load(data_root, subjects=None, sessions=None, window_sec=WINDOW_SEC, step_sec=STEP_SEC):
    data_root = Path(data_root)
    if subjects is None:
        subjects = [f"{i:02d}" for i in range(1, 13)]
    if sessions is None:
        sessions = [f"{i:02d}" for i in range(1, 4)]

    all_data = []
    for subj in subjects:
        for sess in sessions:
            session_dir = data_root / subj / sess
            if not session_dir.exists():
                continue
            print(f"\nSubject {subj}, Session {sess}")
            result = process_session(session_dir, window_sec, step_sec)
            if result is not None:
                all_data.append(result)

    print(f"\n完成！共 {len(all_data)} 个 session")
    return all_data


def aggregate(all_data):
    eeg_list, ecg_list, eda_list = [], [], []
    y_list, subj_list = [], []

    for d in all_data:
        eeg_list.append(d['eeg'])
        ecg_list.append(d['ecg'])
        eda_list.append(d['eda'])
        y_list.append(d['labels'])
        subj_list.extend([d['subject']] * d['n_windows'])

    return {
        'X': {'eeg': np.vstack(eeg_list), 'ecg': np.vstack(ecg_list), 'eda': np.vstack(eda_list)},
        'y': np.concatenate(y_list),
        'subject_ids': subj_list,
    }
