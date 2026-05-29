"""Driver EEG 数据集加载器

从 CNT 文件读取数据（Neuroscan 格式）。
每 subject 有 Fatigue state 和 Normal state 两种状态。
"""

import numpy as np
import struct
import zipfile
from pathlib import Path
from . import utils

# 排除的非EEG通道（EOG + 参考）
NON_EEG_CHS = {'HEOL', 'HEOR', 'VEOU', 'VEOL', 'A1', 'A2'}
# 所有40通道中排除后得到34个EEG通道
ALL_CHS = ['HEOL', 'HEOR', 'FP1', 'FP2', 'VEOU', 'VEOL',
           'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8',
           'T3', 'C3', 'CZ', 'C4', 'T4', 'TP7', 'CP3', 'CPZ', 'CP4', 'TP8',
           'A1', 'T5', 'P3', 'PZ', 'P4', 'T6', 'A2', 'O1', 'OZ', 'O2',
           'FT9', 'FT10', 'PO1', 'PO2']
EEG_CHS = [ch for ch in ALL_CHS if ch not in NON_EEG_CHS]
EEG_INDICES = [i for i, ch in enumerate(ALL_CHS) if ch not in NON_EEG_CHS]

KNOWN_FS = 1000
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 45
WINDOW_SEC = 4.0
STEP_SEC = 2.0


def read_cnt_raw(filepath):
    """读取 Neuroscan CNT 文件，返回 (data, fs, ch_names)

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    data : np.ndarray, shape (n_samples, n_channels)
    fs : int
    ch_names : list of str
    """
    with open(str(filepath), 'rb') as f:
        raw = f.read()

    # 解析通道标签 (从偏移900开始, 每通道75字节)
    n_ch = 40
    ch_names = []
    for i in range(n_ch):
        start = 900 + i * 75
        label = raw[start:start+10].decode('ascii', errors='replace').strip().rstrip('\x00').strip()
        ch_names.append(label)

    # 解析采样率 (Neuroscan CNT格式, 偏移360处为uint32)
    fs = struct.unpack_from('<I', raw, 360)[0]
    if fs not in [128, 256, 500, 1000, 1024]:
        fs = 1000

    # 读取数据 (int16, 40通道, 层叠)
    header_end = 900 + n_ch * 75
    data_bytes = raw[header_end:]
    n_samples = len(data_bytes) // (n_ch * 2)
    data = np.frombuffer(data_bytes[:n_samples * n_ch * 2], dtype=np.int16)
    data = data.reshape(n_samples, n_ch).astype(np.float64)
    # Neuroscan 单位转换: 1 digit = 0.0298 uV
    data *= 0.0298

    return data, fs, ch_names


def extract_eeg(data, ch_names):
    """从多通道数据中提取EEG通道"""
    idx = [ch_names.index(ch) for ch in EEG_CHS if ch in ch_names]
    return data[:, idx]


def preprocess_subject(zip_path, label):
    """处理单个subject的CNT数据

    Parameters
    ----------
    zip_path : Path - zip文件路径
    label : int - 标签 (1=疲劳, 0=正常)

    Returns
    -------
    dict with windows, timestamps, subject, label
    """
    subject = zip_path.stem  # 数字编号 1-12

    # 解压CNT
    extract_dir = Path(zip_path.parent) / 'extracted'
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(str(zip_path)) as z:
        cnt_files = [n for n in z.namelist() if n.endswith('.cnt')]
        if not cnt_files:
            return None
        z.extractall(str(extract_dir))

    cnt_path = extract_dir / cnt_files[0]
    if not cnt_path.exists():
        return None

    # 读取数据
    data, fs, ch_names = read_cnt_raw(str(cnt_path))

    # 提取EEG通道
    eeg_data = extract_eeg(data, ch_names)

    if eeg_data.shape[1] == 0:
        return None

    # 滤波
    b, a = utils.butter_bandpass(BANDPASS_LOW, BANDPASS_HIGH, fs)
    filtered = utils.apply_filter(b, a, eeg_data, axis=0)

    # 滑窗
    target_samples = int(round(WINDOW_SEC * fs))
    window_ts = utils.time_based_windows(np.arange(len(filtered)) / fs, WINDOW_SEC, STEP_SEC)

    wins, ts_list = [], []
    for start_idx, end_idx, t_start, t_end in window_ts:
        seg = filtered[start_idx:end_idx]
        if len(seg) < target_samples * 0.5:
            continue
        if len(seg) >= target_samples:
            wins.append(seg[:target_samples])
        else:
            wins.append(np.pad(seg, ((0, target_samples - len(seg)), (0, 0))))
        ts_list.append((t_start, t_end))

    if not wins:
        return None

    return {
        'windows': np.array(wins),  # (n_windows, n_samples, n_eeg_ch)
        'timestamps': np.array(ts_list),
        'subject': subject,
        'label': label,
        'fs': fs,
        'n_channels': len(EEG_CHS),
        'ch_names': EEG_CHS,
    }


def load(data_root, subjects=None):
    """加载Driver EEG数据集

    Parameters
    ----------
    data_root : str or Path
    subjects : list of int, default all (1-12)

    Returns
    -------
    list of dict, each with windows, subject, label
    """
    data_root = Path(data_root)
    if subjects is None:
        subjects = list(range(1, 13))

    all_data = []
    for subj in subjects:
        subj = str(int(subj)) if not isinstance(subj, str) else subj
        zip_path = data_root / f'{subj}.zip'
        if not zip_path.exists():
            continue

        print(f"\nDriver EEG Subject {int(subj):02d}")
        for state, label in [('Fatigue', 1), ('Normal', 0)]:
            result = preprocess_subject_condition(zip_path, state, label)
            if result is None:
                print(f"  [WARN] {state}: 加载失败")
                continue

            n_wins = len(result['windows'])
            print(f"  {state}: {n_wins} windows, {result['windows'].shape}")
            all_data.append(result)

    print(f"\n完成！共 {len(all_data)} 个 session")
    return all_data


def preprocess_subject_condition(zip_path, condition, label):
    """处理单个subject单个条件的CNT"""
    import zipfile
    from pathlib import Path

    with zipfile.ZipFile(str(zip_path)) as z:
        cnt_name = f'{zip_path.stem}/{condition} state.cnt'
        if cnt_name not in z.namelist():
            return None

        # 直接读取CNT内容到内存
        cnt_bytes = z.read(cnt_name)

    # 临时写入文件（read_raw_cnt需要文件路径）
    tmp_dir = Path(zip_path.parent) / 'tmp'
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f'{zip_path.stem}_{condition}.cnt'
    with open(str(tmp_path), 'wb') as f:
        f.write(cnt_bytes)

    # 读取数据
    data, fs, ch_names = read_cnt_raw(str(tmp_path))

    # 提取EEG通道
    eeg_data = extract_eeg(data, ch_names)
    if eeg_data.shape[1] == 0:
        return None

    # 滤波
    b, a = utils.butter_bandpass(BANDPASS_LOW, BANDPASS_HIGH, fs)
    filtered = utils.apply_filter(b, a, eeg_data, axis=0)

    # 滑窗
    target_samples = int(round(WINDOW_SEC * fs))
    window_ts = utils.time_based_windows(np.arange(len(filtered)) / fs, WINDOW_SEC, STEP_SEC)

    wins, ts_list = [], []
    for start_idx, end_idx, t_start, t_end in window_ts:
        seg = filtered[start_idx:end_idx]
        if len(seg) < target_samples * 0.5:
            continue
        if len(seg) >= target_samples:
            wins.append(seg[:target_samples])
        else:
            wins.append(np.pad(seg, ((0, target_samples - len(seg)), (0, 0))))
        ts_list.append((t_start, t_end))

    # 清理临时文件
    try:
        tmp_path.unlink()
    except:
        pass

    if not wins:
        return None

    return {
        'windows': np.array(wins),
        'timestamps': np.array(ts_list),
        'subject': zip_path.stem,
        'label': label,
        'fs': fs,
        'n_channels': len(EEG_CHS),
        'ch_names': EEG_CHS,
    }


def aggregate(all_data):
    """聚合数据为训练格式

    对每个window取所有EEG通道的平均，然后提取特征
    """
    X_list, y_list, subj_list = [], [], []

    for d in all_data:
        X_list.append(d['windows'])
        y_list.append(np.full(len(d['windows']), d['label'], dtype=int))
        subj_list.extend([d['subject']] * len(d['windows']))

    return {
        'X': np.vstack(X_list),  # (n_windows, n_samples, n_eeg_ch)
        'y': np.concatenate(y_list),
        'subject_ids': subj_list,
    }
