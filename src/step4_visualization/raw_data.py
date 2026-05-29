"""原始信号可视化 — 生成报告用的数据展示图

生成图表：
  1. EEG 时域 + 频谱 (01_eeg_signal_spectrum.png)
  2. ECG 时域 + R波检测 (02_ecg_rwave_detection.png)
  3. EDA 时域 + SCL/SCR分解 (03_eda_scl_scr_decomposition.png)
  4. 标签分布 (04_label_distribution.png)
  5. 综合概览 (05_data_overview_heatmap.png)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
})

EEG_CHS = ['TP9', 'AF7', 'AF8', 'TP10']
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
BAND_COLORS = {'delta': '#1f77b4', 'theta': '#ff7f0e', 'alpha': '#2ca02c', 'beta': '#d62728', 'gamma': '#9467bd'}


def load_session_raw(session_dir):
    """加载单个 session 的原始 CSV 数据"""
    session_dir = Path(session_dir)
    eeg_df = pd.read_csv(session_dir / 'forehead_eeg_raw.csv')
    ecg_df = pd.read_csv(session_dir / 'chest_raw_ecg.csv')
    eda_df = pd.read_csv(session_dir / 'wrist_eda.csv')
    fatigue_df = pd.read_csv(session_dir / 'exp_fatigue.csv')

    eeg_ts = (eeg_df['timestamp'].values - eeg_df['timestamp'].values[0]) / 1000.0
    ecg_ts = (ecg_df['timestamp'].values - ecg_df['timestamp'].values[0]) / 1000.0
    eda_ts = (eda_df['timestamp'].values - eda_df['timestamp'].values[0]) / 1000.0

    return {
        'eeg_ts': eeg_ts, 'eeg_data': eeg_df[EEG_CHS].values.astype(np.float64),
        'ecg_ts': ecg_ts, 'ecg_data': ecg_df['ecg_waveform'].values.astype(np.float64),
        'eda_ts': eda_ts, 'eda_data': eda_df['eda'].values.astype(np.float64),
        'fatigue': fatigue_df,
    }


def _butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype='bandpass')
    return b, a


def _butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyq, btype='lowpass')
    return b, a


def _notch(freq, fs, q=30):
    return signal.iirnotch(freq, q, fs)


def plot_eeg(session_dir, output_dir, duration=15):
    """EEG 时域+频谱图"""
    d = load_session_raw(session_dir)
    fs_eeg = 256
    t = d['eeg_ts']
    data = d['eeg_data']

    mask = t <= duration
    t_plot = t[mask]
    raw_plot = data[mask]

    b, a = _butter_bandpass(0.5, 45, fs_eeg)
    bn, an = _notch(50, fs_eeg)
    filtered = signal.filtfilt(b, a, raw_plot, axis=0)
    filtered = signal.filtfilt(bn, an, filtered, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2.5, 1, 1]})

    # 子图1: 滤波后 EEG 时域 (前15秒)
    ax = axes[0]
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    for i, (ch, c) in enumerate(zip(EEG_CHS, colors)):
        offset = (3 - i) * 80
        ax.plot(t_plot, filtered[:, i] + offset, color=c, lw=0.6, label=f'{ch}')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('幅值 (μV) + 偏移')
    ax.set_title('EEG 四通道信号 (0.5-45Hz 带通 + 50Hz 陷波)')
    ax.legend(loc='upper right', ncol=4, fontsize=8)
    ax.set_xlim(0, duration)

    # 子图2: PSD 频谱
    ax = axes[1]
    for i, (ch, c) in enumerate(zip(EEG_CHS, colors)):
        f, psd = signal.welch(filtered[:, i], fs=fs_eeg, nperseg=1024)
        ax.semilogy(f, psd, color=c, lw=0.8, alpha=0.8, label=ch)
    for name, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, alpha=0.07, color=BAND_COLORS[name], label=f'${name}$' if name == list(BANDS.keys())[0] else '')
    ax.set_xlim(0.5, 50)
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('功率谱密度 (μV²/Hz)')
    ax.set_title('Welch PSD 频谱')
    ax.legend(loc='upper right', ncol=4, fontsize=8)

    # 子图3: 频带地形图 (4通道 × 5频带热力图)
    ax = axes[2]
    band_powers = np.zeros((4, 5))
    for ci in range(4):
        f, psd = signal.welch(filtered[:, ci], fs=fs_eeg, nperseg=1024)
        for bi, (name, (lo, hi)) in enumerate(BANDS.items()):
            idx = (f >= lo) & (f <= hi)
            band_powers[ci, bi] = np.log10(np.mean(psd[idx]) + 1e-10)

    im = ax.imshow(band_powers.T, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(range(4))
    ax.set_xticklabels(EEG_CHS)
    ax.set_yticks(range(5))
    ax.set_yticklabels(list(BANDS.keys()))
    ax.set_title('各通道频带功率 (log10)')
    plt.colorbar(im, ax=ax, shrink=0.85)

    fig.tight_layout()
    out = Path(output_dir) / '01_eeg_signal_spectrum.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")
    return str(out)


def plot_ecg(session_dir, output_dir, duration=10):
    """ECG 时域 + R波检测图"""
    d = load_session_raw(session_dir)
    fs = 250
    t = d['ecg_ts']
    raw = d['ecg_data']

    mask = t <= duration
    t_plot = t[mask]
    raw_plot = raw[mask]

    b, a = _butter_bandpass(0.5, 45, fs)
    filtered = signal.filtfilt(b, a, raw_plot)

    import neurokit2 as nk
    _, info = nk.ecg_peaks(filtered, sampling_rate=fs)
    rpeaks = info['ECG_R_Peaks']

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # 子图1: ECG 滤波后 + R波标记
    ax = axes[0]
    ax.plot(t_plot, filtered, color='#E91E63', lw=0.7)
    for rp in rpeaks:
        if rp < len(t_plot):
            ax.axvline(t_plot[rp], color='#FF5722', alpha=0.35, lw=1.2)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('幅值')
    ax.set_title('ECG 信号 (0.5-45Hz 带通, R波已标记)')
    ax.set_xlim(0, duration)

    # 子图2: RR间期 (tachogram)
    ax = axes[1]
    if len(rpeaks) >= 2:
        rr = np.diff(rpeaks) / fs * 1000
        rr_t = t_plot[rpeaks[1:]]
        ax.plot(rr_t, rr, 'o-', color='#2196F3', ms=3, lw=0.8)
        ax.axhline(np.mean(rr), color='red', ls='--', lw=0.8, label=f'Mean RR={np.mean(rr):.0f}ms')
        ax.legend(fontsize=9)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('RR 间期 (ms)')
    ax.set_title('RR 间期 (心率变异性)')
    ax.set_xlim(0, duration)

    fig.tight_layout()
    out = Path(output_dir) / '02_ecg_rwave_detection.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")
    return str(out)


def plot_eda(session_dir, output_dir, duration=60):
    """EDA 时域 + SCL/SCR分解图"""
    d = load_session_raw(session_dir)
    fs = 4
    t = d['eda_ts']
    raw = d['eda_data']

    mask = t <= duration
    t_plot = t[mask]
    raw_plot = raw[mask]

    # 低通滤波
    b, a = _butter_lowpass(1.0, fs)
    filtered = signal.filtfilt(b, a, raw_plot)

    # SCL/SCR 分解
    b_scl, a_scl = _butter_lowpass(0.05, fs, order=2)
    scl = signal.filtfilt(b_scl, a_scl, filtered)
    scr = np.clip(filtered - scl, 0, None)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # 子图1: 原始 vs 滤波
    ax = axes[0]
    ax.plot(t_plot, raw_plot, color='#BDBDBD', lw=0.5, alpha=0.7, label='原始')
    ax.plot(t_plot, filtered, color='#4CAF50', lw=0.8, label='低通滤波 (<1Hz)')
    ax.set_ylabel('电导 (μS)')
    ax.set_title('EDA 信号 — 原始 vs 滤波')
    ax.legend(fontsize=8)

    # 子图2: SCL (tonic)
    ax = axes[1]
    ax.plot(t_plot, scl, color='#2196F3', lw=0.8)
    ax.fill_between(t_plot, scl.min() - 0.1, scl, alpha=0.15, color='#2196F3')
    ax.set_ylabel('电导 (μS)')
    ax.set_title('皮肤电导水平 SCL (tonic, <0.05Hz)')

    # 子图3: SCR (phasic) + peaks
    ax = axes[2]
    ax.plot(t_plot, scr, color='#FF5722', lw=0.6)
    peaks, _ = signal.find_peaks(scr, height=0.01, distance=10)
    ax.plot(t_plot[peaks], scr[peaks], 'o', color='#9C27B0', ms=3, label=f'n={len(peaks)} peaks')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('电导 (μS)')
    ax.set_title('皮肤电导反应 SCR (phasic)')
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = Path(output_dir) / '03_eda_scl_scr_decomposition.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")
    return str(out)


def plot_label_distribution(data_root, output_dir):
    """全部 subject/session 的标签分布图"""
    data_root = Path(data_root)
    subjects = sorted([d.name for d in data_root.iterdir() if d.is_dir() and d.name.isdigit()])

    all_labels = []
    session_info = []

    for subj in subjects:
        for sess in ['01', '02', '03']:
            csv_path = data_root / subj / sess / 'exp_fatigue.csv'
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            scores = df['mentalFatigueScore'].values
            n_fatigue = np.sum(scores > 30)
            n_nonfatigue = np.sum(scores <= 30)
            session_info.append({'subject': subj, 'session': sess,
                                'n_fatigue': n_fatigue, 'n_nonfatigue': n_nonfatigue,
                                'mean_score': np.mean(scores)})
            all_labels.extend(scores)

    df_info = pd.DataFrame(session_info)
    all_scores = np.array(all_labels)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 疲劳分数分布直方图
    ax = axes[0, 0]
    ax.hist(all_scores, bins=30, color='#607D8B', edgecolor='white', alpha=0.85)
    ax.axvline(30, color='red', ls='--', lw=2, label='阈值=30')
    ax.set_xlabel('mentalFatigueScore')
    ax.set_ylabel('频次')
    ax.set_title('所有 session 疲劳分数分布')
    ax.legend()

    # 子图2: 疲劳 vs 非疲劳 饼图
    ax = axes[0, 1]
    n_f = np.sum(all_scores > 30)
    n_nf = np.sum(all_scores <= 30)
    ax.pie([n_f, n_nf], labels=['疲劳 (>30)', '非疲劳 (≤30)'], autopct='%1.1f%%',
           colors=['#FF5722', '#4CAF50'], startangle=90, explode=(0.02, 0))
    ax.set_title(f'疲劳标签比例 (n={len(all_scores)})')

    # 子图3: 各 subject 的疲劳比例
    ax = axes[1, 0]
    subj_summary = df_info.groupby('subject').agg(n_fatigue=('n_fatigue', 'sum'),
                                                   n_nonfatigue=('n_nonfatigue', 'sum')).reset_index()
    subj_summary['total'] = subj_summary['n_fatigue'] + subj_summary['n_nonfatigue']
    subj_summary['fatigue_ratio'] = subj_summary['n_fatigue'] / subj_summary['total']
    bars = ax.bar(range(len(subj_summary)), subj_summary['fatigue_ratio'] * 100, color='#FF5722', alpha=0.8)
    ax.set_xticks(range(len(subj_summary)))
    ax.set_xticklabels(subj_summary['subject'])
    ax.set_xlabel('Subject')
    ax.set_ylabel('疲劳比例 (%)')
    ax.set_title('各 Subject 疲劳标签占比')
    ax.axhline(50, color='gray', ls='--', lw=0.8)
    ax.set_ylim(0, 105)

    # 子图4: 各 session 疲劳/非疲劳 堆叠柱状图
    ax = axes[1, 1]
    x = np.arange(len(df_info))
    w = 0.6
    ax.bar(x, df_info['n_nonfatigue'], w, color='#4CAF50', alpha=0.8, label='非疲劳')
    ax.bar(x, df_info['n_fatigue'], w, bottom=df_info['n_nonfatigue'], color='#FF5722', alpha=0.8, label='疲劳')
    ax.set_xticks(x[::3])
    ax.set_xticklabels([f"{r['subject']}" for _, r in df_info.iloc[::3].iterrows()])
    ax.set_xlabel('Subject (每个subject的3个session)')
    ax.set_ylabel('样本数')
    ax.set_title('各 Session 标签分布')
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = Path(output_dir) / '04_label_distribution.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")
    return str(out)


def plot_overview(data_root, output_dir):
    """综合概览 — 数据可用性热力图"""
    data_root = Path(data_root)
    subjects = sorted([d.name for d in data_root.iterdir() if d.is_dir() and d.name.isdigit()])

    grid_eeg, grid_ecg, grid_eda = np.zeros((12, 3)), np.zeros((12, 3)), np.zeros((12, 3))
    grid_labels = np.zeros((12, 3))

    for si, subj in enumerate(subjects):
        for ssi, sess in enumerate(['01', '02', '03']):
            d = data_root / subj / sess
            if not d.exists():
                continue
            try:
                eeg = pd.read_csv(d / 'forehead_eeg_raw.csv')
                ecg = pd.read_csv(d / 'chest_raw_ecg.csv')
                eda = pd.read_csv(d / 'wrist_eda.csv')
                fat = pd.read_csv(d / 'exp_fatigue.csv')
                grid_eeg[si, ssi] = len(eeg) / 256  # seconds
                grid_ecg[si, ssi] = len(ecg) / 250
                grid_eda[si, ssi] = len(eda) / 4
                grid_labels[si, ssi] = np.mean(fat['mentalFatigueScore'])
            except Exception:
                pass

    fig, axes = plt.subplots(1, 4, figsize=(16, 6))

    def _heatmap(ax, data, title, cmap='YlOrRd', fmt='.0f'):
        im = ax.imshow(data, aspect='auto', cmap=cmap)
        for i in range(12):
            for j in range(3):
                v = data[i, j]
                ax.text(j, i, f'{v:{fmt}}' if v > 0 else '-', ha='center', va='center', fontsize=7)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['S01', 'S02', 'S03'])
        ax.set_yticks(range(12))
        ax.set_yticklabels(subjects)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

    _heatmap(axes[0], grid_eeg, 'EEG 时长 (s)', fmt='.0f')
    _heatmap(axes[1], grid_ecg, 'ECG 时长 (s)', fmt='.0f')
    _heatmap(axes[2], grid_eda, 'EDA 时长 (s)', fmt='.0f')
    _heatmap(axes[3], grid_labels, '平均疲劳分数', cmap='RdYlGn_r', fmt='.1f')

    fig.suptitle('数据集概览 — 12 Subject × 3 Session', fontsize=14, y=1.01)
    fig.tight_layout()
    out = Path(output_dir) / '05_data_overview_heatmap.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")
    return str(out)


def generate_all(data_root, output_dir):
    """生成全部原始数据可视化图"""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("生成原始数据可视化")
    print("=" * 60)

    # 用 Subject 01, Session 01 展示示例信号
    demo_session = data_root / '01' / '01'

    print("\n[1/5] EEG 时域+频谱...")
    plot_eeg(demo_session, output_dir)

    print("\n[2/5] ECG 时域+R波...")
    plot_ecg(demo_session, output_dir)

    print("\n[3/5] EDA 时域+SCL/SCR...")
    plot_eda(demo_session, output_dir)

    print("\n[4/5] 标签分布...")
    plot_label_distribution(data_root, output_dir)

    print("\n[5/5] 综合概览...")
    plot_overview(data_root, output_dir)

    print(f"\n全部图片已保存至: {output_dir}")
