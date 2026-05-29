"""结果可视化 — 模型评估图表

生成图表：
  1. 混淆矩阵总览 (01_confusion_matrix_all_models.png)
  2. ROC 曲线对比 (02_roc_curves_comparison.png)
  3. 融合策略对比柱状图 (03_fusion_strategy_comparison.png)
  4. 特征重要性 (04_feature_importance_top25.png)
  5. LOSO 各 subject 热力图 (05_loso_subject_heatmap.png)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
})

FUSION_LABELS = {
    'eeg_only': 'EEG only', 'ecg_only': 'ECG only', 'eda_only': 'EDA only',
    'concat': '拼接融合', 'weighted': '加权融合', 'decision': '决策融合',
}

FUSION_COLORS = {
    'eeg_only': '#2196F3', 'ecg_only': '#FF9800', 'eda_only': '#9C27B0',
    'concat': '#4CAF50', 'weighted': '#E91E63', 'decision': '#00BCD4',
}

CLF_COLORS = {'SVM': '#1f77b4', 'RF': '#ff7f0e', 'XGBoost': '#2ca02c'}

# 对照实验标签与颜色
EXPERIMENTS = {
    'FatigueSet': {'label': 'FatigueSet\nEEG+ECG+EDA', 'color': '#E91E53', 'hatch': ''},
    'sEMG': {'label': 'sEMG对照\n单通道肌电', 'color': '#4CAF50', 'hatch': '//'},
    'DriverEEG': {'label': 'Driver EEG对照\n34通道脑电', 'color': '#2196F3', 'hatch': '\\\\'},
}


def plot_confusion_matrices(results, output_dir):
    """混淆矩阵总览 — 所有融合策略 x 分类器 (千问改进版)"""
    fusions = list(results.keys())
    classifiers = list(results[list(fusions)[0]].keys())

    n_rows, n_cols = len(fusions), len(classifiers)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # 统一 colorbar 范围
    all_cms = [results[f][c]['overall']['cm'] for f in fusions for c in classifiers]
    vmax = max(cm.max() for cm in all_cms)

    for fi, fusion in enumerate(fusions):
        for ci, clf in enumerate(classifiers):
            ax = axes[fi, ci]
            cm = results[fusion][clf]['overall']['cm']
            im = ax.imshow(cm, cmap='Blues', aspect='auto', vmin=0, vmax=vmax)

            for i in range(2):
                for j in range(2):
                    weight = 'bold' if i == j else 'normal'
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                           fontsize=14, fontweight=weight,
                           color='white' if cm[i, j] > vmax / 2 else 'black')

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['非疲劳', '疲劳'])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['非疲劳', '疲劳'])

            # 仅每行最后一个子图加 colorbar
            if ci == n_cols - 1:
                plt.colorbar(im, ax=ax, shrink=0.82)

            # 单模态 vs 融合分组：前3行加横线分隔
            if fi == 2 and ci == n_cols - 1:
                ax.axhline(y=-0.5, xmin=-0.5, xmax=n_cols - 0.5,
                          color='gray', linewidth=2, linestyle='--', clip_on=False)

    # 行标签
    for fi, fusion in enumerate(fusions):
        label = FUSION_LABELS.get(fusion, fusion)
        fig.text(0.01, 1 - (fi + 0.5) / n_rows, label, ha='left', va='center', fontsize=10,
                fontweight='bold' if '融合' in str(label) else 'normal',
                color='#E91E63' if '加权' in str(label) else 'black')

    fig.suptitle('混淆矩阵总览', fontsize=14, y=1.01)
    fig.tight_layout(rect=[0.08, 0, 1, 1])
    out = Path(output_dir) / '01_confusion_matrix_all_models.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")
    return str(out)


def plot_roc_curves(results, output_dir):
    """ROC 曲线对比 (千问改进版)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    classifiers = list(results[list(results.keys())[0]].keys())

    # 子图1: weighted融合下的各分类器 ROC
    ax = axes[0]
    fusion = 'weighted' if 'weighted' in results else list(results.keys())[0]
    for clf in classifiers:
        r = results[fusion][clf]
        yt = np.array(r['y_true_all'])
        yp = np.array(r['y_prob_all'])
        if len(yp) == 0 or len(np.unique(yt)) < 2:
            continue
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(yt, yp)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=CLF_COLORS[clf],
                label=f'{clf} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'gray', lw=1.2, linestyle=':', alpha=0.7, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — {FUSION_LABELS.get(fusion, fusion)}')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    # 子图2: 各融合策略最佳模型 ROC (不含decision)
    ax = axes[1]
    for fusion in results:
        if fusion == 'decision':
            continue
        best_clf = max(classifiers, key=lambda c: results[fusion][c]['overall']['f1'])
        r = results[fusion][best_clf]
        yt = np.array(r['y_true_all'])
        yp = np.array(r['y_prob_all'])
        if len(yp) == 0 or len(np.unique(yt)) < 2:
            continue
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(yt, yp)
        roc_auc = auc(fpr, tpr)
        label_short = FUSION_LABELS.get(fusion, fusion).replace('融合', '').replace('only', '')
        ax.plot(fpr, tpr, lw=1.8, color=FUSION_COLORS[fusion],
                label=f'{label_short} AUC={roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'gray', lw=1.2, linestyle=':', alpha=0.7, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC — 各融合策略最佳模型')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(output_dir) / '02_roc_curves_comparison.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")
    return str(out)


def plot_fusion_comparison(results, output_dir):
    """融合策略对比柱状图 (千问改进版)"""
    fusions = list(results.keys())
    classifiers = list(results[list(fusions)[0]].keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = ['accuracy', 'f1', 'auc']
    titles = ['Accuracy', 'F1 Score', 'AUC']
    metric_labels = ['Acc', 'F1', 'AUC']

    for ax_idx, (metric, title, mlab) in enumerate(zip(metrics, titles, metric_labels)):
        ax = axes[ax_idx]
        x = np.arange(len(fusions))
        width = 0.25
        for ci, clf in enumerate(classifiers):
            vals = [results[f][clf]['overall'].get(metric, 0) or 0 for f in fusions]
            offset = (ci - 1) * width
            bars = ax.bar(x + offset, vals, width, label=clf, color=CLF_COLORS[clf], alpha=0.85, edgecolor='white')
            for bar, v in zip(bars, vals):
                if v > 0.02:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           f'{v:.3f}', ha='center', fontsize=7, rotation=90)
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                           f'{v:.3f}', ha='center', fontsize=6, rotation=90, color='gray')

        ax.set_xticks(x)
        ax.set_xticklabels([FUSION_LABELS.get(f, f) for f in fusions], rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(mlab)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('融合策略 x 分类器 性能对比', fontsize=14)
    fig.tight_layout()
    out = Path(output_dir) / '03_fusion_strategy_comparison.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")
    return str(out)


def plot_feature_importance(results, dataset, output_dir, top_n=25):
    """特征重要性 — 基于互信息"""
    from sklearn.feature_selection import mutual_info_classif
    weights = mutual_info_classif(dataset['X'], dataset['y'], random_state=42)
    weights = weights / (weights.max() or 1)
    dims = dataset['modality_dims']
    eeg_dim, ecg_dim, eda_dim = int(dims[0]), int(dims[1]), int(dims[2])

    # 构建特征名
    feat_names = []
    for i in range(eeg_dim):
        if i < 20: feat_names.append(f'EEG_band_power_{i}')
        elif i < 32: feat_names.append(f'EEG_ratio_{i}')
        elif i < 44: feat_names.append(f'EEG_entropy_{i}')
        else: feat_names.append(f'EEG_connectivity_{i}')
    for i in range(ecg_dim):
        hr_feats = ['HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_meanHR',
                     'HRV_LF', 'HRV_HF', 'HRV_LFHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'DFA_a1', 'DFA_a2']
        feat_names.append(hr_feats[i] if i < len(hr_feats) else f'ECG_{i}')
    for i in range(eda_dim):
        eda_feats = ['SCL_mean', 'SCL_std', 'SCL_slope', 'SCR_mean_amp', 'SCR_freq', 'SCR_sum_amp']
        feat_names.append(eda_feats[i])

    feat_names = np.array(feat_names)
    sorted_idx = np.argsort(weights)[-top_n:]
    sorted_w = weights[sorted_idx]
    sorted_names = feat_names[sorted_idx]

    # 按模态着色
    colors = []
    for name in sorted_names:
        if name.startswith('EEG'):
            colors.append('#2196F3')
        elif name.startswith('HRV') or name.startswith('DFA'):
            colors.append('#FF9800')
        else:
            colors.append('#9C27B0')

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), sorted_w, color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('互信息权重 (归一化)')
    ax.set_title(f'特征重要性 Top {top_n} (互信息)')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2196F3', label='EEG'),
                       Patch(facecolor='#FF9800', label='ECG (HRV)'),
                       Patch(facecolor='#9C27B0', label='EDA')]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

    fig.tight_layout()
    out = Path(output_dir) / '04_feature_importance_top25.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")
    return str(out)


def plot_loso_heatmap(results, output_dir):
    """LOSO 各 subject 性能热力图 (千问改进版: 支持更多subject)"""
    fusions = list(results.keys())
    classifiers = list(results[list(fusions)[0]].keys())

    sample_fusion = list(fusions)[0]
    sample_clf = list(classifiers)[0]
    subjects = sorted(results[sample_fusion][sample_clf]['per_subject'].keys())

    n_fusions = len(fusions)
    n_subj = len(subjects)
    # 根据 subject 数量动态调整宽度
    fig, axes = plt.subplots(1, n_fusions, figsize=(max(4, 1.2 * n_subj * n_fusions), 4))

    if n_fusions == 1:
        axes = [axes]

    for fi, fusion in enumerate(fusions):
        ax = axes[fi]
        data = np.zeros((len(classifiers), len(subjects)))
        for ci, clf in enumerate(classifiers):
            for si, subj in enumerate(subjects):
                data[ci, si] = results[fusion][clf]['per_subject'][subj]['f1']

        vmin = max(0.0, data.min() - 0.05)
        im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=vmin, vmax=1.0)

        for ci in range(len(classifiers)):
            for si in range(len(subjects)):
                val = data[ci, si]
                text_color = 'white' if val < (vmin + 1.0) / 2 else 'black'
                ax.text(si, ci, f'{val:.2f}', ha='center', va='center',
                       fontsize=8, fontweight='bold', color=text_color)

        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects, fontsize=8, rotation=0)
        ax.set_yticks(range(len(classifiers)))
        ax.set_yticklabels(classifiers, fontsize=9)
        ax.set_title(FUSION_LABELS.get(fusion, fusion), fontsize=11)

        if fi == n_fusions - 1:
            plt.colorbar(im, ax=ax, shrink=0.8, label='F1 Score')

    fig.suptitle('LOSO 各 Subject F1 得分', fontsize=14)
    fig.tight_layout()
    out = Path(output_dir) / '05_loso_subject_heatmap.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")
    return str(out)


def plot_experiment_comparison(results, dataset, output_dir):
    """对照实验综合对比 — FatigueSet vs sEMG vs Driver EEG (图06, 千问改进版)"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    exps = [
        ('FatigueSet\n(EEG+ECG+EDA)', '#E91E63', 0.507, 0.650),
        ('sEMG\n(单通道肌电)', '#2E7D32', 0.782, 0.715),
        ('Driver EEG\n(34通道脑电)', '#1565C0', 0.646, 0.598),
    ]

    for ax_idx, (ax, metric_idx, title) in enumerate([
        (axes[0], 2, 'F1 Score'), (axes[1], 3, 'AUC')
    ]):
        names = [e[0] for e in exps]
        vals = [e[metric_idx] for e in exps]
        colors = [e[1] for e in exps]
        bars = ax.barh(names, vals, color=colors, alpha=0.9, edgecolor='white', height=0.5)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                   f'{v:.3f}', ha='left', va='center', fontsize=12, fontweight='bold')

        # 高亮 sEMG 最佳值
        if ax_idx == 0:
            max_val = max(vals)
            max_idx = vals.index(max_val)
            ax.annotate('Best', xy=(max_val + 0.12, max_idx),
                       fontsize=11, color='#2E7D32', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2))

        ax.set_xlim(0, 1.0)
        ax.set_title(title, fontsize=13)
        ax.grid(axis='x', alpha=0.3)
        ax.tick_params(labelsize=11)

    fig.suptitle('对照实验最佳结果对比', fontsize=14, y=1.02, fontweight='bold')
    fig.tight_layout()
    out = Path(output_dir) / '06_experiment_comparison.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")
    return str(out)


def plot_optimization_comparison(results, dataset, output_dir):
    """优化前后对比 — baseline vs GA-PSO (图07, 千问改进版)"""
    json_path = Path(output_dir).parent.parent / 'data' / 'results' / 'ga_pso_loso_results.json'
    if not json_path.exists():
        print(f"  [SKIP] 未找到 GA-PSO LOSO 结果: {json_path}")
        return None

    import json
    with open(json_path) as f:
        ga_pso = json.load(f)

    baseline_f1 = results['weighted']['XGBoost']['overall']['f1']
    baseline_auc = results['weighted']['XGBoost']['overall'].get('auc', 0)
    optimized_f1 = ga_pso['overall']['f1']
    optimized_auc = ga_pso['overall'].get('auc', 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(2)
    width = 0.25

    # baseline柱 (成对)
    bars_b = ax.bar(x - width/2, [baseline_f1, baseline_auc], width,
                   label='Baseline (weighted+XGBoost)', color=['#4CAF50', '#2196F3'], alpha=0.85, edgecolor='white')
    # GA-PSO柱 (成对)
    bars_o = ax.bar(x + width/2, [optimized_f1, optimized_auc], width,
                   label='GA-PSO优化后', color='#E91E63', alpha=0.6, edgecolor='white', hatch='//')

    # 标注数值 + 下降箭头
    for i, (b, o, name) in enumerate([(baseline_f1, optimized_f1, 'F1'), (baseline_auc, optimized_auc, 'AUC')]):
        ax.text(i - width/2, b + 0.03, f'{b:.3f}', ha='center', fontsize=10, fontweight='bold', color='#1B5E20')
        ax.text(i + width/2, o + 0.03, f'{o:.3f}', ha='center', fontsize=10, fontweight='bold', color='#C62828')
        if o < b:
            ax.annotate('', xy=(i + width/2, o + 0.07), xytext=(i - width/2, b - 0.05),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax.text(i, max(b, o) + 0.08, '下降', ha='center', fontsize=9, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(['F1 Score', 'AUC'], fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('GA-PSO 优化效果 — LOSO验证\n(3-fold CV过拟合导致LOSO反而下降)', fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    out = Path(output_dir) / '07_optimization_comparison.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")
    return str(out)


def plot_semg_trend(all_data, output_dir):
    """sEMG MDF/MNF疲劳趋势图 (图08, 千问改进版)"""
    fs = 1259

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    subj_mdf = {}
    for d in all_data:
        subj = d['subject']
        if subj not in subj_mdf:
            subj_mdf[subj] = {'time': [], 'mdf': [], 'label': []}
        wins = d['windows']
        labels = d['labels']
        for i in range(len(wins)):
            seg = wins[i] if wins.ndim == 1 else wins[i]
            windowed = seg * np.hanning(len(seg))
            spec = np.abs(np.fft.rfft(windowed)) ** 2
            freqs = np.fft.rfftfreq(len(seg), d=1/fs)
            cum_power = np.cumsum(spec)
            if cum_power[-1] > 0:
                mdf = freqs[np.searchsorted(cum_power, cum_power[-1] / 2)]
                subj_mdf[subj]['mdf'].append(mdf)
                subj_mdf[subj]['label'].append(labels[i])
                subj_mdf[subj]['time'].append(i * 2.0)

    # 子图1: 平均趋势 + 置信区间
    ax = axes[0]
    # 收集所有subject的MDF到统一时间网格
    max_len = max(len(d['mdf']) for d in subj_mdf.values())
    mdf_grid = np.full((len(subj_mdf), max_len), np.nan)
    for si, (subj, data) in enumerate(sorted(subj_mdf.items())):
        if len(data['mdf']) > 5:
            mdf_grid[si, :len(data['mdf'])] = data['mdf']

    time_axis = np.arange(max_len) * 2.0
    mean_mdf = np.nanmean(mdf_grid, axis=0)
    std_mdf = np.nanstd(mdf_grid, axis=0)
    n_valid = np.sum(~np.isnan(mdf_grid), axis=0)
    sem_mdf = std_mdf / np.sqrt(np.maximum(n_valid, 1))

    ax.plot(time_axis, mean_mdf, color='#E91E63', lw=2, label='Mean MDF')
    ax.fill_between(time_axis, mean_mdf - sem_mdf, mean_mdf + sem_mdf,
                    color='#E91E63', alpha=0.2, label='+/- SEM')
    # 也画几条单subject曲线用浅色
    for si, (subj, data) in enumerate(sorted(subj_mdf.items())):
        if si < 5 and len(data['mdf']) > 5:
            ax.plot(np.arange(len(data['mdf'])) * 2.0, data['mdf'],
                   alpha=0.25, lw=0.8, color='gray')

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('MDF (Hz)', fontsize=11)
    ax.set_title('sEMG MDF 随时间变化趋势', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 子图2: 疲劳 vs 非疲劳 MDF 箱线图 + 散点
    ax = axes[1]
    fatigue_mdf, non_fatigue_mdf = [], []
    for data in subj_mdf.values():
        for m, l in zip(data['mdf'], data['label']):
            if l == 1:
                fatigue_mdf.append(m)
            else:
                non_fatigue_mdf.append(m)

    bp = ax.boxplot([non_fatigue_mdf, fatigue_mdf], labels=['非疲劳', '疲劳'],
                    patch_artist=True, widths=0.4, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=6))
    bp['boxes'][0].set_facecolor('#4CAF50')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#E91E63')
    bp['boxes'][1].set_alpha(0.7)

    # 添加散点
    np.random.seed(42)
    for i, data in enumerate([non_fatigue_mdf, fatigue_mdf]):
        jitter = np.random.normal(0, 0.04, len(data))
        ax.scatter(np.ones(len(data)) * (i + 1) + jitter, data, alpha=0.15, s=5, color='gray')

    ax.set_ylabel('MDF (Hz)', fontsize=11)
    ax.set_title('sEMG MDF: 疲劳 vs 非疲劳', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # 显示均值和p值
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(non_fatigue_mdf, fatigue_mdf, equal_var=False)
    ax.text(0.5, 0.95, f'p={p_val:.2e}', transform=ax.transAxes, fontsize=10,
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

    fig.suptitle('sEMG 肌肉疲劳趋势分析', fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = Path(output_dir) / '08_semg_fatigue_trend.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")
    return str(out)


def plot_eeg_band_comparison(output_dir):
    """疲劳前后 EEG 频带对比 (图09, 千问改进版)"""
    from step1_preprocessing import fatigueset
    base = Path(output_dir).parent.parent
    data_root = base / 'data' / 'fatigueset'

    all_data = fatigueset.load(data_root, subjects=['01', '03'], window_sec=4.0, step_sec=2.0)

    fatigue_power = {b: [] for b in ['delta', 'theta', 'alpha', 'beta', 'gamma']}
    non_fatigue_power = {b: [] for b in ['delta', 'theta', 'alpha', 'beta', 'gamma']}

    for d in all_data:
        labels = d['labels']
        for i in range(len(d['eeg'])):
            seg = d['eeg'][i]
            from scipy import signal as sg
            f, Pxx = sg.welch(seg.mean(axis=1), 256, nperseg=256*2)
            bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
            for bname, (lo, hi) in bands.items():
                mask = (f >= lo) & (f < hi)
                bp = Pxx[mask].sum()
                if labels[i] == 1:
                    fatigue_power[bname].append(bp)
                else:
                    non_fatigue_power[bname].append(bp)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    x = np.arange(len(band_names))
    width = 0.3

    nf_means = [np.mean(non_fatigue_power[b]) for b in band_names]
    nf_sems = [np.std(non_fatigue_power[b]) / np.sqrt(len(non_fatigue_power[b])) for b in band_names]
    f_means = [np.mean(fatigue_power[b]) for b in band_names]
    f_sems = [np.std(fatigue_power[b]) / np.sqrt(len(fatigue_power[b])) for b in band_names]

    bars1 = ax.bar(x - width/2, nf_means, width, yerr=nf_sems, label='非疲劳',
                   color='#4CAF50', alpha=0.85, edgecolor='white', capsize=4, error_kw={'lw': 1.5})
    bars2 = ax.bar(x + width/2, f_means, width, yerr=f_sems, label='疲劳',
                   color='#E91E63', alpha=0.85, edgecolor='white', capsize=4, error_kw={'lw': 1.5})

    # 显著性标注
    from scipy.stats import ttest_ind
    for i, band in enumerate(band_names):
        _, p = ttest_ind(non_fatigue_power[band], fatigue_power[band], equal_var=False)
        if p < 0.05:
            y_max = max(nf_means[i] + nf_sems[i], f_means[i] + f_sems[i])
            ax.plot([i - width/2, i - width/2, i + width/2, i + width/2],
                   [y_max * 1.1, y_max * 1.18, y_max * 1.18, y_max * 1.1], lw=1, color='black')
            ax.text(i, y_max * 1.2, '*' if p < 0.05 else '**', ha='center', fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in band_names], fontsize=11)
    ax.set_ylabel('平均功率 (uV^2/Hz)', fontsize=11)
    ax.set_title('EEG 频带功率: 疲劳 vs 非疲劳 (含标准误和显著性)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # 添加注释说明数据来源
    ax.text(0.5, -0.15, '* p<0.05, ** p<0.01 | 数据来自 FatigueSet subj01,03',
           transform=ax.transAxes, ha='center', fontsize=8, color='gray')

    fig.tight_layout()
    out = Path(output_dir) / '09_eeg_band_comparison.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")
    return str(out)


def generate_all(results, dataset, output_dir):
    """生成全部结果可视化图 (5+4张) + 分析说明文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("生成结果可视化 (9张图)")
    print("=" * 60)

    print("\n[1/5] 混淆矩阵...")
    plot_confusion_matrices(results, output_dir)

    print("\n[2/5] ROC 曲线...")
    plot_roc_curves(results, output_dir)

    print("\n[3/5] 融合策略对比...")
    plot_fusion_comparison(results, output_dir)

    print("\n[4/5] 特征重要性...")
    plot_feature_importance(results, dataset, output_dir)

    print("\n[5/5] LOSO Heatmap...")
    plot_loso_heatmap(results, output_dir)

    print("\n[6/9] 对照实验综合对比...")
    plot_experiment_comparison(results, dataset, output_dir)

    print("\n[7/9] 优化前后对比...")
    plot_optimization_comparison(results, dataset, output_dir)

    print(f"\n全部图片已保存至: {output_dir}")
