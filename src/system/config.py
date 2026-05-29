"""系统配置 — 方法选择均有文献依据

文献索引:
  [L1] Kakhi et al. (2024) Fatigue Monitoring Using Wearables and AI Survey
       → 多模态融合有效性、CNN-LSTM、Transformer 架构综述
  [L2] Zhou et al. (2024) Bayes-gcForest Brain Fatigue Recognition
       → gcForest 在小样本场景优于深度学习，贝叶斯优化超参
  [L3] Corvini & Conforto (2022) sEMG MDF/MNF Simulation
       → MDF 优于 MNF，AR Burg(3阶)优于 Welch
  [L4] Wang et al. (2023) EEG+ECG Multi-Sensor Fusion
       → EEG+ECG融合+LightGBM，融合优于单模态
  [L5] Peivandi et al. (2023) Deep Learning Multi-Level Driver Fatigue
       → GAN+CNN+Type-2 Fuzzy 多级疲劳分类
  [L7] 课程材料: SVM/GA/PSO — 基础算法实现
"""

# ============================================================
# 信号处理参数
# ============================================================
SIGNAL_PROCESSING = {
    'ECG': {
        'bandpass': [0.5, 45],
        'notch': 50,
        'r_wave_algorithm': 'pan_tompkins',
    },
    'EEG': {
        'bandpass': [0.5, 45],
        'notch': 50,
        'ica': True,
    },
    'sEMG': {
        'bandpass': [20, 450],       # [L3] 4阶 Butterworth，官方代码规格
        'filter_order': 4,
        'psd_method': 'burg',         # [L3] AR Burg(3阶)优于 Welch
        'burg_order': 3,
        'window_sec': 4,              # [L3] 4秒滑动窗口
        'step_sec': 2,                # 50% 重叠
    },
    'EDA': {
        'lowpass': 1,
        'decompose': 'cvxopt',        # SCL/SCR 凸优化分解
    }
}

# ============================================================
# 特征提取参数 — [L2][L3] 特征集设计
# ============================================================
FEATURES = {
    'ECG_DIM': 12,
    'EEG_DIM': 74,
    'EDA_DIM': 6,
    'sEMG_DIM': 20,
    'sEMG': {
        'time_domain': ['MAV', 'RMS', 'ZC', 'SSC', 'WL', 'VAR',
                        'IEMG', 'SSI', 'SKEW', 'KURT'],  # 10维
        'freq_domain': ['MDF', 'MNF', 'PKF', 'FR', 'TTP',
                        'BAND_0_30', 'BAND_30_60', 'BAND_60_100',
                        'BAND_100_200', 'BAND_200_450'],   # 10维
    },
    'EEG': {
        'bands': {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 13],
                  'beta': [13, 30], 'gamma': [30, 45]},
        'entropy': ['multiscale', 'permutation', 'sample'],
        'connectivity': 'plv',        # 相位锁定值
    },
    'ECG': {
        'hrv_time': ['SDNN', 'RMSSD', 'pNN50', 'HR_mean'],
        'hrv_freq': ['LF', 'HF', 'LF_HF'],
        'nonlinear': ['SD1', 'SD2', 'SD1_SD2', 'DFA_alpha', 'DFA_alpha2'],
    }
}

# ============================================================
# 融合策略 — [L1][L2][L4] 三种融合级别
# ============================================================
FUSION = {
    'methods': [
        'concat',       # 特征拼接（基线方法）[L2]
        'weighted',     # 互信息加权融合（核心方法）[L1]
        'decision',     # 决策级投票融合 [L1]
    ],
    'weight_method': 'mutual_info',   # [L2] 互信息 vs ReliefF
    'decision_meta': 'rf',            # 元分类器
}

# ============================================================
# 分类器 — [L2][L4][L7] 文献推荐 + 课程实现
# ============================================================
CLASSIFIERS = {
    'SVM': {
        'enabled': True,
        'kernel': 'rbf',              # [L7] 课程SVM示例 + RBF核
        'C': 1.0,
        'gamma': 'scale',
        'reference': '课程材料 SVM.py + [L4] MTFN-SAM 使用SVM',
        'note': '全量18k样本较慢，快速测试时启用',
    },
    'RF': {
        'enabled': True,
        'n_estimators': 200,
        'max_depth': 15,
        'reference': '[L1] 随机森林是多模态疲劳检测常用基线',
        'note': '稳健基线，不容易过拟合',
    },
    'XGBoost': {
        'enabled': True,
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'reference': '[L1] XGBoost在表格特征上通常优于RF',
        'note': '强分类器，上限高但需调参',
    },
    'gcForest': {
        'enabled': True,
        'n_cascade_forests': 4,
        'n_trees': 100,
        'tolerance': 0.0,
        'reference': '[L2] Zhou et al. (2024) gcForest在小样本(FatigueSet 12人)优于深度学习',
        'note': '深度森林：无需大量数据，自动决定级联层数',
    },
}

# ============================================================
# 优化算法 — [L7] 课程GA/PSO + [L2] 贝叶斯优化
# ============================================================
OPTIMIZATION = {
    'GA': {
        'enabled': True,
        'pop_size': 100,
        'generations': 500,
        'crossover_rate': 0.9,
        'mutation_rate': 0.01,
        'reference': '课程材料 GA.py',
        'note': '二进制编码特征选择',
    },
    'PSO': {
        'enabled': True,
        'n_particles': 50,
        'iterations': 100,
        'reference': '课程材料 PSO.py',
        'note': '连续参数优化（学习率、树深度等）',
    },
}

# ============================================================
# 评估参数
# ============================================================
EVALUATION = {
    'cv': 'loso',                      # Leave-One-Subject-Out
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
    'quick_test_subjects': 3,          # 快速测试用3人
    'full_subjects': 12,               # FatigueSet 12人
}
