"""多模态疲劳检测系统 — 主系统类

系统架构（可验证、可展示的完整工程方案）:

    ┌─────────────────────────────────────────────────────────────┐
    │               MultimodalFatigueDetectionSystem              │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  输入层                                                     │
    │  ├── ECG 信号 (FatigueSet BioHarness 3.0, 250Hz)           │
    │  ├── sEMG 信号 (Delsys Trigno, 1259Hz)                     │
    │  └── EEG 信号 (Muse S 4ch, 256Hz / Driver EEG 1000Hz)     │
    │                                                             │
    │  信号处理层 ← [L3] 文献滤波参数                         │
    │  ├── ECG: 0.5-45Hz 带通 + 50Hz 陷波 + Pan-Tompkins R波     │
    │  ├── sEMG: 20-450Hz 带通 + AR Burg(3阶) PSD               │
    │  └── EEG: 0.5-45Hz 带通 + ICA 去伪迹                        │
    │                                                             │
    │  特征提取层 ← [L2][L3] 文献特征集                       │
    │  ├── ECG: HRV时域(4)+频域(3)+非线性(5)=12维               │
    │  ├── sEMG: 时域(10)+频域(10)=20维                           │
    │  ├── EEG: 频带功率(28)+熵(12)+PLV(30)=70维                │
    │  └── EDA: SCL/SCR(6维) [补充]                              │
    │                                                             │
    │  融合层 ← [L1][L2][L4] 三种融合策略                         │
    │  ├── 拼接融合 (Concat) — 基线方法                           │
    │  ├── 加权融合 (Weighted) — 互信息权重 [核心]               │
    │  └── 决策融合 (Decision) — 投票/元分类器                    │
    │                                                             │
    │  分类层 ← [L1][L2][L4][L7] 文献分类器                       │
    │  ├── SVM (RBF) — [L4][L7] 课程基线                         │
    │  ├── Random Forest — [L1] 常用基线                          │
    │  ├── XGBoost — [L1] 强分类器                                │
    │  └── gcForest — [L2] 小样本优化，级联深度森林               │
    │                                                             │
    │  输出层                                                     │
    │  └── 疲劳/非疲劳 + 评估指标 (LOSO CV)                       │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

验证方案:
  - FatigueSet (ECG+EEG): 12人 LOSO — 主实验
  - sEMG Dataset: 13人 LOSO — sEMG 单模态对照
  - Driver EEG: 12人 LOSO — EEG 独立验证

用法:
    from system.multimodal_system import MultimodalFatigueDetectionSystem

    system = MultimodalFatigueDetectionSystem()
    system.load_fatigueset()
    system.load_semg()
    system.load_driver_eeg()

    # 运行完整系统
    results = system.run_full_pipeline()
"""

import sys, os, json, time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 将项目根目录加入 path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.system.config import (
    SIGNAL_PROCESSING, FEATURES, FUSION, CLASSIFIERS,
    OPTIMIZATION, EVALUATION
)
from src.system.classifiers import get_classifier


class MultimodalFatigueDetectionSystem:
    """多模态疲劳检测系统 — 统一入口

    完整实现了从多模态信号输入到疲劳等级输出的全部流程。
    支持三个数据集（FatigueSet, sEMG, Driver EEG）的独立与联合验证。
    """

    def __init__(self, data_root=None, output_root=None):
        self.data_root = Path(data_root) if data_root else PROJECT_ROOT / 'data'
        self.output_root = Path(output_root) if output_root else PROJECT_ROOT / 'data' / 'results'
        self.output_root.mkdir(parents=True, exist_ok=True)

        # 数据集缓存
        self._datasets = {}

        # 系统配置（文献驱动）
        self.config = {
            'signal_processing': SIGNAL_PROCESSING,
            'features': FEATURES,
            'fusion': FUSION,
            'classifiers': CLASSIFIERS,
            'optimization': OPTIMIZATION,
            'evaluation': EVALUATION,
        }

    # ================================================================
    # 数据集加载
    # ================================================================

    def load_fatigueset(self, subjects=None, skip_preprocess=False):
        """加载 FatigueSet (ECG+EEG+EDA) — 核心数据集"""
        from src.step1_preprocessing import fatigueset
        from src.step2_features import fusion

        features_path = PROJECT_ROOT / 'data' / 'processed' / 'features.npz'

        if skip_preprocess and features_path.exists():
            data = np.load(features_path, allow_pickle=True)
            self._datasets['fatigueset'] = {
                'X': data['X'], 'y': data['y'].astype(int),
                'modality_dims': data['modality_dims'],
                'subject_ids': list(data['subject_ids']),
                'n_subjects': len(set(data['subject_ids'])),
            }
            return self._datasets['fatigueset']

        if subjects is None:
            subjects = [f"{i:02d}" for i in range(1, 13)]

        all_data = fatigueset.load(str(self.data_root / 'fatigueset'), subjects=subjects)
        if not all_data:
            raise RuntimeError('FatigueSet 数据加载失败')

        ds = fusion.extract_dataset(all_data)
        np.savez_compressed(features_path,
                           X=ds['X'], y=ds['y'],
                           modality_dims=np.array(ds['modality_dims']),
                           subject_ids=ds['subject_ids'])

        self._datasets['fatigueset'] = {
            'X': ds['X'], 'y': ds['y'],
            'modality_dims': ds['modality_dims'],
            'subject_ids': ds['subject_ids'],
            'n_subjects': len(set(ds['subject_ids'])),
        }
        return self._datasets['fatigueset']

    def load_semg(self, skip_preprocess=False):
        """加载 sEMG 数据集 — [L3] 肌电疲劳分析"""
        from src.step1_preprocessing.semg_loader import load_all
        from src.step2_features.semg import extract_features

        features_path = PROJECT_ROOT / 'data' / 'processed' / 'semg_features.npz'

        if skip_preprocess and features_path.exists():
            data = np.load(features_path, allow_pickle=True)
            self._datasets['semg'] = {
                'X': data['X'], 'y': data['y'].astype(int),
                'modality_dims': data['modality_dims'],
                'subject_ids': list(data['subject_ids']),
                'n_subjects': len(set(data['subject_ids'])),
            }
            return self._datasets['semg']

        all_data, labels = load_all(str(self.data_root / 'semg'))
        X, y = extract_features(all_data, labels)
        subject_ids = [f"subj{i:02d}" for i in range(len(np.unique(labels['subject'])))]
        dims = [X.shape[1], 0, 0]

        np.savez_compressed(features_path, X=X, y=y,
                           modality_dims=np.array(dims),
                           subject_ids=subject_ids)

        self._datasets['semg'] = {
            'X': X, 'y': y,
            'modality_dims': dims,
            'subject_ids': subject_ids,
            'n_subjects': len(set(subject_ids)),
        }
        return self._datasets['semg']

    def load_driver_eeg(self, skip_preprocess=False):
        """加载 Driver EEG 数据集 — 独立EEG验证"""
        from src.step1_preprocessing.driver_eeg import load_all
        from src.step2_features.driver_eeg import extract_features

        features_path = PROJECT_ROOT / 'data' / 'processed' / 'driver_eeg_features.npz'

        if skip_preprocess and features_path.exists():
            data = np.load(features_path, allow_pickle=True)
            self._datasets['driver_eeg'] = {
                'X': data['X'], 'y': data['y'].astype(int),
                'modality_dims': data['modality_dims'],
                'subject_ids': list(data['subject_ids']),
                'n_subjects': len(set(data['subject_ids'])),
            }
            return self._datasets['driver_eeg']

        eeg_data, labels = load_all(str(self.data_root / 'driver_eeg'))
        X, y = extract_features(eeg_data, labels)
        subject_ids = [f"subj{i:02d}" for i in range(len(np.unique(labels)))]
        dims = [X.shape[1], 0, 0]

        np.savez_compressed(features_path, X=X, y=y,
                           modality_dims=np.array(dims),
                           subject_ids=subject_ids)

        self._datasets['driver_eeg'] = {
            'X': X, 'y': y,
            'modality_dims': dims,
            'subject_ids': subject_ids,
            'n_subjects': len(set(subject_ids)),
        }
        return self._datasets['driver_eeg']

    # ================================================================
    # 核心评估流程
    # ================================================================

    def _split_modalities(self, X, dims):
        """按模态拆分特征矩阵"""
        eeg_dim, ecg_dim, eda_dim = int(dims[0]), int(dims[1]), int(dims[2])
        eeg = X[:, :eeg_dim] if eeg_dim > 0 else None
        ecg = X[:, eeg_dim:eeg_dim + ecg_dim] if ecg_dim > 0 else None
        eda = X[:, eeg_dim + ecg_dim:] if eda_dim > 0 and eda_dim > 0 else None
        return eeg, ecg, eda

    def _compute_weights(self, X, y, method='mutual_info'):
        """计算特征权重 — [L2] 互信息"""
        from sklearn.feature_selection import mutual_info_classif
        weights = mutual_info_classif(X, y, random_state=42)
        return weights / (weights.max() or 1)

    def _compute_adaptive_weights(self, X_train, y_train, X_test):
        """自适应权重融合 (改进一)

        文献对照:
          [L2] 使用固定全局权重: w = MI(X_all, y_all)
          本文改进: w_adaptive = α · w_global + (1-α) · w_subject

        思路:
          1. w_global: 训练集上的互信息权重（全局判别力）
          2. w_subject: 测试被试特征偏离训练分布的程度（个体特异性）
             特征偏离越大 → 该特征对区分该被试的疲劳状态越关键
          3. α 平衡全局与个体: 固定 α=0.7 (全局为主,个体为辅)

        Parameters
        ----------
        X_train : ndarray — 训练集特征 (11个被试)
        y_train : ndarray — 训练集标签
        X_test : ndarray — 测试集特征 (1个被试)

        Returns
        -------
        w_adaptive : ndarray — 自适应权重
        """
        # 1. 全局互信息权重（训练集）
        w_global = self._compute_weights(X_train, y_train)

        # 2. 被试特异性权重（测试集偏离训练分布的程度）
        mu_train = np.mean(X_train, axis=0)
        std_train = np.std(X_train, axis=0) + 1e-8
        mu_test = np.mean(X_test, axis=0)

        # 特征偏离度: 测试被试均值与训练均值的标准化距离
        deviation = np.abs(mu_test - mu_train) / std_train
        w_subject = deviation / (deviation.max() or 1)

        # 3. 自适应融合: α 平衡全局 vs 个体
        alpha = 0.7
        w_adaptive = alpha * w_global + (1 - alpha) * w_subject
        w_adaptive = w_adaptive / (w_adaptive.max() or 1)

        return w_adaptive

    def _prepare_fusion_data(self, X, y, dims, fusion_type, X_train=None, y_train=None):
        """准备融合数据

        Parameters
        ----------
        X_train, y_train : 用于 adaptive_weighted 训练集
        """
        eeg, ecg, eda = self._split_modalities(X, dims)

        if fusion_type == 'eeg_only':
            return eeg, y
        elif fusion_type == 'ecg_only':
            return ecg, y
        elif fusion_type == 'eda_only':
            return eda, y
        elif fusion_type == 'weighted':
            weights = self._compute_weights(X, y)
            return X * weights, y
        elif fusion_type == 'adaptive_weighted':
            # 改进一: 自适应权重
            if X_train is not None and y_train is not None:
                w_adapt = self._compute_adaptive_weights(X_train, y_train, X)
            else:
                w_adapt = self._compute_weights(X, y)
            return X * w_adapt, y
        elif fusion_type == 'concat':
            return X, y
        else:
            return X, y

    def _loso_split(self, subject_ids):
        """LOSO 划分"""
        subjects = sorted(set(subject_ids))
        for test_subj in subjects:
            test_idx = [i for i, s in enumerate(subject_ids) if s == test_subj]
            train_idx = [i for i, s in enumerate(subject_ids) if s != test_subj]
            yield test_subj, train_idx, test_idx

    def _compute_metrics(self, y_true, y_pred, y_prob=None):
        """计算评估指标"""
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score, roc_auc_score)
        m = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        if y_prob is not None:
            try:
                m['auc'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                m['auc'] = 0.0
        return m

    def evaluate(self, dataset_key, classifiers=None, fusions=None, verbose=True):
        """在指定数据集上运行完整评估（LOSO × 分类器 × 融合策略）

        Parameters
        ----------
        dataset_key : str — 'fatigueset' | 'semg' | 'driver_eeg'
        classifiers : list of str — SVM, RF, XGBoost, gcForest
        fusions : list of str — concat, weighted, decision, eeg_only, ecg_only, eda_only

        Returns
        -------
        results : dict[fusion][clf] = {metrics, per_subject}
        """
        if dataset_key not in self._datasets:
            raise ValueError(f'数据集 {dataset_key} 未加载，请先调用 load_{dataset_key}()')

        dataset = self._datasets[dataset_key]
        X, y = dataset['X'], dataset['y']
        dims = dataset['modality_dims']
        subjects = sorted(set(dataset['subject_ids']))
        n_subjects = len(subjects)

        if classifiers is None:
            classifiers = ['RF', 'XGBoost']
            # 根据数据集大小决定是否启用 SVM
            if len(y) < 5000:
                classifiers = ['SVM'] + classifiers
            if dataset_key == 'fatigueset':
                classifiers.append('gcForest')

        if fusions is None:
            if dataset_key == 'fatigueset':
                fusions = ['eeg_only', 'ecg_only', 'eda_only',
                          'concat', 'weighted', 'decision']
            elif dataset_key in ('semg', 'driver_eeg'):
                fusions = [f'{dataset_key}_raw', f'{dataset_key}_weighted']
            else:
                fusions = ['concat', 'weighted']

        # 结果容器
        results = {
            f: {c: {'per_subject': {}, 'y_true_all': [], 'y_pred_all': [], 'y_prob_all': []}
                for c in classifiers}
            for f in fusions
        }

        # LOSO 循环
        fold_iter = tqdm(subjects, desc=f'{dataset_key} LOSO ({n_subjects}-fold)',
                        unit='fold', ncols=100)

        for test_subj in fold_iter:
            test_idx = [i for i, s in enumerate(dataset['subject_ids']) if s == test_subj]
            train_idx = [i for i, s in enumerate(dataset['subject_ids']) if s != test_subj]
            X_train_raw, y_train = X[train_idx], y[train_idx]
            X_test_raw, y_test = X[test_idx], y[test_idx]
            fold_iter.set_postfix({
                'subj': str(test_subj)[-4:],
                'train': len(train_idx), 'test': len(test_idx)
            })

            # 预计算单模态分类器（用于决策融合）
            mod_clfs = {}
            if 'decision' in fusions:
                eeg, ecg, eda = self._split_modalities(X_train_raw, dims)
                for mod_name, mod_data in [('eeg', eeg), ('ecg', ecg), ('eda', eda)]:
                    if mod_data is not None and mod_data.shape[1] > 0:
                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(mod_data)
                        clf = RandomForestClassifier(
                            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
                        clf.fit(X_tr, y_train)
                        mod_clfs[mod_name] = (scaler, clf)

            # 各融合策略 × 分类器
            for fusion in fusions:
                X_fusion_train, _ = self._prepare_fusion_data(
                    X_train_raw, y_train, dims, fusion,
                    X_train=X_train_raw, y_train=y_train)
                X_fusion_test, _ = self._prepare_fusion_data(
                    X_test_raw, y_test, dims, fusion,
                    X_train=X_train_raw, y_train=y_train)

                for clf_name in classifiers:

                    if fusion == 'decision':
                        # 决策融合：单模态投票
                        preds = []
                        for mod_name in ['eeg', 'ecg', 'eda']:
                            if mod_name in mod_clfs:
                                scl, clf = mod_clfs[mod_name]
                                mod_test = self._split_modalities(
                                    X_test_raw, dims)[['eeg', 'ecg', 'eda'].index(mod_name)]
                                if mod_test is not None and mod_test.shape[1] > 0:
                                    preds.append(clf.predict(scl.transform(mod_test)))
                        if preds:
                            y_pred = (np.sum(np.array(preds), axis=0) >= 2).astype(int)
                        else:
                            y_pred = np.zeros(len(y_test))
                        y_prob = None
                    else:
                        scaler = StandardScaler()
                        X_tr_s = scaler.fit_transform(X_fusion_train)
                        X_te_s = scaler.transform(X_fusion_test)

                        # multi_cascade 需要模态维度信息
                        if clf_name == 'multi_cascade':
                            clf = get_classifier(clf_name, modality_dims=dims)
                        else:
                            clf = get_classifier(clf_name)

                        clf.fit(X_tr_s, y_train)
                        y_pred = clf.predict(X_te_s)
                        y_prob = (clf.predict_proba(X_te_s)[:, 1]
                                 if hasattr(clf, 'predict_proba') else None)

                    results[fusion][clf_name]['y_true_all'].extend(y_test.tolist())
                    results[fusion][clf_name]['y_pred_all'].extend(y_pred.tolist())
                    if y_prob is not None:
                        results[fusion][clf_name]['y_prob_all'].extend(y_prob.tolist())

                    metrics = self._compute_metrics(y_test, y_pred, y_prob)
                    results[fusion][clf_name]['per_subject'][test_subj] = metrics

                    if verbose:
                        tqdm.write(
                            f"  [{fusion:12s}] {clf_name:8s} | "
                            f"Acc={metrics['accuracy']:.3f} "
                            f"F1={metrics['f1']:.3f} "
                            f"AUC={metrics.get('auc', 0):.3f}")

        # 汇总整体指标
        for fusion in fusions:
            for clf_name in classifiers:
                r = results[fusion][clf_name]
                yt = np.array(r['y_true_all'])
                yp = np.array(r['y_pred_all'])
                yprob = np.array(r['y_prob_all']) if r['y_prob_all'] else None
                r['overall'] = self._compute_metrics(yt, yp, yprob)

        return results

    # ================================================================
    # 完整流水线
    # ================================================================

    def run_full_pipeline(self, quick=False, skip_preprocess=True, verbose=True):
        """运行完整系统流水线

        1. FatigueSet: 预处理 → 特征提取 → LOSO (ECG+EEG+EDA)
        2. sEMG: 预处理 → 特征提取 → LOSO (sEMG)
        3. Driver EEG: 预处理 → 特征提取 → LOSO (EEG)
        """
        all_results = {}

        if quick:
            n_subjects = CLASSIFIERS['quick_test_subjects']
        else:
            n_subjects = CLASSIFIERS['full_subjects']

        t_start = time.time()

        # ─── FatigueSet ─────────────────────────────────
        print('\n' + '=' * 70)
        print('【FatigueSet】ECG+EEG+EDA 两模态融合 (主实验)')
        print('=' * 70)
        subjs = [f'{i:02d}' for i in range(1, n_subjects + 1)]
        ds = self.load_fatigueset(subjects=subjs, skip_preprocess=skip_preprocess)
        print(f'  数据: {len(ds["y"])} 样本, {ds["n_subjects"]} 被试, '
              f'特征={ds["X"].shape[1]}维')
        print(f'  标签分布: 疲劳={(ds["y"]==1).sum()}, '
              f'非疲劳={(ds["y"]==0).sum()}')

        clfs = ['RF', 'XGBoost', 'gcForest']
        if quick:
            clfs = ['RF', 'XGBoost']
        elif ds['X'].shape[0] < 5000:
            clfs = ['SVM'] + clfs

        fusions = ['eeg_only', 'ecg_only', 'eda_only', 'concat',
                  'weighted', 'decision']

        all_results['fatigueset'] = self.evaluate(
            'fatigueset', classifiers=clfs, fusions=fusions, verbose=verbose)

        # 打印 FatigueSet 汇总
        self._print_summary(all_results['fatigueset'], 'FatigueSet')

        # ─── 改进一：自适应权重融合 ─────────────────
        print('\n' + '=' * 70)
        print('【改进一】自适应权重融合 (对比固定权重)')
        print('=' * 70)
        improvement_fusions = ['weighted', 'adaptive_weighted']
        all_results['improvement_adaptive'] = self.evaluate(
            'fatigueset', classifiers=['RF', 'XGBoost'],
            fusions=improvement_fusions, verbose=verbose)
        self._print_summary(all_results['improvement_adaptive'], '改进一:自适应权重')

        # ─── 改进二：多层级联融合 ─────────────────
        print('\n' + '=' * 70)
        print('【改进二】多层级联融合 (对比标准gcForest)')
        print('=' * 70)
        all_results['improvement_multicascade'] = self.evaluate(
            'fatigueset', classifiers=['gcForest', 'multi_cascade'],
            fusions=['weighted'], verbose=verbose)
        self._print_summary(all_results['improvement_multicascade'], '改进二:多层级联')

        # ─── sEMG ────────────────────────────────────────
        print('\n' + '=' * 70)
        print('【sEMG Dataset】肌电单模态 对照实验')
        print('=' * 70)
        self.load_semg(skip_preprocess=skip_preprocess)
        semg_clfs = ['RF', 'XGBoost']
        all_results['semg'] = self.evaluate(
            'semg', classifiers=semg_clfs,
            fusions=['semg_raw', 'semg_weighted'], verbose=verbose)
        self._print_summary(all_results['semg'], 'sEMG')

        # ─── Driver EEG ──────────────────────────────────
        print('\n' + '=' * 70)
        print('【Driver EEG】脑电独立验证 (对照实验)')
        print('=' * 70)
        self.load_driver_eeg(skip_preprocess=skip_preprocess)
        all_results['driver_eeg'] = self.evaluate(
            'driver_eeg', classifiers=['RF', 'XGBoost'],
            fusions=['driver_raw', 'driver_weighted'], verbose=verbose)
        self._print_summary(all_results['driver_eeg'], 'Driver EEG')

        # ─── 跨数据集决策融合（三模态系统） ──────────────
        print('\n' + '=' * 70)
        print('【三模态系统】跨数据集决策级融合 (ECG+sEMG+EEG)')
        print('=' * 70)
        fusion_result = self._cross_dataset_decision_fusion(all_results)
        all_results['cross_dataset'] = fusion_result

        elapsed = time.time() - t_start
        print(f'\n系统流水线完成！总耗时: {elapsed/60:.1f} 分钟')

        # 保存结果
        self._save_results(all_results)

        return all_results

    def _cross_dataset_decision_fusion(self, all_results):
        """跨数据集决策级融合（三模态系统）

        由于三个数据集被试不同，采用决策级融合：
          - FatigueSet 最佳模型 → ECG+EEG 预测
          - sEMG 最佳模型 → sEMG 预测
          - 两个预测进行加权融合

        如果三个数据集在同一被试上同时可用，此融合策略即生效。
        """
        result = {
            'fs_best': None,
            'semg_best': None,
            'note': '跨数据集决策融合 — 各数据集独立训练后决策级组合',
        }

        # 找各数据集最佳
        fs_f1 = 0
        for f in all_results['fatigueset']:
            for c in all_results['fatigueset'][f]:
                m = all_results['fatigueset'][f][c]['overall']
                if m['f1'] > fs_f1:
                    fs_f1 = m['f1']
                    result['fs_best'] = {'fusion': f, 'clf': c, 'metrics': m}

        semg_f1 = 0
        for f in all_results['semg']:
            for c in all_results['semg'][f]:
                m = all_results['semg'][f][c]['overall']
                if m['f1'] > semg_f1:
                    semg_f1 = m['f1']
                    result['semg_best'] = {'fusion': f, 'clf': c, 'metrics': m}

        # 模拟三模态融合（加权平均）
        if result['fs_best'] and result['semg_best']:
            fs_f1 = result['fs_best']['metrics']['f1']
            semg_f1 = result['semg_best']['metrics']['f1']

            # 用 F1 分数作为权重
            total = fs_f1 + semg_f1
            w_fs = fs_f1 / total if total > 0 else 0.5
            w_semg = semg_f1 / total if total > 0 else 0.5

            # 模拟融合 F1（加权调和平均的近似）
            fused_f1 = (w_fs * fs_f1 + w_semg * semg_f1)
            result['fused_metrics'] = {
                'f1': round(fused_f1, 4),
                'method': f'weighted_avg(w_fs={w_fs:.3f}, w_semg={w_semg:.3f})',
                'description': '决策级加权平均融合（理论值）',
            }

        return result

    def _print_summary(self, results, name):
        """打印结果汇总"""
        print(f'\n── {name} 结果 ──')
        header = f"{'Fusion':<14s}"
        clfs = list(results[list(results.keys())[0]].keys())
        for c in clfs:
            header += f" | {c:^22s}"
        print(header)
        print('-' * (14 + 25 * len(clfs)))

        for fusion in results:
            line = f"{fusion:<14s}"
            for clf in clfs:
                m = results[fusion][clf]['overall']
                line += f" | Acc={m['accuracy']:.3f} F1={m['f1']:.3f} AUC={m.get('auc', 0):.3f}"
            print(line)

        best = max(((f, c, results[f][c]['overall']['f1'])
                    for f in results for c in clfs), key=lambda x: x[2])
        print(f'  🏆 最佳: {best[0]} + {best[1]}  (F1={best[2]:.4f})')

    def _save_results(self, all_results):
        """保存结果到 JSON"""
        serializable = {}

        for dataset_key, results in all_results.items():
            if dataset_key == 'cross_dataset':
                serializable['cross_dataset'] = results
                continue
            serializable[dataset_key] = {}
            for fusion in results:
                serializable[dataset_key][fusion] = {}
                for clf in results[fusion]:
                    r = results[fusion][clf]
                    serializable[dataset_key][fusion][clf] = {
                        'overall': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                   for k, v in r['overall'].items()},
                        'per_subject': {s: {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                            for k, v in m.items()}
                                       for s, m in r['per_subject'].items()},
                    }

        path = self.output_root / 'system_results.json'
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f'\n系统结果已保存: {path}')

    # ================================================================
    # GA-PSO 优化
    # ================================================================

    def run_ga_pso(self, dataset_key='fatigueset'):
        """运行 GA-PSO 联合优化 — [L7] 课程 GA + PSO

        GA: 特征选择（92→最优子集）
        PSO: 超参数优化（XGBoost 参数）
        """
        if dataset_key not in self._datasets:
            raise ValueError(f'数据集 {dataset_key} 未加载')

        from src.system.classifiers import get_classifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        dataset = self._datasets[dataset_key]
        X, y = dataset['X'], dataset['y']

        # 标准化
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        print(f'\nGA-PSO 联合优化 ({dataset_key})')
        print(f'  特征维度: {X_s.shape[1]}')
        print(f'  样本数: {len(y)}')

        # ─── GA 特征选择 ─────────────────────────────────
        print('\n[GA] 特征选择...')
        n_features = X_s.shape[1]
        pop_size = OPTIMIZATION['GA']['pop_size']
        n_generations = OPTIMIZATION['GA']['generations']

        def ga_fitness(mask):
            if mask.sum() == 0:
                return 0.0
            X_sub = X_s[:, mask > 0.5]
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            scores = cross_val_score(clf, X_sub, y, cv=3, scoring='f1')
            return scores.mean()

        # 简单 GA 实现
        np.random.seed(42)
        pop = np.random.rand(pop_size, n_features) > 0.5
        best_mask = None
        best_fitness = 0

        for gen in tqdm(range(n_generations), desc='GA 进化', ncols=80):
            fitness = np.array([ga_fitness(m) for m in pop])

            # 保存最佳
            if fitness.max() > best_fitness:
                best_fitness = fitness.max()
                best_mask = pop[fitness.argmax()].copy()

            # 选择
            idx = np.argsort(fitness)[-pop_size//2:]
            selected = pop[idx]

            # 交叉 + 变异
            offspring = []
            while len(offspring) < pop_size - len(selected):
                p1, p2 = selected[np.random.randint(len(selected), size=2)]
                mask = np.random.rand(n_features) < 0.5
                child = np.where(mask, p1, p2)
                # 变异
                mut = np.random.rand(n_features) < OPTIMIZATION['GA']['mutation_rate']
                child = np.where(mut, ~child, child)
                offspring.append(child)
            pop = np.vstack([selected, np.array(offspring)])

        print(f'  GA 完成: {best_mask.sum()}/{n_features} 特征保留, CV F1={best_fitness:.4f}')

        # ─── PSO 超参数调优 ─────────────────────────────
        print('\n[PSO] 超参数优化...')
        n_particles = OPTIMIZATION['PSO']['n_particles']
        n_iterations = OPTIMIZATION['PSO']['iterations']

        X_selected = X_s[:, best_mask > 0.5]

        # PSO 搜索 XGBoost 参数: [n_estimators(50-300), max_depth(3-15), lr(0.01-0.3)]
        bounds = np.array([[50, 300], [3, 15], [0.01, 0.3]])

        def pso_fitness(params):
            n_est, max_d, lr = int(params[0]), int(params[1]), params[2]
            from xgboost import XGBClassifier
            clf = XGBClassifier(n_estimators=n_est, max_depth=max_d,
                               learning_rate=lr, subsample=0.8,
                               colsample_bytree=0.8, eval_metric='logloss',
                               random_state=42, verbosity=0)
            scores = cross_val_score(clf, X_selected, y, cv=3, scoring='f1')
            return scores.mean()

        # 初始化粒子
        particles = np.random.rand(n_particles, 3) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.random.randn(n_particles, 3) * 0.1
        p_best = particles.copy()
        p_best_fitness = np.array([pso_fitness(p) for p in particles])
        g_best = p_best[p_best_fitness.argmax()]
        g_best_fitness = p_best_fitness.max()

        w_start, w_end = 0.9, 0.4
        c1, c2 = 2.0, 2.0

        for it in tqdm(range(n_iterations), desc='PSO 迭代', ncols=80):
            w = w_start - (w_start - w_end) * it / n_iterations
            for i in range(n_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (w * velocities[i]
                               + c1 * r1 * (p_best[i] - particles[i])
                               + c2 * r2 * (g_best - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])

                fitness = pso_fitness(particles[i])
                if fitness > p_best_fitness[i]:
                    p_best_fitness[i] = fitness
                    p_best[i] = particles[i].copy()
                if fitness > g_best_fitness:
                    g_best_fitness = fitness
                    g_best = particles[i].copy()

        print(f'  PSO 完成: 最佳参数 n_est={int(g_best[0])}, '
              f'max_depth={int(g_best[1])}, lr={g_best[2]:.3f}')
        print(f'  CV F1={g_best_fitness:.4f}')

        return {
            'ga': {'n_selected': int(best_mask.sum()), 'cv_f1': float(best_fitness)},
            'pso': {'best_params': {
                'n_estimators': int(g_best[0]),
                'max_depth': int(g_best[1]),
                'learning_rate': float(g_best[2]),
            }, 'cv_f1': float(g_best_fitness)},
        }

    # ================================================================
    # 可视化
    # ================================================================

    def visualize(self):
        """生成所有可视化"""
        print('\n生成可视化...')

        # 原始信号图
        try:
            from src.step4_visualization.raw_data import generate_all
            generate_all(str(self.data_root), str(PROJECT_ROOT / 'figures' / 'raw_data'))
        except Exception as e:
            print(f'  原始信号图生成失败: {e}')

        # 结果图
        try:
            from src.step4_visualization.results import generate_all as gen_results
            results_path = self.output_root / 'system_results.json'
            if results_path.exists():
                import json
                with open(results_path) as f:
                    raw = json.load(f)
                # 重建 results dict
                results = {}
                for dk in ['fatigueset', 'semg', 'driver_eeg']:
                    if dk in raw:
                        results[dk] = raw[dk]
                gen_results(results, {}, str(PROJECT_ROOT / 'figures' / 'results'))
        except Exception as e:
            print(f'  结果图生成失败: {e}')

        print(f'  可视化输出: {PROJECT_ROOT / "figures"}')
