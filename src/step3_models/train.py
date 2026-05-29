"""模型训练与评估

LOSO 交叉验证 + SVM/RF/XGBoost + 三种融合策略
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm
import time
import sys
import warnings
warnings.filterwarnings('ignore')

MODALITY_NAMES = ['EEG', 'ECG', 'EDA']
EEG_DIM, ECG_DIM, EDA_DIM = 74, 12, 6


def _fast_feature_weights(X, y, method='mutual_info', n_samples=500, random_state=42):
    """快速特征权重计算

    method='mutual_info': 互信息 (默认, 快速, O(n*d))
    method='relief': ReliefF采样版 (n_samples个采样点, 比全量快10倍以上)
    """
    if method == 'mutual_info':
        weights = mutual_info_classif(X, y, random_state=random_state)
        return weights / (weights.max() or 1)
    else:  # relief with sampling
        from sklearn.neighbors import NearestNeighbors
        n, d = X.shape
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, min(n_samples, n), replace=False)
        weights = np.zeros(d)
        y = y.astype(int)

        for i in idx:
            diff = np.abs(X - X[i])
            same_mask = (y == y[i])
            same_mask[i] = False
            diff_mask = (y != y[i])

            knn_same = NearestNeighbors(n_neighbors=min(5, same_mask.sum())).fit(X[same_mask])
            knn_diff = NearestNeighbors(n_neighbors=min(5, diff_mask.sum())).fit(X[diff_mask])

            _, idx_same = knn_same.kneighbors(X[i].reshape(1, -1))
            _, idx_diff = knn_diff.kneighbors(X[i].reshape(1, -1))

            near_hit = X[same_mask][idx_same[0]].mean(axis=0)
            near_miss = X[diff_mask][idx_diff[0]].mean(axis=0)
            weights += np.abs(X[i] - near_miss) - np.abs(X[i] - near_hit)

        weights = np.clip(weights / max(n, 1), 0, None)
        return weights / (weights.max() or 1)


def load_features(npz_path):
    """加载预处理好的特征文件"""
    data = np.load(npz_path, allow_pickle=True)
    return {
        'X': data['X'],
        'y': data['y'].astype(int),
        'modality_dims': data['modality_dims'],
        'subject_ids': list(data['subject_ids']),
    }


def split_modalities(X, modality_dims):
    """将92维特征拆分为 EEG(74) + ECG(12) + EDA(6)"""
    eeg_dim, ecg_dim, eda_dim = int(modality_dims[0]), int(modality_dims[1]), int(modality_dims[2])
    return X[:, :eeg_dim], X[:, eeg_dim:eeg_dim + ecg_dim], X[:, eeg_dim + ecg_dim:]


def weighted_features(X, weights, top_k=None):
    """用 ReliefF 权重对特征加权"""
    if top_k is not None:
        top_idx = np.argsort(weights)[-top_k:]
        mask = np.zeros_like(weights)
        mask[top_idx] = weights[top_idx]
        return X * mask
    return X * weights


def prepare_fusion_data(dataset, fusion_type='concat', relief_k=None):
    """准备融合数据

    Parameters
    ----------
    dataset : dict with X, y, modality_dims
    fusion_type : 'concat' | 'weighted' | 'eeg_only' | 'ecg_only' | 'eda_only'
    relief_k : int or None, ReliefF top-k features for weighted fusion

    Returns
    -------
    X_train, y_train (all data, will be split by LOSO)
    """
    X, y = dataset['X'], dataset['y']
    dims = dataset['modality_dims']

    if fusion_type == 'eeg_only':
        return X[:, :int(dims[0])], y
    elif fusion_type == 'ecg_only':
        return X[:, int(dims[0]):int(dims[0]) + int(dims[1])], y
    elif fusion_type == 'eda_only':
        return X[:, int(dims[0]) + int(dims[1]):], y
    elif fusion_type == 'weighted':
        weights = _fast_feature_weights(X, y, method='mutual_info')
        return weighted_features(X, weights, relief_k), y
    elif fusion_type == 'semg_weighted':
        weights = _fast_feature_weights(X, y, method='mutual_info')
        return weighted_features(X, weights, relief_k), y
    elif fusion_type == 'semg_raw':
        return X, y
    elif fusion_type == 'driver_weighted':
        weights = _fast_feature_weights(X, y, method='mutual_info')
        return weighted_features(X, weights, relief_k), y
    elif fusion_type == 'driver_raw':
        return X, y
    else:  # concat
        return X, y


def loso_split(dataset):
    """按 Subject 划分 LOSO fold"""
    subjects = sorted(set(dataset['subject_ids']))
    for test_subj in subjects:
        test_idx = [i for i, s in enumerate(dataset['subject_ids']) if s == test_subj]
        train_idx = [i for i, s in enumerate(dataset['subject_ids']) if s != test_subj]
        yield test_subj, train_idx, test_idx


def train_classifier(name, X_train, y_train, X_test):
    """训练分类器并返回预测"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if name == 'SVM':
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    elif name == 'RF':
        clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    elif name == 'XGBoost':
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                           random_state=42, verbosity=0)
    else:
        raise ValueError(f'Unknown classifier: {name}')

    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1] if hasattr(clf, 'predict_proba') else None
    return y_pred, y_prob


def decision_fusion_predict(X_test_mods, clfs):
    """决策级融合：三个模态分类器投票"""
    preds = []
    for mod_name in ['eeg', 'ecg', 'eda']:
        scaler, clf = clfs[mod_name]
        X_s = scaler.transform(X_test_mods[mod_name])
        preds.append(clf.predict(X_s))
    preds = np.array(preds)
    # 多数投票
    return (np.sum(preds, axis=0) >= 2).astype(int)


def compute_metrics(y_true, y_pred, y_prob=None):
    """计算全部评估指标"""
    m = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'cm': confusion_matrix(y_true, y_pred),
    }
    if y_prob is not None:
        m['auc'] = roc_auc_score(y_true, y_prob)
    return m


def run_experiment(dataset, classifiers=None, fusions=None, verbose=True):
    """运行完整实验：LOSO × 分类器 × 融合策略

    Parameters
    ----------
    dataset : dict
    classifiers : list of str, default ['SVM', 'RF', 'XGBoost']
    fusions : list of str, default ['eeg_only', 'ecg_only', 'eda_only', 'concat', 'weighted', 'decision']

    Returns
    -------
    results : dict[fusion][clf] = {metrics, per_subject}
    """
    if classifiers is None:
        classifiers = ['SVM', 'RF', 'XGBoost']
    if fusions is None:
        fusions = ['eeg_only', 'ecg_only', 'eda_only', 'concat', 'weighted', 'decision']

    X, y = dataset['X'], dataset['y']
    dims = dataset['modality_dims']
    subjects = sorted(set(dataset['subject_ids']))

    results = {f: {c: {'per_subject': {}, 'y_true_all': [], 'y_pred_all': [], 'y_prob_all': []}
                   for c in classifiers} for f in fusions}

    # 进度条: LOSO folds
    fold_pbar = tqdm(subjects, desc='LOSO Folds', unit='fold', ncols=100)
    for test_subj in fold_pbar:
        t0 = time.time()
        test_idx = [i for i, s in enumerate(dataset['subject_ids']) if s == test_subj]
        train_idx = [i for i, s in enumerate(dataset['subject_ids']) if s != test_subj]

        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_test_raw, y_test = X[test_idx], y[test_idx]
        fold_pbar.set_postfix({'subj': test_subj, 'train': len(train_idx), 'test': len(test_idx)})

        # 训练单模态分类器（用于决策融合，用 RF 替代 SVC 避免大样本耗时）
        mod_clfs = {}
        if 'decision' in fusions and len(dims) >= 3:
            for mod_name, mod_range in [('eeg', (0, int(dims[0]))),
                                         ('ecg', (int(dims[0]), int(dims[0]) + int(dims[1]))),
                                         ('eda', (int(dims[0]) + int(dims[1]), X.shape[1]))]:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train_raw[:, mod_range[0]:mod_range[1]])
                clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
                clf.fit(X_tr, y_train)
                mod_clfs[mod_name] = (scaler, clf)

        # 融合策略 + 分类器
        total_combos = len(fusions) * len(classifiers)
        combo_pbar = tqdm(total=total_combos, desc=f'  Subj {test_subj}', unit='combo',
                         leave=False, ncols=100)

        for fusion in fusions:
            X_fusion_train, _ = prepare_fusion_data(
                {'X': X_train_raw, 'y': y_train, 'modality_dims': dims}, fusion)
            X_fusion_test, _ = prepare_fusion_data(
                {'X': X_test_raw, 'y': y_test, 'modality_dims': dims}, fusion)

            for clf_name in classifiers:
                if fusion == 'decision':
                    y_pred = decision_fusion_predict(
                        {'eeg': X_test_raw[:, :int(dims[0])],
                         'ecg': X_test_raw[:, int(dims[0]):int(dims[0]) + int(dims[1])],
                         'eda': X_test_raw[:, int(dims[0]) + int(dims[1]):]}, mod_clfs)
                    y_prob = None
                else:
                    y_pred, y_prob = train_classifier(clf_name, X_fusion_train, y_train, X_fusion_test)

                results[fusion][clf_name]['y_true_all'].extend(y_test.tolist())
                results[fusion][clf_name]['y_pred_all'].extend(y_pred.tolist())
                if y_prob is not None:
                    results[fusion][clf_name]['y_prob_all'].extend(y_prob.tolist())

                metrics = compute_metrics(y_test, y_pred, y_prob)
                results[fusion][clf_name]['per_subject'][test_subj] = metrics

                combo_pbar.set_postfix(
                    {clf_name[:4]: f"{metrics['accuracy']:.2f}", 'fusion': fusion[:7]})
                combo_pbar.update(1)
                if verbose:
                    tqdm.write(f"  [{fusion:12s}] {clf_name:7s} | Acc={metrics['accuracy']:.3f} "
                              f"F1={metrics['f1']:.3f} AUC={metrics.get('auc', 0):.3f}")

        combo_pbar.close()
        elapsed = time.time() - t0
        if verbose:
            tqdm.write(f"  Fold {test_subj} 完成, 耗时 {elapsed:.1f}s\n")

    # 汇总整体指标
    for fusion in fusions:
        for clf_name in classifiers:
            r = results[fusion][clf_name]
            yt = np.array(r['y_true_all'])
            yp = np.array(r['y_pred_all'])
            yprob = np.array(r['y_prob_all']) if r['y_prob_all'] else None
            r['overall'] = compute_metrics(yt, yp, yprob)

    return results


def summarize_results(results):
    """打印结果汇总表"""
    fusions = list(results.keys())
    classifiers = list(results[list(fusions)[0]].keys())

    print("\n" + "=" * 90)
    print("结果汇总 (LOSO × 融合策略 × 分类器)")
    print("=" * 90)

    header = f"{'Fusion':<14s}"
    for clf in classifiers:
        header += f" | {clf:^22s}"
    print(header)
    print("-" * 90)

    for fusion in fusions:
        line = f"{fusion:<14s}"
        for clf in classifiers:
            m = results[fusion][clf]['overall']
            line += f" | Acc={m['accuracy']:.3f} F1={m['f1']:.3f} AUC={m.get('auc', 0):.3f}"
        print(line)

    # 找出最佳组合
    best = max(((f, c, results[f][c]['overall']['f1'])
                for f in fusions for c in classifiers), key=lambda x: x[2])
    print(f"\n最佳组合: {best[0]} + {best[1]} (F1={best[2]:.4f})")

    return results
