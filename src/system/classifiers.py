"""分类器集合 — 文献驱动的方法选择

包含文献推荐的所有分类器:
  - SVM (RBF): [L4][L7] 课程基线 + MTFN-SAM 使用
  - Random Forest: [L1] 多模态疲劳检测常用基线
  - XGBoost: [L1] 表格特征强分类器
  - gcForest: [L2] 小样本场景优于深度学习，级联深度森林

Reference:
  [L2] Zhou et al. (2024) Bayes-gcForest. Sensors, 24(9), 2910.
  [L4] Cao et al. (2024) MTFN-SAM. Biomedical Signal Processing, 89, 105756.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def get_classifier(name, **kwargs):
    """根据名称返回分类器实例"""
    if name == 'SVM':
        return SVC(kernel=kwargs.get('kernel', 'rbf'),
                   C=kwargs.get('C', 1.0),
                   gamma=kwargs.get('gamma', 'scale'),
                   probability=True,
                   random_state=42)
    elif name == 'RF':
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 15),
            random_state=42, n_jobs=-1)
    elif name == 'XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.05),
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42, verbosity=0)
    elif name == 'gcForest':
        return CascadeForest(
            n_forests=kwargs.get('n_cascade_forests', 4),
            n_trees=kwargs.get('n_trees', 100),
            tolerance=kwargs.get('tolerance', 0.0))
    elif name == 'multi_cascade':
        # 多层级联融合 (改进二) — 需要 modality_dims 参数
        return MultiLayerCascadeForest(
            modality_dims=kwargs.get('modality_dims', None),
            n_forests=kwargs.get('n_cascade_forests', 4),
            n_trees=kwargs.get('n_trees', 100),
            tolerance=kwargs.get('tolerance', 0.0))
    else:
        raise ValueError(f'未知分类器: {name}')


class CascadeForest:
    """级联森林 (gcForest 级联部分) — [L2] Zhou et al. 2024

    核心思想：
      1. 每一层由多个随机森林组成（完全随机森林 + 普通随机森林）
      2. 每层输出类概率向量 + 原始特征 → 输入下一层
      3. 用 k-fold 内层交叉验证生成类向量（防止过拟合）
      4. 当验证集性能不再提升时自动停止级联

    Reference:
      Zhou & Feng (2017). Deep Forest. arXiv:1702.08835.
      被 [L2] Zhou et al. (2024) 用于脑疲劳识别，在小样本(12人)上优于深度学习
    """

    def __init__(self, n_forests=4, n_trees=100, tolerance=0.0, random_state=42):
        self.n_forests = n_forests          # 每层森林数（一半完全随机，一半普通RF）
        self.n_trees = n_trees               # 每个森林的树数量
        self.tolerance = tolerance           # 早停容忍度
        self.random_state = random_state
        self.layers_ = []                    # 级联层
        self.n_classes_ = None
        self.scaler_ = StandardScaler()

    def _create_forests(self):
        """创建一层中的所有森林"""
        forests = []
        for i in range(self.n_forests):
            if i < self.n_forests // 2:
                # 完全随机森林：max_features=1
                rf = RandomForestClassifier(
                    n_estimators=self.n_trees,
                    max_features=1,
                    max_depth=None,
                    min_samples_leaf=1,
                    random_state=self.random_state + i,
                    n_jobs=-1)
            else:
                # 普通随机森林：max_features=sqrt(n_features)
                rf = RandomForestClassifier(
                    n_estimators=self.n_trees,
                    max_features='sqrt',
                    max_depth=None,
                    min_samples_leaf=1,
                    random_state=self.random_state + i,
                    n_jobs=-1)
            forests.append(rf)
        return forests

    def _predict_forests(self, forests, X):
        """所有森林预测，返回拼接的类概率向量 [n_samples, n_classes * n_forests]"""
        probas = []
        for rf in forests:
            p = rf.predict_proba(X)
            probas.append(p)
        return np.concatenate(probas, axis=1)

    def fit(self, X, y):
        """训练级联森林

        每层用 k-fold 生成类概率向量，拼接后作为下一层输入。
        当验证集 accuracy 不再提升时停止。
        """
        self.n_classes_ = len(np.unique(y))
        n_splits = min(3, len(np.unique(y)))

        # 标准化
        X = self.scaler_.fit_transform(X)

        # 初始化：原始特征 → 第一层
        X_current = X.copy()
        best_acc = 0.0
        layer_idx = 0

        with tqdm(desc='gcForest 级联层', unit='层', ncols=80, disable=None) as pbar:
            while True:
                forests = self._create_forests()
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

                # k-fold 生成类向量
                fold_probas = []
                for train_idx, val_idx in skf.split(X_current, y):
                    X_tr, y_tr = X_current[train_idx], y[train_idx]

                    for rf in forests:
                        rf.fit(X_tr, y_tr)

                    val_proba = self._predict_forests(forests, X_current[val_idx])
                    fold_probas.append((val_idx, val_proba))

                # 拼接所有fold的类向量
                class_probas = np.zeros((X.shape[0], self.n_classes_ * self.n_forests))
                for idx, proba in fold_probas:
                    class_probas[idx] = proba

                # 用完整数据重新训练这层的森林
                for rf in forests:
                    rf.fit(X_current, y)

                # 保存这层（必须在 predict 之前，predict 依赖 layers_）
                self.layers_.append(forests)

                # 评估当前层
                y_pred = self.predict(X, stop_at=layer_idx + 1)
                acc = accuracy_score(y, y_pred)
                pbar.set_postfix({'acc': f'{acc:.4f}'})
                layer_idx += 1
                pbar.update(1)

                # 检查收敛
                if acc > best_acc + self.tolerance:
                    best_acc = acc
                elif layer_idx >= 2:  # 至少两层
                    pbar.set_description(f'gcForest 收敛于 {layer_idx} 层 (acc={best_acc:.4f})')
                    break

                # 准备下一层输入：原始特征 + 类概率向量
                X_current = np.hstack([X, class_probas])

                # 安全限制：最多20层
                if layer_idx >= 20:
                    break

        return self

    def predict_proba(self, X):
        """预测类概率

        每层依次预测，最后一层的类概率向量取平均。
        """
        X = self.scaler_.transform(X)
        X_current = X.copy()

        for forests in self.layers_:
            probas = self._predict_forests(forests, X_current)
            # 下一层输入：原始特征 + 类概率
            X_current = np.hstack([X, probas])

        # 最后一层的类概率：所有森林平均
        n = self.n_forests
        final_probas = np.zeros((X.shape[0], self.n_classes_))
        for i in range(0, probas.shape[1], self.n_classes_):
            final_probas += probas[:, i:i + self.n_classes_]
        final_probas /= n
        return final_probas

    def predict(self, X, stop_at=None):
        """预测类别"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self


class MultiLayerCascadeForest:
    """多层级联融合 (改进二) — 模态特定分支 + 逐层融合

    与标准 gcForest [L2] 的区别:
      [L2]:          ECG特征+EEG特征 → concat(92维) → 级联森林
      本文改进:      ECG分支→[RF×4] + EEG分支→[RF×4] → 融合 → 下一层

    每层内部:
      1. 按模态拆分特征
      2. 每模态独立用随机森林预测，生成类概率向量
      3. 所有模态的类概率向量拼接 + 原始特征 → 下一层
      4. 性能收敛时停止级联

    Reference:
      [L2] Zhou et al. (2024) Bayes-gcForest. Sensors, 24(9), 2910.
      改进: 模态分支 + 逐层融合替代单次拼接
    """

    def __init__(self, modality_dims=None, n_forests=4, n_trees=100,
                 tolerance=0.0, random_state=42):
        self.modality_dims = modality_dims  # [eeg, ecg, eda] 维度
        self.n_forests = n_forests
        self.n_trees = n_trees
        self.tolerance = tolerance
        self.random_state = random_state
        self.layers_ = []
        self.n_classes_ = None
        self.scaler_ = StandardScaler()
        self._mod_ranges = []  # 模态范围缓存

    def _get_mod_ranges(self, n_features):
        """按模态拆分特征范围"""
        if self.modality_dims is not None:
            dims = [int(d) for d in self.modality_dims if int(d) > 0]
            ranges = []
            start = 0
            for d in dims:
                if d > 0:
                    ranges.append((start, start + d))
                    start += d
            return ranges
        # 无模态信息时，等分成 n_forests 组
        chunk = n_features // self.n_forests
        return [(i * chunk, (i + 1) * chunk if i < self.n_forests - 1 else n_features)
                for i in range(self.n_forests)]

    def _create_mod_forests(self, n_mod_features):
        """创建一组模态特定森林（一半完全随机，一半普通RF）"""
        forests = []
        for i in range(self.n_forests):
            if i < self.n_forests // 2:
                rf = RandomForestClassifier(
                    n_estimators=self.n_trees,
                    max_features=min(1, n_mod_features),
                    max_depth=None, min_samples_leaf=1,
                    random_state=self.random_state + i, n_jobs=-1)
            else:
                rf = RandomForestClassifier(
                    n_estimators=self.n_trees,
                    max_features='sqrt',
                    max_depth=None, min_samples_leaf=1,
                    random_state=self.random_state + i, n_jobs=-1)
            forests.append(rf)
        return forests

    def _predict_forests(self, forests, X):
        """所有森林预测，返回类概率向量 [n, n_classes * n_forests]"""
        probas = np.column_stack([rf.predict_proba(X) for rf in forests])
        return probas

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        X = self.scaler_.fit_transform(X)
        self._mod_ranges = self._get_mod_ranges(X.shape[1])

        X_current = X.copy()
        best_acc = 0.0
        layer_idx = 0

        with tqdm(desc='multi_cascade 级联层', unit='层', ncols=80, disable=None) as pbar:
            while True:
                layer_forests = {}  # mod_idx → [forests]
                fold_probas = []

                for train_idx, val_idx in StratifiedKFold(
                    n_splits=min(3, len(np.unique(y))),
                    shuffle=True, random_state=42
                ).split(X_current, y):
                    X_tr, y_tr = X_current[train_idx], y[train_idx]

                    # 各模态分支
                    mod_forests = {}
                    for m_idx, (s, e) in enumerate(self._mod_ranges):
                        forests = self._create_mod_forests(e - s)
                        for rf in forests:
                            rf.fit(X_tr[:, s:e], y_tr)
                        mod_forests[m_idx] = forests

                    # 融合各模态类概率
                    val_probas = []
                    for m_idx, (s, e) in enumerate(self._mod_ranges):
                        p = self._predict_forests(mod_forests[m_idx],
                                                  X_current[val_idx][:, s:e])
                        val_probas.append(p)
                    fold_probas.append((val_idx, np.concatenate(val_probas, axis=1),
                                       mod_forests))

                # 构建完整训练集的类概率
                class_probas = np.zeros((X.shape[0], self.n_classes_ * self.n_forests
                                        * len(self._mod_ranges)))
                # 存储最后一折的 forests 作为该层模型
                final_forests = {}
                for idx, proba, mf in fold_probas:
                    class_probas[idx] = proba
                    final_forests = mf

                # 用完整数据重训练
                for m_idx, (s, e) in enumerate(self._mod_ranges):
                    for rf in final_forests[m_idx]:
                        rf.fit(X_current[:, s:e], y)

                # 保存层（必须在 predict 之前）
                self.layers_.append(final_forests)

                # 评估
                y_pred = self.predict(X)
                acc = accuracy_score(y, y_pred)
                pbar.set_postfix({'acc': f'{acc:.4f}'})

                layer_idx += 1
                pbar.update(1)

                if acc > best_acc + self.tolerance:
                    best_acc = acc
                elif layer_idx >= 2:
                    pbar.set_description(
                        f'multi_cascade 收敛于 {layer_idx} 层 (acc={best_acc:.4f})')
                    break

                # 下一层输入: 原始特征 + 所有模态类概率
                X_current = np.hstack([X, class_probas])
                if layer_idx >= 20:
                    break

        return self

    def predict_proba(self, X):
        X = self.scaler_.transform(X)
        X_current = X.copy()
        last_probas = None

        for forests_dict in self.layers_:
            layer_probas = []
            for m_idx, (s, e) in enumerate(self._mod_ranges):
                p = self._predict_forests(forests_dict[m_idx],
                                          X_current[:, s:e])
                layer_probas.append(p)
            last_probas = np.concatenate(layer_probas, axis=1)
            X_current = np.hstack([X] + layer_probas)

        if last_probas is None:
            # 无级联层（不应发生）
            return np.ones((X.shape[0], self.n_classes_)) / self.n_classes_

        # 最后一层所有模态所有森林平均
        n_forests_total = self.n_forests * len(self._mod_ranges)
        final_probas = np.zeros((X.shape[0], self.n_classes_))
        for i in range(0, last_probas.shape[1], self.n_classes_):
            final_probas += last_probas[:, i:i + self.n_classes_]
        final_probas /= n_forests_total
        return final_probas

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self
