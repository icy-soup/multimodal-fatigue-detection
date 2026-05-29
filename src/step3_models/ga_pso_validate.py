"""GA-PSO 优化后 LOSO 验证

加载 GA-PSO 找到的最佳特征子集和超参，跑 LOSO 验证真实效果。
"""
import sys, os, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from step3_models.train import load_features, run_experiment, summarize_results

# 加载 GA-PSO 结果
results_path = Path(__file__).parent.parent.parent / 'data' / 'results' / 'ga_pso_results.json'
features_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'features.npz'

with open(results_path) as f:
    ga_pso = json.load(f)

selected_mask = np.array(ga_pso['selected_features'], dtype=int)
params = ga_pso['best_params']

print("=" * 60)
print("GA-PSO LOSO 验证")
print("=" * 60)
print(f"选中特征: {len(selected_mask)}/92")
print(f"XGBoost 参数: {params}")

# 加载数据
dataset = load_features(str(features_path))
X_full, y = dataset['X'], dataset['y']

# 应用特征选择
X_selected = X_full[:, selected_mask]

print(f"\n数据: {X_selected.shape}, 疲劳={(y==1).sum()}, 非疲劳={(y==0).sum()}")
print(f"特征维度: {X_selected.shape[1]}")

# 构建优化后的 dataset
dataset_opt = {
    'X': X_selected,
    'y': y,
    'modality_dims': np.array([X_selected.shape[1]]),
    'subject_ids': dataset['subject_ids'],
}

# 跑 LOSO
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix

subjects = sorted(set(dataset_opt['subject_ids']))
classifiers = ['XGBoost_opt']
fusions = ['ga_pso_optimized']

results = {f: {c: {'per_subject': {}, 'y_true_all': [], 'y_pred_all': [], 'y_prob_all': []}
               for c in classifiers} for f in fusions}

t_start = time.time()
for test_subj in subjects:
    t0 = time.time()
    test_idx = [i for i, s in enumerate(dataset_opt['subject_ids']) if s == test_subj]
    train_idx = [i for i, s in enumerate(dataset_opt['subject_ids']) if s != test_subj]

    X_tr, X_te = X_selected[train_idx], X_selected[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = XGBClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        eval_metric='logloss', random_state=42, verbosity=0)
    clf.fit(X_tr_s, y_tr)

    y_pred = clf.predict(X_te_s)
    y_prob = clf.predict_proba(X_te_s)[:, 1]

    f1 = f1_score(y_te, y_pred, zero_division=0)
    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)
    cm = confusion_matrix(y_te, y_pred)

    results['ga_pso_optimized']['XGBoost_opt']['y_true_all'].extend(y_te.tolist())
    results['ga_pso_optimized']['XGBoost_opt']['y_pred_all'].extend(y_pred.tolist())
    results['ga_pso_optimized']['XGBoost_opt']['y_prob_all'].extend(y_prob.tolist())
    results['ga_pso_optimized']['XGBoost_opt']['per_subject'][test_subj] = {
        'accuracy': acc, 'f1': f1, 'auc': auc, 'cm': cm}

    elapsed = time.time() - t0
    print(f"  Subj {test_subj}: Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f} ({elapsed:.1f}s)")

# 汇总
yt = np.array(results['ga_pso_optimized']['XGBoost_opt']['y_true_all'])
yp = np.array(results['ga_pso_optimized']['XGBoost_opt']['y_pred_all'])
yprob = np.array(results['ga_pso_optimized']['XGBoost_opt']['y_prob_all'])

overall = {
    'accuracy': accuracy_score(yt, yp),
    'f1': f1_score(yt, yp, zero_division=0),
    'auc': roc_auc_score(yt, yprob),
    'cm': confusion_matrix(yt, yp).tolist(),
}
results['ga_pso_optimized']['XGBoost_opt']['overall'] = overall

total_time = time.time() - t_start
print(f"\n总耗时: {total_time/60:.1f} 分钟")
print(f"\n{'='*60}")
print(f"GA-PSO LOSO 验证结果")
print(f"{'='*60}")
print(f"Accuracy: {overall['accuracy']:.4f}")
print(f"F1 Score: {overall['f1']:.4f}")
print(f"AUC:      {overall['auc']:.4f}")
print(f"\n对比 baseline (weighted+XGBoost):")
print(f"  原始: F1=0.507, AUC=0.650")
print(f"  优化: F1={overall['f1']:.4f}, AUC={overall['auc']:.4f}")

# 保存结果
output = {
    'overall': {k: (v if isinstance(v, (int, float, str)) else v) for k, v in overall.items()},
    'per_subject': {str(s): results['ga_pso_optimized']['XGBoost_opt']['per_subject'][s]
                    for s in subjects},
    'params': params,
    'n_features_selected': len(selected_mask),
}
out_path = Path(__file__).parent.parent.parent / 'data' / 'results' / 'ga_pso_loso_results.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n结果已保存: {out_path}")
