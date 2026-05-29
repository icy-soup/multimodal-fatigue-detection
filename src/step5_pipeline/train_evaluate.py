#!/usr/bin/env python
"""模型训练与评估流水线入口

运行完整流程：
  python -m step5_pipeline.train_evaluate --all        # 全流程
  python -m step5_pipeline.train_evaluate --quick      # 快速测试 (subj 01-03)
  python -m step5_pipeline.train_evaluate --train-only  # 只训练
  python -m step5_pipeline.train_evaluate --viz-only    # 只可视化
"""

import sys, os, argparse, json
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--semg', action='store_true', help='sEMG 对照实验')
    p.add_argument('--driver-eeg', action='store_true', help='Driver EEG 对照实验')
    p.add_argument('--ga-pso', action='store_true', help='GA-PSO 联合优化')
    p.add_argument('--quick', action='store_true', help='快速测试: 只加载3个subject')
    p.add_argument('--all', action='store_true', help='全流程: 预处理→特征→训练→可视化')
    p.add_argument('--train-only', action='store_true')
    p.add_argument('--viz-only', action='store_true')
    p.add_argument('--viz-raw', action='store_true', help='只生成原始数据可视化')
    p.add_argument('--skip-preprocess', action='store_true', help='跳过预处理,直接用已保存的features.npz')
    p.add_argument('--data-root', default=None)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--subjects', type=int, nargs='+', default=None, help='指定 subject 列表')
    args = p.parse_args()

    base = Path(args.data_root) if args.data_root else Path(__file__).parent.parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else base / 'data' / 'results'
    processed_dir = base / 'data' / 'processed'
    figures_dir = Path(args.output_dir) if args.output_dir else base / 'figures'
    raw_viz_dir = figures_dir / 'raw_data'
    result_viz_dir = figures_dir / 'results'

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.semg:
        _run_semg(args, base, output_dir, processed_dir, result_viz_dir)
        return
    if args.ga_pso:
        _run_ga_pso(args, base, output_dir, processed_dir)
        return
    if args.driver_eeg:
        _run_driver_eeg(args, base, output_dir, processed_dir, result_viz_dir)
        return

    # ─── 原始数据可视化 ─────────────────────────────────
    if args.viz_raw or args.all:
        from step4_visualization.raw_data import generate_all
        generate_all(data_root, raw_viz_dir)

    if args.viz_only:
        # 只加载已保存的结果进行可视化
        results_path = output_dir / 'results.json'
        features_path = processed_dir / 'features.npz'

        if not results_path.exists():
            print(f"[ERROR] 未找到 {results_path}，请先运行 --train-only")
            return

        import json
        with open(results_path) as f:
            results_raw = json.load(f)

        # 重建 results dict（cm 还原为 np.array）
        results = _rebuild_results(results_raw)
        dataset = dict(np.load(features_path, allow_pickle=True))

        from step4_visualization.results import generate_all as gen_results
        gen_results(results, dataset, result_viz_dir)
        return

    # ─── 预处理 ────────────────────────────────────────
    features_path = processed_dir / 'features.npz'

    if not args.skip_preprocess or not features_path.exists():
        print("\n" + "=" * 60)
        print("Step 1/3: 信号预处理...")
        print("=" * 60)

        from step1_preprocessing import fatigueset

        if args.quick:
            subjects = [f"{i:02d}" for i in range(1, 4)]
        else:
            subjects = [f"{i:02d}" for i in range(1, 13)]

        all_data = fatigueset.load(data_root, subjects=subjects)
        if not all_data:
            print("[ERROR] 无可用数据")
            return

        ds_raw = fatigueset.aggregate(all_data)
        print(f"\n聚合: samples={len(ds_raw['y'])}, "
              f"EEG={ds_raw['X']['eeg'].shape}, ECG={ds_raw['X']['ecg'].shape}, EDA={ds_raw['X']['eda'].shape}")
        print(f"疲劳: {(ds_raw['y']==1).sum()}, 非疲劳: {(ds_raw['y']==0).sum()}")

        np.savez_compressed(processed_dir / 'fatigueset_preprocessed.npz',
                           **ds_raw['X'], labels=ds_raw['y'],
                           subject_ids=ds_raw['subject_ids'])

        # ─── 特征提取 ────────────────────────────────────
        print("\n" + "=" * 60)
        print("Step 2/3: 特征提取...")
        print("=" * 60)

        from step2_features import fusion
        ds = fusion.extract_dataset(all_data)
        print(f"\n特征提取完成: X={ds['X'].shape}, "
              f"y分布=疲劳{(ds['y']==1).sum()} 非疲劳{(ds['y']==0).sum()}, "
              f"NaN={np.isnan(ds['X']).sum()}")

        np.savez_compressed(features_path, X=ds['X'], y=ds['y'],
                           modality_dims=np.array(ds['modality_dims']),
                           subject_ids=ds['subject_ids'])
    else:
        print("[SKIP] 使用已保存的特征文件")

    if args.train_only or args.all:
        # ─── 模型训练 ────────────────────────────────────
        print("\n" + "=" * 60)
        print("Step 3/3: 模型训练与评估 (LOSO)")
        print("=" * 60)

        from step3_models.train import load_features, run_experiment, summarize_results
        import time

        dataset = load_features(features_path)

        n_subjects = len(set(dataset['subject_ids']))
        n_samples = len(dataset['y'])
        print(f"\n数据: {n_samples} 样本, {n_subjects} 个被试")
        print(f"特征维度: {dataset['X'].shape[1]}")
        print(f"标签分布: 疲劳 {(dataset['y']==1).sum()}, 非疲劳 {(dataset['y']==0).sum()}")
        print(f"\n预计耗时: 每fold约30-60s, 共 {n_subjects} folds, 总计约 {n_subjects*0.5:.0f}-{n_subjects*1:.0f} 分钟")
        print("训练中会显示实时进度条，请耐心等待...\n")
        sys.stdout.flush()

        t_start = time.time()
        classifiers = ['RF', 'XGBoost']
        fusions = ['eeg_only', 'ecg_only', 'eda_only', 'concat', 'weighted', 'decision']

        results = run_experiment(dataset, classifiers=classifiers, fusions=fusions, verbose=True)
        elapsed = time.time() - t_start
        print(f"\n训练总耗时: {elapsed/60:.1f} 分钟")
        summarize_results(results)

        # 保存结果（JSON，cm 序列化）
        results_serializable = {}
        for fusion in fusions:
            results_serializable[fusion] = {}
            for clf in classifiers:
                r = results[fusion][clf]
                results_serializable[fusion][clf] = {
                    'overall': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                               for k, v in r['overall'].items()},
                    'y_true_all': r['y_true_all'],
                    'y_pred_all': r['y_pred_all'],
                    'y_prob_all': r['y_prob_all'],
                    'per_subject': {s: {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                        for k, v in m.items()}
                                   for s, m in r['per_subject'].items()},
                }

        import json
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\n结果已保存: {output_dir / 'results.json'}")

        # ─── 结果可视化 ──────────────────────────────────
        from step4_visualization.results import generate_all as gen_results
        gen_results(results, dataset, result_viz_dir)

    print("\n" + "=" * 60)
    print("完成！输出文件:")
    print(f"  预处理数据: {processed_dir}")
    print(f"  原始数据图: {raw_viz_dir}")
    print(f"  结果分析图: {result_viz_dir}")
    print(f"  图片说明:   {result_viz_dir / 'analysis.txt'}")
    print(f"  实验结果:   {output_dir / 'results.json'}")
    print("=" * 60)


def _run_semg(args, base, output_dir, processed_dir, result_viz_dir):
    """sEMG 对照实验：加载→特征→LOSO训练"""
    data_root = base / 'data' / 'semg'
    features_path = processed_dir / 'semg_features.npz'

    if not args.skip_preprocess or not features_path.exists():
        print("\n" + "=" * 60)
        print("sEMG: 数据加载与特征提取")
        print("=" * 60)

        from step1_preprocessing import semg_loader
        from step2_features import semg as semg_features

        subjects = args.subjects if args.subjects else list(range(1, 14))
        if args.quick:
            subjects = subjects[:3]

        all_data = semg_loader.load(data_root, subjects=subjects)
        if not all_data:
            print("[ERROR] 无可用数据")
            return

        ds = semg_features.extract_dataset(all_data)
        print(f"\n特征提取完成: X={ds['X'].shape}, "
              f"y分布=疲劳{(ds['y']==1).sum()} 非疲劳{(ds['y']==0).sum()}, "
              f"NaN={np.isnan(ds['X']).sum()}")

        np.savez_compressed(features_path, X=ds['X'], y=ds['y'],
                           modality_dims=np.array(ds['modality_dims']),
                           subject_ids=ds['subject_ids'])
    else:
        print("[SKIP] 使用已保存的特征文件")

    if args.train_only or args.all:
        print("\n" + "=" * 60)
        print("sEMG: 模型训练与评估 (LOSO)")
        print("=" * 60)

        from step3_models.train import load_features, run_experiment, summarize_results
        import time

        dataset = load_features(features_path)
        n_subjects = len(set(dataset['subject_ids']))
        n_samples = len(dataset['y'])
        print(f"\n数据: {n_samples} 样本, {n_subjects} 个被试")
        print(f"特征维度: {dataset['X'].shape[1]}")
        print(f"标签分布: 疲劳 {(dataset['y']==1).sum()}, 非疲劳 {(dataset['y']==0).sum()}")
        sys.stdout.flush()

        t_start = time.time()
        # sEMG 单模态, 只用 raw (concat) 和 weighted 两种融合
        classifiers = ['RF', 'XGBoost']
        fusions = ['semg_raw', 'semg_weighted']

        # 对 sEMG 定制 fusion: raw=原特征, weighted=互信息加权
        results = run_experiment(dataset, classifiers=classifiers, fusions=fusions, verbose=True)
        elapsed = time.time() - t_start
        print(f"\n训练总耗时: {elapsed/60:.1f} 分钟")
        summarize_results(results)

        import json
        results_serializable = {}
        for fusion in fusions:
            results_serializable[fusion] = {}
            for clf in classifiers:
                r = results[fusion][clf]
                results_serializable[fusion][clf] = {
                    'overall': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                               for k, v in r['overall'].items()},
                    'y_true_all': r['y_true_all'],
                    'y_pred_all': r['y_pred_all'],
                    'y_prob_all': r['y_prob_all'],
                    'per_subject': {s: {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                        for k, v in m.items()}
                                   for s, m in r['per_subject'].items()},
                }

        with open(output_dir / 'semg_results.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\n结果已保存: {output_dir / 'semg_results.json'}")

    print(f"\n完成！sEMG 实验结果: {output_dir / 'semg_results.json'}")
    print(f"  预处理数据: {processed_dir / 'semg_features.npz'}")


def _run_driver_eeg(args, base, output_dir, processed_dir, result_viz_dir):
    """Driver EEG 对照实验"""
    data_root = base / 'data' / 'driver_eeg'
    features_path = processed_dir / 'driver_eeg_features.npz'

    if not args.skip_preprocess or not features_path.exists():
        print("\n" + "=" * 60)
        print("Driver EEG: 数据加载与特征提取")
        print("=" * 60)

        from step1_preprocessing import driver_eeg
        from step2_features import driver_eeg as deeg_features

        subjects = args.subjects if args.subjects else list(range(1, 13))
        if args.quick:
            subjects = subjects[:3]

        all_data = driver_eeg.load(data_root, subjects=subjects)
        if not all_data:
            print("[ERROR] 无可用数据")
            return

        ds = deeg_features.extract_dataset(all_data)
        print(f"\n特征提取完成: X={ds['X'].shape}, "
              f"y分布=疲劳{(ds['y']==1).sum()} 非疲劳{(ds['y']==0).sum()}, "
              f"NaN={np.isnan(ds['X']).sum()}")

        np.savez_compressed(features_path, X=ds['X'], y=ds['y'],
                           modality_dims=np.array(ds['modality_dims']),
                           subject_ids=ds['subject_ids'])
    else:
        print("[SKIP] 使用已保存的特征文件")

    if args.train_only or args.all:
        print("\n" + "=" * 60)
        print("Driver EEG: 模型训练与评估 (LOSO)")
        print("=" * 60)

        from step3_models.train import load_features, run_experiment, summarize_results
        import time

        dataset = load_features(features_path)
        n_subjects = len(set(dataset['subject_ids']))
        n_samples = len(dataset['y'])
        print(f"\n数据: {n_samples} 样本, {n_subjects} 个被试")
        print(f"特征维度: {dataset['X'].shape[1]}")
        print(f"标签分布: 疲劳 {(dataset['y']==1).sum()}, 非疲劳 {(dataset['y']==0).sum()}")
        sys.stdout.flush()

        t_start = time.time()
        classifiers = ['RF', 'XGBoost']
        fusions = ['driver_raw', 'driver_weighted']

        results = run_experiment(dataset, classifiers=classifiers, fusions=fusions, verbose=True)
        elapsed = time.time() - t_start
        print(f"\n训练总耗时: {elapsed/60:.1f} 分钟")
        summarize_results(results)

        import json
        results_serializable = {}
        for fusion in fusions:
            results_serializable[fusion] = {}
            for clf in classifiers:
                r = results[fusion][clf]
                results_serializable[fusion][clf] = {
                    'overall': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                               for k, v in r['overall'].items()},
                    'y_true_all': r['y_true_all'],
                    'y_pred_all': r['y_pred_all'],
                    'y_prob_all': r['y_prob_all'],
                    'per_subject': {s: {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                        for k, v in m.items()}
                                   for s, m in r['per_subject'].items()},
                }

        with open(output_dir / 'driver_eeg_results.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\n结果已保存: {output_dir / 'driver_eeg_results.json'}")

    print(f"\n完成！Driver EEG 实验结果: {output_dir / 'driver_eeg_results.json'}")
    print(f"  预处理数据: {processed_dir / 'driver_eeg_features.npz'}")


def _run_ga_pso(args, base, output_dir, processed_dir):
    """GA-PSO 联合优化"""
    features_path = processed_dir / 'features.npz'
    if not features_path.exists():
        print(f"[ERROR] 未找到特征文件 {features_path}，请先运行 FatigueSet 训练")
        return

    print("\n" + "=" * 60)
    print("GA-PSO 联合优化")
    print("=" * 60)

    import numpy as np
    data = np.load(features_path, allow_pickle=True)
    X, y = data['X'], data['y']

    print(f"\n数据: X={X.shape}, 疲劳={(y==1).sum()}, 非疲劳={(y==0).sum()}")
    print(f"GA: pop={30}, gen={40}, 约 40×30×3CV = 3600次模型训练")
    print("PSO: particles={}, iter={}, 约 20×15×3CV = 900次模型训练".format(15, 20))
    print("预估耗时: 20-40 分钟\n")

    from step3_models.ga_pso import run_ga_pso
    result = run_ga_pso(X, y, verbose=True)

    import json
    # 保存结果
    result_serializable = {
        'n_selected': result['n_selected'],
        'selected_features': result['selected_features'],
        'best_params': {k: (int(v) if isinstance(v, np.integer) else float(v))
                       for k, v in result['best_params'].items()},
        'ga_cv_f1': float(result['ga_cv_f1']),
        'pso_cv_f1': float(result['pso_cv_f1']),
    }
    with open(output_dir / 'ga_pso_results.json', 'w') as f:
        json.dump(result_serializable, f, indent=2)

    print(f"\n结果已保存: {output_dir / 'ga_pso_results.json'}")
    print(f"\nGA-PSO 完成！")
    print(f"  选中特征: {result['n_selected']}/{X.shape[1]}")
    print(f"  最佳参数: {result['best_params']}")
    print(f"  GA CV F1: {result['ga_cv_f1']:.4f}")
    print(f"  PSO CV F1: {result['pso_cv_f1']:.4f}")
    print(f"\n下一步: 用选中特征和最佳参数运行 LOSO 验证以获得最终结果")


def _rebuild_results(raw):
    """从 JSON 重建 results dict"""
    import numpy as np
    results = {}
    for fusion, clf_dict in raw.items():
        results[fusion] = {}
        for clf, r in clf_dict.items():
            results[fusion][clf] = {
                'overall': {k: (np.array(v) if k == 'cm' else v) for k, v in r['overall'].items()},
                'y_true_all': r['y_true_all'],
                'y_pred_all': r['y_pred_all'],
                'y_prob_all': r['y_prob_all'],
                'per_subject': {s: {k: (np.array(v) if k == 'cm' else v) for k, v in m.items()}
                               for s, m in r['per_subject'].items()},
            }
    return results


if __name__ == '__main__':
    main()
