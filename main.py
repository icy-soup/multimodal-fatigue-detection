#!/usr/bin/env python
"""多模态疲劳检测系统 — 主入口

可验证、可展示的完整工程方案：
  输入: ECG + sEMG + EEG 多模态生理信号
  处理: 信号滤波 → 特征提取 → 多模态融合 → 分类识别
  输出: 疲劳/非疲劳等级 + 评估指标

文献驱动的方法选择:
  [L1] Kakhi et al. (2024) — 多模态融合 AI 疲劳检测综述
  [L2] Zhou et al. (2024) — gcForest 小样本疲劳识别
  [L3] Corvini & Conforto (2022) — sEMG MDF/AR Burg 分析
  [L4] Cao et al. (2024) — MTFN-SAM 注意力融合框架
  [L4] Wang et al. (2023) — EEG+ECG Multi-Sensor Fusion
  [L7] 课程材料 — SVM/GA/PSO 基础算法

用法:
  python main.py --quick          # 快速测试 (3人, 跳过预处理)
  python main.py --full           # 全量运行 (12人 LOSO)
  python main.py --fatigueset     # 仅 FatigueSet 实验
  python main.py --semg           # 仅 sEMG 对照实验
  python main.py --driver-eeg     # 仅 Driver EEG 对照实验
  python main.py --ga-pso         # GA-PSO 联合优化
  python main.py --viz            # 仅可视化
"""

import sys, os
from pathlib import Path

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_system_banner():
    """打印系统 banner"""
    print(r"""
╔══════════════════════════════════════════════════════════════╗
║        多模态疲劳检测系统 — MultimodalFatigueDetection        ║
║        ECG + sEMG + EEG → 信号处理 → 融合 → 分类 → 输出      ║
╚══════════════════════════════════════════════════════════════╝
    """)
    print('文献驱动方法:')
    print('  [L1] Kakhi et al. (2024) — Wearables+AI Survey')
    print('  [L2] Zhou et al. (2024) — gcForest Fatigue Recognition')
    print('  [L3] Corvini & Conforto (2022) — sEMG MDF/AR Burg')
    print('  [L4] Cao et al. (2024) — MTFN-SAM Attention Fusion')
    print('  [L4] Wang et al. (2023) — EEG+ECG Fusion')
    print('  [L7] 课程材料 — SVM/GA/PSO')
    print()


def main():
    import argparse
    p = argparse.ArgumentParser(description='多模态疲劳检测系统')
    p.add_argument('--quick', action='store_true', help='快速测试 (3人)')
    p.add_argument('--full', action='store_true', help='全量运行')
    p.add_argument('--fatigueset', action='store_true', help='仅 FatigueSet')
    p.add_argument('--semg', action='store_true', help='仅 sEMG 对照')
    p.add_argument('--driver-eeg', action='store_true', help='仅 Driver EEG')
    p.add_argument('--ga-pso', action='store_true', help='GA-PSO 优化')
    p.add_argument('--viz', action='store_true', help='仅可视化')
    p.add_argument('--skip-preprocess', action='store_true', default=True,
                   help='跳过预处理(默认启用)')
    p.add_argument('--no-skip', action='store_true', help='重新预处理')
    args = p.parse_args()

    print_system_banner()

    skip_preprocess = not args.no_skip

    # 创建系统实例
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    from src.system.multimodal_system import MultimodalFatigueDetectionSystem
    system = MultimodalFatigueDetectionSystem()

    # ─── 可视化 ────────────────────────────────────
    if args.viz:
        system.visualize()
        return

    # ─── GA-PSO ────────────────────────────────────
    if args.ga_pso:
        system.load_fatigueset(skip_preprocess=skip_preprocess)
        result = system.run_ga_pso()
        import json
        print(json.dumps(result, indent=2))
        return

    # ─── 单数据集 ──────────────────────────────────
    if args.fatigueset:
        system.load_fatigueset(skip_preprocess=skip_preprocess)
        clfs = ['RF', 'XGBoost', 'gcForest']
        fusions = ['eeg_only', 'ecg_only', 'eda_only',
                  'concat', 'weighted', 'decision']
        results = system.evaluate('fatigueset', classifiers=clfs, fusions=fusions)
        system._print_summary(results, 'FatigueSet')
        return

    if args.semg:
        system.load_semg(skip_preprocess=skip_preprocess)
        results = system.evaluate('semg', classifiers=['RF', 'XGBoost'],
                                  fusions=['semg_raw', 'semg_weighted'])
        system._print_summary(results, 'sEMG')
        return

    if args.driver_eeg:
        system.load_driver_eeg(skip_preprocess=skip_preprocess)
        results = system.evaluate('driver_eeg', classifiers=['RF', 'XGBoost'],
                                  fusions=['driver_raw', 'driver_weighted'])
        system._print_summary(results, 'Driver EEG')
        return

    # ─── 完整流水线（默认） ────────────────────────
    if args.full:
        system.run_full_pipeline(quick=False, skip_preprocess=skip_preprocess)
    else:
        # 默认快速测试
        system.run_full_pipeline(quick=True, skip_preprocess=skip_preprocess)


if __name__ == '__main__':
    main()
