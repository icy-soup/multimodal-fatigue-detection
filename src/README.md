# src/ — 源代码目录说明

## 目录结构与执行顺序

```
src/
├── README.md                          ← 本文件
├── step1_preprocessing/               ← ① 信号预处理
│   ├── eeg.py                         EEG加载->带通滤波->陷波->滑动窗口
│   ├── ecg.py                         ECG加载->滤波->R波检测->滑动窗口
│   ├── eda.py                         EDA加载->低通滤波->SCL/SCR分解->滑动窗口
│   ├── semg.py                        sEMG滤波->滑动窗口 (被semg_loader调用)
│   ├── semg_loader.py                 sEMG从嵌套zip读取+标签对齐
│   ├── driver_eeg.py                  Driver EEG从CNT二进制读取+EEG提取
│   ├── fatigueset.py                  FatigueSet加载器: 12人x3session, EEG+ECG+EDA对齐
│   └── utils.py                       通用工具: 滤波器/窗口/时间对齐
│
├── step2_features/                    ← ② 特征提取
│   ├── __init__.py
│   ├── eeg.py                         74维: 频带功率(32)+熵特征(12)+功能连接(30)
│   ├── ecg.py                         12维: HRV时域(4)+HRV频域(3)+非线性(5)
│   ├── eda.py                         6维: SCL均值/标准差/斜率+SCR幅值/频率
│   ├── semg.py                        20维: 时域(10)+频域(10) sEMG特征
│   ├── driver_eeg.py                  340维: 34通道x10(5频带绝对+相对功率)
│   └── fusion.py                      三模态聚合EEG+ECG+EDA -> 92维
│
├── step3_models/                      ← ③ 模型训练与优化
│   ├── train.py                       RF/XGBoost + LOSO + 6种融合策略 + 互信息权重
│   ├── ga_pso.py                      GA特征选择 + PSO超参调优
│   └── ga_pso_validate.py            GA-PSO结果的LOSO验证
│
├── step4_visualization/               ← ④ 可视化
│   ├── raw_data.py                    原始信号图(5张)
│   └── results.py                     模型评估图(5张)
│
└── step5_pipeline/                    ← ⑤ 运行入口
    └── train_evaluate.py              统一流水线: --semg / --driver-eeg / --ga-pso
```

## 实验入口

```bash
cd src

# FatigueSet (ECG+EEG 两模态融合实验)
python -m step5_pipeline.train_evaluate --train-only                # 全量12人
python -m step5_pipeline.train_evaluate --quick --all               # 3人快速测试

# sEMG 单模态实验
python -m step5_pipeline.train_evaluate --semg --train-only         # 全量13人

# Driver EEG 独立验证实验
python -m step5_pipeline.train_evaluate --driver-eeg --train-only   # 全量12人

# GA-PSO 联合优化
python -m step5_pipeline.train_evaluate --ga-pso

# GA-PSO LOSO验证
python -m step3_models.ga_pso_validate

# 仅重新生成可视化
python -m step5_pipeline.train_evaluate --viz-only

# 跳过预处理 (使用已缓存的features.npz)
python -m step5_pipeline.train_evaluate --train-only --skip-preprocess
```

## 最终实验结果

| 实验 | 最佳模型 | F1 | AUC |
|------|---------|-----|-----|
| FatigueSet | weighted + XGBoost | 0.507 | 0.650 |
| sEMG | semg_raw + XGBoost | 0.782 | 0.715 |
| Driver EEG | driver_raw + RF | 0.646 | 0.598 |
| GA-PSO | -- | 0.361 | 0.389 (过拟合) |

## 输出文件

| 文件 | 说明 |
|------|------|
| data/results/results.json | FatigueSet 全量LOSO结果 |
| data/results/semg_results.json | sEMG LOSO结果 |
| data/results/driver_eeg_results.json | Driver EEG LOSO结果 |
| data/results/ga_pso_results.json | GA-PSO优化结果 |
| data/results/ga_pso_loso_results.json | GA-PSO LOSO验证结果 |
| data/processed/features.npz | FatigueSet 92维特征 |
| data/processed/semg_features.npz | sEMG 20维特征 |
| data/processed/driver_eeg_features.npz | Driver EEG 340维特征 |
| figures/results/ | 9张评估图 + analysis.txt |
| figures/raw_data/ | 5张原始信号图 |

## 关键参数

| 参数 | 值 | 位置 |
|------|-----|------|
| EEG采样率 | 256Hz | step1_preprocessing/eeg.py |
| ECG采样率 | 250Hz | step1_preprocessing/ecg.py |
| EDA采样率 | 4Hz | step1_preprocessing/eda.py |
| sEMG采样率 | 1259Hz | step1_preprocessing/semg_loader.py |
| Driver EEG采样率 | 1000Hz | step1_preprocessing/driver_eeg.py |
| 滑动窗口 | 4s窗 / 2s步长 | step1_preprocessing/utils.py |
| 疲劳阈值 | mentalFatigueScore > 30 | step1_preprocessing/fatigueset.py |
| LOSO验证 | Leave-One-Subject-Out | step3_models/train.py |
| RF | 200棵树, max_depth=15 | step3_models/train.py |
| XGBoost | 200轮, lr=0.05, max_depth=6 | step3_models/train.py |
