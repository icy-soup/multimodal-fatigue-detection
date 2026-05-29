# 多模态疲劳检测系统 Multimodal Fatigue Detection

基于心电(ECG)、肌电(sEMG)、脑电(EEG)三类生理信号的多模态疲劳检测系统。以可穿戴式疲劳监测为应用场景，围绕三类生理信号构建从信号输入到疲劳等级输出的完整工程方案。

**核心设计原则**：每个方法选择均有文献依据（详见 `docs/技术路线.md`）。

## 数据集

| 数据集 | 信号 | 被试 | 采样率 | 标签 | 角色 |
|--------|------|------|--------|------|------|
| **FatigueSet** | EEG(4ch) + ECG + EDA | 12人×3session | 256/250Hz | mentalFatigueScore>30 | 主实验：两模态融合 |
| **sEMG Dataset** (Zenodo) | 4通道sEMG | 13人×12trial | 1259Hz | 自评RPE | 对照：sEMG单模态 |
| **Driver EEG** (Figshare) | 9+通道EEG | 12人 | 250/1000Hz | Fatigue/Normal | 对照：EEG独立验证 |

## 信号处理

| 模态 | 流程 | 文献依据 |
|------|------|---------|
| **ECG** | 0.5-45Hz带通 → 50Hz陷波 → Pan-Tompkins R波检测 → 5秒滑窗(步长2.5s) | FatigueSet文档 |
| **sEMG** | 20-450Hz带通(4阶Butterworth) → AR Burg(3阶)PSD → 4秒滑窗(50%重叠) | [L3] |
| **EEG** | 0.5-45Hz带通 → 50Hz陷波 → ICA去伪迹 → 5秒滑窗(步长2.5s) | FatigueSet文档 |
| **EDA** | 1Hz低通 → SCL/SCR凸优化分解 → 对齐后滑窗 | FatigueSet文档 |

## 特征提取

| 模态 | 类别 | 特征 | 维度 |
|------|------|------|------|
| **ECG** [L2] | HRV时域 | SDNN, RMSSD, pNN50, HR_mean | 4 |
| | HRV频域 | LF, HF, LF/HF | 3 |
| | 非线性 | SD1, SD2, SD1/SD2, DFA_α₁, DFA_α₂ | 5 |
| **sEMG** [L3] | 时域 | MAV, RMS, ZC, SSC, WL, VAR, IEMG, SSI, SKEW, KURT | 10 |
| | 频域 | **MDF**（核心）, MNF, PKF, FR, TTP, 5频段能量 | 10 |
| **EEG** [L2] | 频带功率 | δ/θ/α/β/γ绝对功率 + θ/β, α/β, (θ+α)/β比值 | 28 |
| | 熵特征 | 多尺度熵、排列熵、样本熵 | 12 |
| | 功能连接 | 通道间PLV(相位锁定值) | 30 |
| **EDA** | SCL/SCR | 皮肤电导水平/反应 | 6 |
| **FatigueSet融合** | | **92维 (12+74+6)** | |

## 融合策略

| 策略 | 方法 | 角色 |
|------|------|------|
| **拼接融合** (concat) | ECG+EEG+EDA 直接拼接为92维向量 | 基线方法 |
| **加权融合** (weighted) | 互信息权重对特征加权 | 核心方法 |
| **决策融合** (decision) | 各模态独立训练 → 多数投票 | 对比方法 |
| **自适应权重融合** (adaptive_weighted) | 全局互信息 + 被试基线期校准 → 个体自适应权重 | **改进一** |
| **多层级联融合** (multi_cascade) | 每层级联内模态分支 + 逐层融合 | **改进二** |

## 分类器

| 分类器 | 文献依据 | 特点 | 角色 |
|--------|---------|------|------|
| **Random Forest** | [L1] | 稳健，不易过拟合 | 主分类器：基线 |
| **XGBoost** | [L1] | 上限高，适合结构化数据 | 主分类器：强对比 |
| **gcForest** (级联深度森林) | [L2] | 自动决定层数，适合小样本 | 小样本优化分类器 |
| **MultiLayerCascadeForest** | [L2]改进 | 逐层模态分支+融合 | 改进分类器 |
| **SVM (RBF)** | [L4] | 经典非线性分类器 | 辅助（仅小规模数据） |

## 验证方法

**LOSO（留一被试交叉验证）**[L1]：
- FatigueSet: 12折（11人训练 → 1人测试）
- sEMG Dataset: 13折（12人训练 → 1人测试）
- Driver EEG: 12折（11人训练 → 1人测试）

**评估指标**：Accuracy、Precision、Recall、**F1 Score**（主要指标）、AUC

## 快速开始

```bash
# 安装依赖
pip install numpy scipy scikit-learn xgboost tqdm matplotlib

# 快速测试（3人，跳过预处理）
python main.py --quick

# 全量运行（12人LOSO，含改进实验）
python main.py --full

# 单数据集
python main.py --fatigueset
python main.py --semg
python main.py --driver-eeg

# GA-PSO优化
python main.py --ga-pso

# 可视化
python main.py --viz
```

## 基线结果

| 实验 | 最佳融合 | 最佳分类器 | F1 | AUC |
|------|---------|-----------|----|-----|
| FatigueSet (ECG+EEG) | weighted | XGBoost | 0.507 | 0.650 |
| sEMG Dataset | semg_raw | XGBoost | **0.782** | 0.715 |
| Driver EEG | driver_raw | RF | 0.646 | 0.598 |

> ⚠️ 改进实验（自适应权重 / 多层级联融合）结果待更新。

## 项目结构

```
main.py                     ← 统一入口（--quick / --full / --fatigueset / --semg / --driver-eeg / --ga-pso / --viz）
src/
├── system/
│   ├── config.py           ← 系统配置（所有参数标注文献来源）
│   ├── classifiers.py     ← RF / XGBoost / gcForest / SVM / MultiLayerCascadeForest
│   └── multimodal_system.py  ← 主系统类：统一流水线
├── step1_preprocessing/   ← 信号预处理（各模态独立处理器）
├── step2_features/        ← 特征提取（各模态独立提取器 + 融合）
├── step3_models/          ← 模型训练（LOSO + GA-PSO）
├── step4_visualization/   ← 可视化
└── step5_pipeline/        ← 旧训练入口（保留兼容）
```

## 参考文献

- [L1] Kakhi, K., et al. (2024). Fatigue Monitoring Using Wearables and AI: Trends, Challenges, and Future Opportunities. *Computers in Biology and Medicine*, 195. arXiv:2412.16847.
- [L2] Zhou, Y., Chen, P., Fan, Y., & Wu, Z. (2024). A Multimodal Feature Fusion Brain Fatigue Recognition System Based on Bayes-gcForest. *Sensors*, 24(9), 2910.
- [L3] Corvini, G. & Conforto, S. (2022). A Simulation Study to Assess the Factors of Influence on Mean and Median Frequency of sEMG Signals during Muscle Fatigue. *Sensors*, 22(17), 6360.
- [L4] Wang, L., Song, F., Zhou, T.H., Hao, J., & Ryu, K.H. (2023). EEG and ECG-Based Multi-Sensor Fusion Computing for Real-Time Fatigue Driving Recognition Based on Feedback Mechanism. *Sensors*, 23(20), 8386.
- [L5] Peivandi, M., et al. (2023). Deep Learning for Detecting Multi-Level Driver Fatigue Using Physiological Signals. *Sensors*, 23(19), 8171.
