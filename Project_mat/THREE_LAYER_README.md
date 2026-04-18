# 三层融合预警模型架构说明

## 架构概述

本系统实现了三层融合预警模型，用于基于中医体质、活动量表和代谢指标预测高血脂风险。

```
输入：中老年个体多维特征
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  第一层：临床规则层 (Clinical Rule Layer)                   │
│  - 计算血脂异常项数 N_i = I(TC异常) + I(TG异常) +          │
│    I(LDL异常) + I(HDL异常)                                  │
│  - 若 N_i ≥ 1，标记为「临床确诊高风险」                       │
│  - 全体样本继续流向第二层                                   │
└─────────────────────────────────────────────────────────────┘
         │
         │ 全体样本 (标签 y ∈ {0,1})
         ▼
┌─────────────────────────────────────────────────────────────┐
│  第二层：统计模型层 (LightGBM Prediction Layer)              │
│  - 严格屏蔽血脂相关指标（TC, TG, LDL, HDL及其派生项）         │
│  - 使用允许的特征：体质积分、活动量表、人口学、BMI、血糖、尿酸  │
│  - LightGBM + 5折交叉验证 + Focal Loss（可选）                │
│  - 输出潜在风险概率 p_hat ∈ [0,1]                           │
│    (不依赖血脂的潜在生理病理风险概率)                        │
└─────────────────────────────────────────────────────────────┘
         │
         │ p_hat
         ▼
┌─────────────────────────────────────────────────────────────┐
│  第三层：中医功能层 (TCM Functional Layer)                   │
│  - 若 p_hat ∈ [0.35, 0.65]（不确定区间），执行修正：          │
│    - 【升档规则】痰湿质 ≥80 且 活动总分 <40 → 高风险          │
│    - 【降档规则】痰湿质 <60 且 活动总分 ≥60 → 低风险          │
│  - 否则，按概率划分等级：                                     │
│    - <0.35 → 低风险，0.35-0.65 → 中风险，>0.65 → 高风险      │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
最终风险等级
  - 低风险
  - 中风险
  - 高风险（含「临床确诊高风险」和「中医预警高风险」）
```

## 文件结构

```
Project_mat/
├── src/
│   ├── three_layer_architecture.py  # 三层架构核心实现
│   ├── preprocessing.py             # 数据预处理
│   ├── feature_engineering.py       # 特征工程
│   ├── risk_model.py                # 旧版风险模型
│   ├── visualization.py             # 可视化工具
│   └── intervention_optimizer.py    # 干预优化
├── three_layer_main.py              # 三层架构主流程
├── config/
│   └── constants.py                 # 常量配置
└── data/
    ├── raw/
    │   └── 附件1：样例数据.xlsx
    └── processed/
        └── three_layer_result.pkl   # 预测结果
```

## 使用说明

### 快速开始

```python
from src.three_layer_architecture import ThreeLayerRiskPredictor
import pandas as pd

# 1. 加载和预处理数据
df = pd.read_excel('data/raw/附件1：样例数据.xlsx')

# 2. 初始化预测器
predictor = ThreeLayerRiskPredictor()

# 3. 训练模型
predictor.fit(df, target_col='高血脂症二分类标签')

# 4. 预测
result = predictor.predict(df)

# 5. 查看结果
print(result['最终风险等级'].value_counts())
```

### 运行完整流程

```bash
python three_layer_main.py
```

## 各层功能详解

### 第一层：临床规则层 (ClinicalRuleLayer)

该层基于西医金标准进行初步判断：

```python
from src.three_layer_architecture import ClinicalRuleLayer

# 计算血脂异常项数
df_with_abnormalities = ClinicalRuleLayer.calculate_lipid_abnormality_count(df)

# 应用临床规则
df_processed, clinical_high_risk = ClinicalRuleLayer.apply_clinical_rules(df)
```

**血脂异常判定标准**（根据用户提供的标准）：
- TC（总胆固醇）异常：<3.1 或 >6.2
- TG（甘油三酯）异常：<0.56 或 >1.7
- LDL-C（低密度脂蛋白）异常：<2.07 或 >3.1
- HDL-C（高密度脂蛋白）异常：<1.04 或 >1.55

### 第二层：统计模型层 (LightGBMPredictionLayer)

该层基于非血脂特征预测潜在风险：

```python
from src.three_layer_architecture import LightGBMPredictionLayer

# 查看允许使用的特征
allowed_features = LightGBMPredictionLayer.get_allowed_features(df)

# 训练模型
model = LightGBMPredictionLayer(n_splits=5)
model.train(df, target_col='高血脂症二分类标签')

# 预测概率
probs = model.predict_probability(df)
```

**特征屏蔽策略**：
- 严格屏蔽：TC, TG, LDL, HDL 原始四项
- 严格屏蔽：血脂异常项数、AIP、TC/HDL等派生项
- 允许使用：体质积分、活动量表、年龄、性别、BMI、血糖、尿酸等

### 第三层：中医功能层 (TCMFunctionalLayer)

该层对不确定区间进行中医规则修正：

```python
from src.three_layer_architecture import TCMFunctionalLayer

tcm_layer = TCMFunctionalLayer()

df_result = tcm_layer.apply_tcm_rules(df, predicted_probs)
```

**中医修正规则**：
- 升档（邪盛正衰）：痰湿质 ≥80 且 活动总分 <40 → 高风险
- 降档（正盛邪微）：痰湿质 <60 且 活动总分 ≥60 → 低风险
- 不确定区间：[0.35, 0.65]

## 主要API

### ThreeLayerRiskPredictor 类

主预测器，整合所有三层功能。

```python
class ThreeLayerRiskPredictor:
    def fit(self, df, target_col='高血脂症二分类标签'):
        """训练模型"""
        ...
    
    def predict(self, df):
        """预测，返回包含结果的数据框"""
        ...
```

### 返回结果列

预测后的数据框将包含以下新列：
- `血脂异常项数`：计算得到的血脂异常项数
- `临床确诊高风险`：第一层输出的标记
- `模型预测概率`：第二层输出的概率
- `最终风险等级`：第三层修正后的最终等级

## 注意事项

1. 数据必须包含的列：
   - 体质类：痰湿质、气虚质、阳虚质、...
   - 活动量表：ADL总分、IADL总分、活动量表总分（ADL总分+IADL总分）
   - 血脂四项：TC（总胆固醇）、TG（甘油三酯）、LDL-C、HDL-C
   - 其他：BMI、空腹血糖、血尿酸、年龄、性别等
   - 目标：高血脂症二分类标签

2. 第二层严格屏蔽血脂相关特征，目的是模拟无血脂化验单情况下的预测能力。

3. 第三层的修正只在模型不确定区间[0.35, 0.65]内执行。
