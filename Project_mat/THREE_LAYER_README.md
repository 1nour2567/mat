# 三层融合预警模型架构说明

## 架构概述

本系统实现了三层融合预警模型，用于基于中医体质、活动量表和代谢指标预测高血脂风险。模型已升级到2.0版本，包含更完善的特征工程！

```
输入：中老年个体多维特征
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  [新增] 特征工程层 (Feature Engineering Layer)               │
│  - 创建中西医交叉特征：                                      │
│    - 痰湿质×BMI、痰湿质×活动量表、痰湿质×血尿酸             │
│    - 气虚质×BMI、气虚质×活动量表                             │
│  - 同时保留原始特征，用于后续各层                           │
└─────────────────────────────────────────────────────────────┘
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
│  - 使用扩展的特征（25个）：                                   │
│    - 基础特征（20个）+ 交叉特征（5个）+ 尿酸异常              │
│  - LightGBM + 5折交叉验证 + Focal Loss（可选）                │
│  - 输出潜在风险概率 p_hat ∈ [0,1]                           │
│    (不依赖血脂的潜在生理病理风险概率)                        │
└─────────────────────────────────────────────────────────────┘
         │
         │ p_hat
         ▼
┌─────────────────────────────────────────────────────────────┐
│  第三层：中医功能层 (TCM Functional Layer)                   │
│  - 若 p_hat ∈ [0.20, 0.60]（不确定区间），执行修正：         │
│    - 【升档规则】痰湿质 ≥60 且 活动总分 <40 → 高风险          │
│    - 【降档规则】痰湿质 <17 且 活动总分 ≥60 → 低风险          │
│  - 否则，按概率划分等级：                                     │
│    - <0.20 → 低风险，0.20-0.60 → 中风险，>0.60 → 高风险      │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
最终风险等级
  - 低风险（含「中医支持降档」）
  - 中风险
  - 高风险（含「临床确诊高风险」和「中医预警高风险」）
```

## 模型特征详解

### 血脂屏蔽清单（严格隔离，防止标签泄露）
```
血脂相关特征（禁止进入第二层模型）：
- 原始血脂：HDL-C、LDL-C、TG、TC
- 派生指标：AIP、TC/HDL、non-HDL-C、LDL/HDL、TG/HDL
- 缩尾特征：non-HDL-C_缩尾、AIP_缩尾、TC/HDL比值_缩尾等
- 标记特征：TC异常、TG异常、LDL-C异常、HDL-C异常、血脂异常项数
```

### 模型可用特征（25个）

| 特征类别 | 特征列表 |
|---------|---------|
| 中医体质（9个） | 平和质、气虚质、阳虚质、阴虚质、痰湿质、湿热质、血瘀质、气郁质、特禀质 |
| 活动量表（3个） | ADL总分、IADL总分、活动量表总分（ADL总分+IADL总分） |
| 人口学（4个） | 年龄组、性别、吸烟史、饮酒史 |
| 代谢指标（4个） | 空腹血糖、血尿酸、BMI、尿酸异常 |
| 中西医交叉（5个） | 痰湿质×BMI、痰湿质×活动量表、痰湿质×血尿酸、气虚质×BMI、气虚质×活动量表 |

**总计：25个特征**

## 文件结构

```
Project_mat/
├── src/
│   ├── three_layer_architecture.py  # 三层架构核心实现（V2.0）
│   ├── preprocessing.py             # 数据预处理
│   ├── feature_engineering.py       # 特征工程（补充）
│   ├── risk_model.py                # 旧版风险模型
│   ├── visualization.py             # 可视化工具
│   └── intervention_optimizer.py    # 干预优化
├── three_layer_main.py              # 三层架构主流程
├── config/
│   └── constants.py                 # 常量配置
├── data/
│   ├── raw/
│   │   └── 附件1：样例数据.xlsx
│   └── processed/
│       └── three_layer_result.pkl   # 预测结果
├── test_feature_engineering.py      # 特征工程验证脚本
├── 模型改进总结.md                  # V2.0改进说明
└── THREE_LAYER_README.md            # 本文档
```

## 使用说明

### 快速开始

```python
from src.three_layer_architecture import TripleLayerPredictor
import pandas as pd

# 1. 加载和预处理数据
df = pd.read_excel('data/raw/附件1：样例数据.xlsx')

# 2. 初始化预测器
predictor = TripleLayerPredictor()

# 3. 训练模型（自动完成特征工程）
predictor.fit(df, target_col='高血脂症二分类标签')

# 4. 预测（自动完成特征工程）
result = predictor.predict(df)

# 5. 查看结果
print(result['最终风险等级'].value_counts())
```

### 特征工程函数独立使用

```python
from src.three_layer_architecture import create_tcm_interactions

# 仅创建交叉特征
df_with_interactions = create_tcm_interactions(df)
print("新增交叉特征：", [col for col in df_with_interactions.columns if "×" in col])
```

### 运行完整流程

```bash
python three_layer_main.py
```

## 各层功能详解

### 特征工程层（新增）

该层自动生成中西医交叉特征，增强模型表达能力：

```python
from src.three_layer_architecture import create_tcm_interactions

# 自动创建交叉特征
df_with_features = create_tcm_interactions(df)
```

**交叉特征说明**：
- 痰湿质×BMI：痰湿体质与肥胖的交互效应
- 痰湿质×活动量表：痰湿体质与活动能力的交互
- 痰湿质×血尿酸：痰湿体质与代谢的协同效应
- 气虚质×BMI：气虚体质与肥胖的协同
- 气虚质×活动量表：气虚体质与活动能力的交互

### 第一层：临床规则层 (ClinicalRuleLayer)

该层基于西医金标准进行初步判断：

```python
from src.three_layer_architecture import ClinicalRuleLayer

# 计算血脂异常项数
df_with_abnormalities = ClinicalRuleLayer.calculate_lipid_abnormality_count(df)

# 应用临床规则
df_processed, clinical_high_risk = ClinicalRuleLayer.apply_clinical_rules(df)
```

**血脂异常判定标准**（根据题目提供的标准）：
- TC（总胆固醇）异常：<3.1 或 >6.2
- TG（甘油三酯）异常：<0.56 或 >1.7
- LDL-C（低密度脂蛋白）异常：<2.07 或 >3.1
- HDL-C（高密度脂蛋白）异常：<1.04 或 >1.55

### 第二层：统计模型层 (LightGBMPredictionLayer)

该层基于非血脂特征预测潜在风险：

```python
from src.three_layer_architecture import LightGBMPredictionLayer

# 查看允许使用的特征
# 现在包含：基础特征20个 + 尿酸异常 + 交叉特征5个 = 25个

# 训练模型
model = LightGBMPredictionLayer(n_splits=5)
model.train(df, target_col='高血脂症二分类标签')

# 预测概率
probs = model.predict_probability(df)
```

**特征屏蔽策略（V2.0更严格）**：
- 严格屏蔽：TC, TG, LDL, HDL 原始四项
- 严格屏蔽：血脂异常项数、AIP、TC/HDL等派生项
- 严格屏蔽：所有血脂缩尾特征和异常标记
- 允许使用：体质积分、活动量表、年龄、性别、BMI、血糖、尿酸、尿酸异常、中西医交叉特征

### 第三层：中医功能层 (TCMFunctionalLayer)

该层对不确定区间进行中医规则修正：

```python
from src.three_layer_architecture import TCMFunctionalLayer

tcm_layer = TCMFunctionalLayer()

df_result = tcm_layer.apply_tcm_rules(df, predicted_probs)
```

**中医修正规则**：
- 升档（邪盛正衰）：痰湿质 ≥60 且 活动总分 <40 → 高风险
- 降档（正盛邪微）：痰湿质 <17 且 活动总分 ≥60 → 低风险
- 不确定区间：[0.20, 0.60]

## 主要API

### TripleLayerPredictor 类

主预测器，整合所有三层功能（V2.0已升级）。

```python
class TripleLayerPredictor:
    def fit(self, df, target_col='高血脂症二分类标签'):
        """训练模型（自动完成特征工程）"""
        ...
    
    def predict(self, df):
        """预测，返回包含结果的数据框（自动完成特征工程）"""
        ...
        
    def predict_instance(self, row):
        """预测单个实例（支持交叉特征）"""
        ...
```

### 返回结果列

预测后的数据框将包含以下新列：
- `血脂异常项数`：计算得到的血脂异常项数
- `临床确诊高风险`：第一层输出的标记
- `模型预测概率`：第二层输出的概率
- `最终风险等级`：第三层修正后的最终等级

## 版本历史

### V2.0（当前版本）
- 新增特征工程层，创建5个中西医交叉特征
- 扩展模型可用特征从20个到25个
- 更严格的血脂屏蔽清单（19个血脂相关特征）
- 新增尿酸异常特征
- 优化所有预测流程，自动支持特征工程

### V1.0（初始版本）
- 基础三层架构
- 20个基础特征
- 7个血脂特征屏蔽

## 注意事项

1. 数据必须包含的列：
   - 体质类：痰湿质、气虚质、阳虚质、...
   - 活动量表：ADL总分、IADL总分、活动量表总分（ADL总分+IADL总分）
   - 血脂四项：TC（总胆固醇）、TG（甘油三酯）、LDL-C、HDL-C
   - 其他：BMI、空腹血糖、血尿酸、年龄、性别、尿酸异常
   - 目标：高血脂症二分类标签

2. 第二层严格屏蔽血脂相关特征（19个），目的是模拟无血脂化验单情况下的预测能力。V2.0版本的屏蔽更彻底！

3. 第三层的修正只在模型不确定区间[0.20, 0.60]内执行。

4. 所有预测流程会自动完成特征工程，无需手动调用！

