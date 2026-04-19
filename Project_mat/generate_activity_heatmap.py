#!/usr/bin/env python3
# 生成活动能力影响的生命周期热力日历图

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 活动量关键指标
activity_indicators = [
    "ADL总分",
    "活动量表总分(ADL+IADL)",
    "ADL吃饭",
    "ADL洗澡",
    "ADL穿衣",
    "ADL用厕",
    "ADL步行",
    "IADL总分",
    "IADL交通",
    "IADL理财",
    "IADL购物",
    "IADL服药",
    "IADL做饭"
]

# 年龄组
age_groups = ["40-49岁", "50-59岁", "60-69岁", "70-79岁", "80-89岁"]

# 活动量指标在不同年龄组的综合评分数据
# 数据来源：/workspace/Project_mat/问题一：不同性别和年龄组的关键指标分析.txt
activity_data = {
    "ADL总分": {
        "40-49岁": 0.1412,
        "50-59岁": 0.1171,
        "60-69岁": np.nan,
        "70-79岁": 0.1047,
        "80-89岁": 0.1229
    },
    "活动量表总分(ADL+IADL)": {
        "40-49岁": 0.1642,
        "50-59岁": np.nan,
        "60-69岁": 0.2117,
        "70-79岁": 0.1685,
        "80-89岁": 0.2136
    },
    "ADL吃饭": {
        "40-49岁": 0.0585,
        "50-59岁": 0.1268,
        "60-69岁": 0.3223,
        "70-79岁": 0.1215,
        "80-89岁": 0.1354
    },
    "ADL洗澡": {
        "40-49岁": np.nan,
        "50-59岁": np.nan,
        "60-69岁": 0.1853,
        "70-79岁": np.nan,
        "80-89岁": np.nan
    },
    "ADL穿衣": {
        "40-49岁": 0.0785,
        "50-59岁": 0.0642,
        "60-69岁": np.nan,
        "70-79岁": np.nan,
        "80-89岁": 0.1168
    },
    "ADL用厕": {
        "40-49岁": 0.1607,
        "50-59岁": 0.1966,
        "60-69岁": 0.1961,
        "70-79岁": 0.0998,
        "80-89岁": np.nan
    },
    "ADL步行": {
        "40-49岁": 0.0689,
        "50-59岁": 0.1362,
        "60-69岁": np.nan,
        "70-79岁": 0.1325,
        "80-89岁": 0.1200
    },
    "IADL总分": {
        "40-49岁": 0.1059,
        "50-59岁": 0.1055,
        "60-69岁": 0.3593,
        "70-79岁": 0.1510,
        "80-89岁": 0.1827
    },
    "IADL交通": {
        "40-49岁": 0.0461,
        "50-59岁": 0.0748,
        "60-69岁": 0.1748,
        "70-79岁": 0.0703,
        "80-89岁": 0.1191
    },
    "IADL理财": {
        "40-49岁": 0.0797,
        "50-59岁": 0.1512,
        "60-69岁": 0.1765,
        "70-79岁": 0.0760,
        "80-89岁": 0.1144
    },
    "IADL购物": {
        "40-49岁": np.nan,
        "50-59岁": 0.1285,
        "60-69岁": 0.1576,
        "70-79岁": np.nan,
        "80-89岁": np.nan
    },
    "IADL服药": {
        "40-49岁": 0.0904,
        "50-59岁": 0.0599,
        "60-69岁": 0.3090,
        "70-79岁": 0.0724,
        "80-89岁": 0.1183
    },
    "IADL做饭": {
        "40-49岁": np.nan,
        "50-59岁": np.nan,
        "60-69岁": 0.1740,
        "70-79岁": 0.0874,
        "80-89岁": 0.1097
    }
}

# 创建热图数据矩阵
heatmap_matrix = []
for indicator in activity_indicators:
    row = []
    for age_group in age_groups:
        row.append(activity_data[indicator][age_group])
    heatmap_matrix.append(row)

# 转换为numpy数组
heatmap_array = np.array(heatmap_matrix)

# 创建图形
plt.figure(figsize=(14, 10))

# 绘制热图，使用深红配色方案
sns.heatmap(
    heatmap_array,
    annot=True,
    fmt=".4f",
    cmap="Reds",
    xticklabels=age_groups,
    yticklabels=activity_indicators,
    cbar_kws={"label": "综合评分"},
    annot_kws={"size": 10}
)

# 设置标题和标签
plt.title("活动能力影响的生命周期热力日历图", fontsize=16, pad=20)
plt.xlabel("年龄组", fontsize=12)
plt.ylabel("活动量关键指标", fontsize=12)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig("活动能力影响的生命周期热力日历图.png", dpi=300, bbox_inches="tight")
print("活动能力影响的生命周期热力日历图已保存为 活动能力影响的生命周期热力日历图.png")

# 显示图片
plt.show()
