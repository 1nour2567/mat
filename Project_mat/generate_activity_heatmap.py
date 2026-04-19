#!/usr/bin/env python3
# 生成活动能力影响的生命周期热力日历图

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def create_activity_heatmap():
    # 年龄组
    age_groups = ['40-49岁', '50-59岁', '60-69岁', '70-79岁', '80-89岁']
    
    # 活动量关键指标（前6个）
    activity_indicators = [
        'ADL总分',
        '活动量表总分',
        'ADL吃饭',
        'ADL洗澡',
        'ADL穿衣',
        'IADL总分'
    ]
    
    # 数据矩阵（行：指标，列：年龄组）
    # 数据来自文本文件中的年龄组分组分析
    data = np.array([
        [0.1412, 0.1171, 0.1047, 0.1047, 0.1229],  # ADL总分
        [0.1642, 0.1263, 0.2117, 0.1685, 0.2136],  # 活动量表总分
        [0.0585, 0.1268, 0.3223, 0.1215, 0.1354],  # ADL吃饭
        [0.0903, 0.0773, 0.1853, 0.0903, 0.1168],  # ADL洗澡
        [0.0785, 0.0642, 0.1740, 0.0773, 0.1168],  # ADL穿衣
        [0.1059, 0.1055, 0.3593, 0.1510, 0.1827]   # IADL总分
    ])
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建热图
    sns.heatmap(
        data,
        annot=True,
        fmt='.4f',
        cmap='Reds',
        xticklabels=age_groups,
        yticklabels=activity_indicators,
        ax=ax,
        cbar_kws={'label': '综合评分'},
        annot_kws={'size': 12}
    )
    
    # 设置标题和标签
    ax.set_xlabel('年龄组', fontsize=14, fontweight='bold')
    ax.set_ylabel('活动量指标', fontsize=14, fontweight='bold')
    ax.set_title('活动能力影响的生命周期热力日历图', fontsize=16, fontweight='bold', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('活动能力影响的生命周期热力日历图.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    print('活动能力影响的生命周期热力日历图已成功保存！')

if __name__ == "__main__":
    create_activity_heatmap()
