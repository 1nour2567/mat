#!/usr/bin/env python3
# 生成核心高危特征组合风险对比柱状图

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150

def generate_core_feature_combinations():
    """生成核心高危特征组合风险对比柱状图"""
    print("=" * 80)
    print("生成核心高危特征组合风险对比柱状图")
    print("=" * 80)
    
    # 加载数据
    try:
        print("\n加载数据...")
        df = pd.read_pickle('data/processed/preprocessed_data.pkl')
        print(f"数据加载成功，形状: {df.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 定义组合条件
    print("\n计算各组高风险比例...")
    conditions = {
        '对照组\n(痰湿低+活动高)': (df['痰湿质']<30) & (df['活动量表总分（ADL总分+IADL总分）']>=60),
        '组合A\n(痰湿高+活动低)': (df['痰湿质']>=60) & (df['活动量表总分（ADL总分+IADL总分）']<40),
        '组合B\n(组合A+血脂异常≥1)': (df['痰湿质']>=60) & (df['活动量表总分（ADL总分+IADL总分）']<40) & (df['血脂异常项数']>=1),
        '组合C\n(痰湿中+尿酸高+血脂异常≥2)': (df['痰湿质']>=30) & (df['痰湿质']<60) & (df['血尿酸']>420) & (df['血脂异常项数']>=2)
    }
    
    # 计算各组高风险比例
    high_risk_ratios = {}
    group_sizes = {}
    for name, cond in conditions.items():
        subset = df[cond]
        group_sizes[name] = len(subset)
        if len(subset) > 0:
            high_risk_ratios[name] = (subset['高血脂症二分类标签'] == 1).mean()
        else:
            high_risk_ratios[name] = 0
        print(f"{name}: 样本数={group_sizes[name]}, 高风险比例={high_risk_ratios[name]:.4f}")
    
    # 准备数据
    labels = list(high_risk_ratios.keys())
    values = list(high_risk_ratios.values())
    sizes = list(group_sizes.values())
    
    # 定义颜色
    colors = ['gray', 'orange', 'crimson', 'purple']
    
    # 绘制柱状图
    print("\n绘制柱状图...")
    plt.figure(figsize=(12, 8))
    
    # 绘制柱子
    bars = plt.bar(labels, values, color=colors, width=0.6)
    
    # 添加百分比标签
    for bar, value, size in zip(bars, values, sizes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.1%}\n(n={size})',
                 ha='center', va='bottom', fontsize=10)
    
    # 设置Y轴范围
    plt.ylim(0, 1.1)
    
    # 添加标题和标签
    plt.xlabel('特征组合', fontsize=12, fontweight='bold')
    plt.ylabel('高风险实际比例', fontsize=12, fontweight='bold')
    plt.title('核心高危特征组合风险对比', fontsize=14, fontweight='bold')
    
    # 添加网格线
    plt.grid(axis='y', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = '/workspace/Project_mat/核心高危特征组合风险对比柱状图.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n核心高危特征组合风险对比柱状图已保存到: {output_path}")
    
    plt.show()
    
    return output_path

if __name__ == '__main__':
    generate_core_feature_combinations()
