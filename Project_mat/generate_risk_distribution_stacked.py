#!/usr/bin/env python3
# 生成最终风险等级分布堆叠柱状图

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150

def generate_risk_distribution_stacked():
    """生成最终风险等级分布堆叠柱状图"""
    print("=" * 80)
    print("生成最终风险等级分布堆叠柱状图")
    print("=" * 80)
    
    # 加载数据
    try:
        print("\n加载数据...")
        # 加载三层模型预测结果
        df_result = pd.read_pickle('data/processed/three_layer_result.pkl')
        print(f"数据加载成功，形状: {df_result.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 分析风险等级分布
    print("\n分析风险等级分布...")
    
    # 统计各风险等级
    risk_counts = df_result['最终风险等级'].value_counts()
    print("风险等级分布:")
    print(risk_counts)
    
    # 直接从数据中获取各风险等级的数量
    low_risk = risk_counts.get('低风险', 0)
    medium_risk = risk_counts.get('中风险', 0)
    high_risk_clinical = risk_counts.get('临床确诊高风险', 0)
    
    # 计算总样本数
    total = len(df_result)
    print(f"\n总样本数: {total}")
    print(f"低风险: {low_risk}")
    print(f"中风险: {medium_risk}")
    print(f"临床确诊高风险: {high_risk_clinical}")
    
    # 准备堆叠数据
    # 注意堆叠顺序：底层到顶层
    categories = ['风险等级分布']
    
    # 堆叠数据（从下到上）
    low_risk_data = [low_risk]
    medium_risk_data = [medium_risk]
    high_risk_clinical_data = [high_risk_clinical]
    
    # 定义颜色
    colors = ['#4CAF50', '#FFC107', '#F44336']  # 绿色、黄色、深红色
    
    # 计算百分比
    low_risk_pct = (low_risk / total) * 100
    medium_risk_pct = (medium_risk / total) * 100
    high_risk_clinical_pct = (high_risk_clinical / total) * 100
    
    # 绘制堆叠柱状图
    print("\n绘制堆叠柱状图...")
    plt.figure(figsize=(10, 8))
    
    # 绘制堆叠柱子
    bottom = 0
    
    # 低风险
    p1 = plt.bar(categories, low_risk_data, color=colors[0], width=0.6, label='低风险')
    bottom += low_risk
    
    # 中风险
    p2 = plt.bar(categories, medium_risk_data, color=colors[1], width=0.6, bottom=low_risk, label='中风险')
    bottom += medium_risk
    
    # 临床确诊高风险
    p3 = plt.bar(categories, high_risk_clinical_data, color=colors[2], width=0.6, bottom=low_risk + medium_risk, label='临床确诊高风险')
    
    # 添加标注
    print("\n添加标注...")
    
    # 低风险标注
    plt.text(0, low_risk/2, f'N={low_risk}\n{low_risk_pct:.1f}%', ha='center', va='center', color='white', fontweight='bold')
    
    # 中风险标注
    plt.text(0, low_risk + medium_risk/2, f'N={medium_risk}\n{medium_risk_pct:.1f}%', ha='center', va='center', color='black', fontweight='bold')
    
    # 临床确诊高风险标注
    plt.text(0, low_risk + medium_risk + high_risk_clinical/2, f'N={high_risk_clinical}\n{high_risk_clinical_pct:.1f}%', ha='center', va='center', color='white', fontweight='bold')
    
    # 设置Y轴范围和标签
    plt.ylim(0, total)
    plt.ylabel('样本数', fontsize=12, fontweight='bold')
    
    # 添加百分比Y轴
    ax2 = plt.gca().twinx()
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('百分比 (%)', fontsize=12, fontweight='bold')
    
    # 添加标题
    plt.title('最终风险等级分布', fontsize=14, fontweight='bold')
    
    # 添加图例
    plt.legend(loc='upper left', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = '/workspace/Project_mat/最终风险等级分布堆叠柱状图.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n最终风险等级分布堆叠柱状图已保存到: {output_path}")
    
    plt.show()
    
    return output_path

if __name__ == '__main__':
    generate_risk_distribution_stacked()
