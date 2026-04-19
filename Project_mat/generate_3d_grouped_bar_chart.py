#!/usr/bin/env python3
# 生成学术风格的3D分组柱状图

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["font.size"] = 10  # 调整字体大小
plt.rcParams["axes.labelsize"] = 12  # 调整坐标轴标签大小
plt.rcParams["axes.titlesize"] = 14  # 调整标题大小

def load_data():
    """加载数据"""
    try:
        # 加载preprocessed_data.pkl文件
        df = pd.read_pickle('/workspace/Project_mat/data/processed/preprocessed_data.pkl')
        print(f"成功加载数据，样本数：{len(df)}")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def prepare_data(df):
    """准备数据"""
    # 检查必要的列
    required_cols = ['年龄组', 'TC（总胆固醇）', 'TG（甘油三酯）', 'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', '血尿酸', '空腹血糖', 'BMI',
                    'ADL总分', '活动量表总分（ADL总分+IADL总分）', 'ADL吃饭', 'ADL用厕', 'ADL洗澡', 'ADL穿衣', 'ADL步行', 'IADL总分',
                    '高血脂症二分类标签']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"缺少必要的列：{missing_cols}")
        return None
    
    # 重命名列，方便后续处理
    df = df.rename(columns={
        '年龄组': 'age_group',
        'TC（总胆固醇）': 'TC',
        'TG（甘油三酯）': 'TG',
        'HDL-C（高密度脂蛋白）': 'HDL',
        'LDL-C（低密度脂蛋白）': 'LDL',
        '血尿酸': 'UA',
        '空腹血糖': 'FPG',
        'BMI': 'BMI',
        'ADL总分': 'ADL_total',
        '活动量表总分（ADL总分+IADL总分）': 'activity_total',
        'ADL吃饭': 'ADL_eating',
        'ADL用厕': 'ADL_toilet',
        'ADL洗澡': 'ADL_bathing',
        'ADL穿衣': 'ADL_dressing',
        'ADL步行': 'ADL_walking',
        'IADL总分': 'IADL_total',
        '高血脂症二分类标签': 'high_lipid'
    })
    
    # 检查现有年龄组的取值
    print(f"现有年龄组取值：{df['age_group'].unique()}")
    
    return df

def calculate_scores(df):
    """计算各指标的综合评分"""
    # 血常规体检指标
    blood_indices = ['TC', 'TG', 'HDL', 'LDL', 'UA', 'FPG', 'BMI']
    # 活动量指标
    activity_indices = ['ADL_total', 'activity_total', 'ADL_eating', 'ADL_toilet', 'ADL_bathing', 'ADL_dressing', 'ADL_walking', 'IADL_total']
    
    # 按年龄组分组
    age_groups = sorted(df['age_group'].unique())
    age_group_names = ['40-49', '50-59', '60-69', '70-79', '80-89']
    
    # 计算每个年龄组内各指标的综合评分
    # 这里使用与高血脂的相关性作为评分
    scores = {}
    
    for i, age_group in enumerate(age_groups):
        group_df = df[df['age_group'] == age_group]
        age_name = age_group_names[i]
        scores[age_name] = {}
        
        # 计算血常规指标评分
        for idx in blood_indices:
            if len(group_df) > 0:
                corr = abs(group_df[idx].corr(group_df['high_lipid']))
                scores[age_name][idx] = corr
            else:
                scores[age_name][idx] = 0
        
        # 计算活动量指标评分
        for idx in activity_indices:
            if len(group_df) > 0:
                corr = abs(group_df[idx].corr(group_df['high_lipid']))
                scores[age_name][idx] = corr
            else:
                scores[age_name][idx] = 0
    
    return scores, blood_indices, activity_indices, age_group_names

def select_top_10_indices(scores, blood_indices, activity_indices):
    """选择综合评分前10的指标"""
    # 计算每个指标的平均评分
    avg_scores = {}
    for idx in blood_indices + activity_indices:
        total = 0
        count = 0
        for age_group in scores:
            total += scores[age_group].get(idx, 0)
            count += 1
        avg_scores[idx] = total / count
    
    # 排序并选择前10个
    sorted_indices = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top_indices = [idx for idx, score in sorted_indices]
    
    # 重命名指标，使用中文标签
    index_labels = {
        'TC': 'TC（总胆固醇）',
        'TG': 'TG（甘油三酯）',
        'HDL': 'HDL-C（高密度脂蛋白）',
        'LDL': 'LDL-C（低密度脂蛋白）',
        'UA': '血尿酸',
        'FPG': '空腹血糖',
        'BMI': 'BMI',
        'ADL_total': 'ADL总分',
        'activity_total': '活动量表总分',
        'ADL_eating': 'ADL吃饭',
        'ADL_toilet': 'ADL用厕',
        'ADL_bathing': 'ADL洗澡',
        'ADL_dressing': 'ADL穿衣',
        'ADL_walking': 'ADL步行',
        'IADL_total': 'IADL总分'
    }
    
    top_labels = [index_labels.get(idx, idx) for idx in top_indices]
    
    return top_indices, top_labels

def create_3d_grouped_bar_chart(scores, top_indices, top_labels, age_group_names):
    """创建3D分组柱状图"""
    # 设置图表尺寸
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置背景为白色
    ax.set_facecolor('white')
    
    # 设置网格为白色
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(color='white', linestyle='-', linewidth=0.5)
    
    # 颜色设置
    colors = ['#E6E6FA', '#B0E0E6', '#FFB6C1']  # 浅紫色、浅蓝色、浅粉色
    
    # 准备数据
    x_pos = np.arange(len(top_indices))
    y_pos = np.arange(len(age_group_names))
    x_pos, y_pos = np.meshgrid(x_pos, y_pos)
    
    # 计算柱子位置
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(x_pos)
    
    # 柱子宽度和深度
    dx = 0.25
    dy = 0.25
    
    # 绘制柱子
    for i, age_group in enumerate(age_group_names):
        for j, idx in enumerate(top_indices):
            z = scores[age_group].get(idx, 0)
            ax.bar3d(j, i, 0, dx, dy, z, color=colors[i % 3], edgecolor='none', alpha=0.8)
    
    # 设置坐标轴标签
    ax.set_xlabel('关键指标', fontweight='bold')
    ax.set_ylabel('年龄组', fontweight='bold')
    ax.set_zlabel('综合评分', fontweight='bold')
    
    # 设置坐标轴刻度
    ax.set_xticks(np.arange(len(top_indices)))
    ax.set_xticklabels(top_labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(age_group_names)))
    ax.set_yticklabels(age_group_names)
    
    # 设置视角为斜45度俯视
    ax.view_init(elev=30, azim=45)
    
    # 设置标题
    ax.set_title('不同年龄组关键指标综合评分', fontweight='bold', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('/workspace/Project_mat/学术风格3D分组柱状图.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    print('学术风格3D分组柱状图已成功生成！')

def main():
    print("=== 生成学术风格的3D分组柱状图 ===")
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 准备数据
    prepared_df = prepare_data(df)
    if prepared_df is None:
        return
    
    # 计算评分
    print("计算各指标的综合评分...")
    scores, blood_indices, activity_indices, age_group_names = calculate_scores(prepared_df)
    
    # 选择前10个指标
    print("选择综合评分前10的指标...")
    top_indices, top_labels = select_top_10_indices(scores, blood_indices, activity_indices)
    print(f"前10个关键指标：{top_labels}")
    
    # 创建3D分组柱状图
    print("创建学术风格3D分组柱状图...")
    create_3d_grouped_bar_chart(scores, top_indices, top_labels, age_group_names)

if __name__ == "__main__":
    main()
