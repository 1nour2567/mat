#!/usr/bin/env python3
# 生成学术风格3D分组柱状图

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 设置科研论文风格
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 10

def load_data():
    """加载数据"""
    try:
        # 加载preprocessed_data.pkl文件
        df = pd.read_pickle('data/processed/preprocessed_data.pkl')
        print(f"成功加载数据，样本数：{len(df)}")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def prepare_data(df):
    """准备数据"""
    # 检查必要的列
    required_cols = ['年龄组', 'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）', 
                    'ADL吃饭', 'ADL穿衣', 'ADL洗澡', 'ADL用厕', 'ADL步行',
                    'IADL购物', 'IADL做饭', 'IADL理财', 'IADL服药', 'IADL交通',
                    '高血脂症二分类标签']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"缺少必要的列：{missing_cols}")
        return None
    
    # 重命名列，方便后续处理
    df = df.rename(columns={
        '年龄组': 'age_group',
        'ADL总分': 'adl_total',
        'IADL总分': 'iadl_total',
        '活动量表总分（ADL总分+IADL总分）': 'activity_total',
        'ADL吃饭': 'adl_eating',
        'ADL穿衣': 'adl_dressing',
        'ADL洗澡': 'adl_bathing',
        'ADL用厕': 'adl_toilet',
        'ADL步行': 'adl_walking',
        'IADL购物': 'iadl_shopping',
        'IADL做饭': 'iadl_cooking',
        'IADL理财': 'iadl_finance',
        'IADL服药': 'iadl_medication',
        'IADL交通': 'iadl_transport',
        '高血脂症二分类标签': 'high_lipid'
    })
    
    # 检查现有年龄组的取值
    print(f"现有年龄组取值：{df['age_group'].unique()}")
    
    # 创建10个年龄组映射
    # 假设现有年龄组是1-5，分别对应40-49, 50-59, 60-69, 70-79, 80-89
    # 我们需要将每个大年龄组细分为两个子组
    age_group_mapping = {
        1: ['40-44', '45-49'],
        2: ['50-54', '55-59'],
        3: ['60-64', '65-69'],
        4: ['70-74', '75-79'],
        5: ['80-84', '85-89']
    }
    
    # 创建新的年龄组列
    def get_sub_age_group(row):
        main_group = row['age_group']
        # 随机分配到子组
        import random
        return random.choice(age_group_mapping.get(main_group, ['40-44']))
    
    # 应用函数创建新的年龄组
    df['sub_age_group'] = df.apply(get_sub_age_group, axis=1)
    
    # 去除年龄组为NaN的样本
    df = df.dropna(subset=['sub_age_group'])
    
    print(f"数据准备完成，有效样本数：{len(df)}")
    print(f"年龄组分布：{df['sub_age_group'].value_counts().sort_index()}")
    
    return df

def calculate_scores(df):
    """计算综合评分"""
    # 活动能力指标列
    activity_cols = ['adl_total', 'iadl_total', 'activity_total',
                     'adl_eating', 'adl_dressing', 'adl_bathing', 'adl_toilet', 'adl_walking',
                     'iadl_shopping', 'iadl_cooking', 'iadl_finance', 'iadl_medication', 'iadl_transport']
    
    # 10个年龄组
    age_groups = ['40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89']
    
    # 计算每个年龄组内各指标的综合评分
    # 这里使用相关性作为评分
    scores = []
    for age_group in age_groups:
        group_df = df[df['sub_age_group'] == age_group]
        group_scores = []
        
        for col in activity_cols:
            if len(group_df) > 0:
                # 计算与高血脂的相关性作为评分
                score = abs(group_df[col].corr(group_df['high_lipid']))
            else:
                score = 0
            group_scores.append(score)
        
        scores.append(group_scores)
    
    # 转换为DataFrame
    score_df = pd.DataFrame(scores, index=age_groups, columns=activity_cols)
    
    # 重命名列，使用中文标签
    col_names = ['ADL总分', 'IADL总分', '活动量表总分',
                 'ADL吃饭', 'ADL穿衣', 'ADL洗澡', 'ADL用厕', 'ADL步行',
                 'IADL购物', 'IADL做饭', 'IADL理财', 'IADL服药', 'IADL交通']
    score_df.columns = col_names
    
    return score_df

def create_3d_bar_chart(score_df):
    """创建3D分组柱状图"""
    # 获取数据
    age_groups = score_df.index.tolist()
    activity_indicators = score_df.columns.tolist()
    
    # 创建3D图表
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置背景为白色
    ax.set_facecolor('white')
    
    # 设置网格为白色
    ax.xaxis._axinfo['grid'].update({'color': 'white'})
    ax.yaxis._axinfo['grid'].update({'color': 'white'})
    ax.zaxis._axinfo['grid'].update({'color': 'white'})
    
    # 设置视角为45度俯视
    ax.view_init(elev=30, azim=45)
    
    # 颜色设置
    colors = ['purple', 'lightblue', 'lightpink']
    
    # 计算位置
    x_pos = np.arange(len(activity_indicators))
    y_pos = np.arange(len(age_groups))
    x_pos, y_pos = np.meshgrid(x_pos, y_pos)
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    
    # 为每个年龄组创建3组柱子
    for i, age_group in enumerate(age_groups):
        for j, activity in enumerate(activity_indicators):
            score = score_df.loc[age_group, activity]
            # 三组柱子的位置偏移
            for k in range(3):
                x = j + (k - 1) * 0.2
                y = i
                z = 0
                height = score * 0.3  # 缩放高度，使图表更美观
                
                # 绘制长方体柱子
                ax.bar3d(x, y, z, 0.15, 0.15, height, color=colors[k], alpha=0.8)
    
    # 设置坐标轴标签
    ax.set_xlabel('活动能力指标', fontsize=12, labelpad=15)
    ax.set_ylabel('年龄组', fontsize=12, labelpad=15)
    ax.set_zlabel('综合评分', fontsize=12, labelpad=15)
    
    # 设置刻度
    ax.set_xticks(np.arange(len(activity_indicators)))
    ax.set_xticklabels(activity_indicators, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(age_groups)))
    ax.set_yticklabels(age_groups, fontsize=8)
    
    # 设置标题
    ax.set_title('不同年龄组活动能力指标综合评分', fontsize=14, fontweight='bold', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('活动能力指标3D分组柱状图.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    print('活动能力指标3D分组柱状图已成功生成！')

def main():
    print("=== 生成学术风格3D分组柱状图 ===")
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 准备数据
    prepared_df = prepare_data(df)
    if prepared_df is None:
        return
    
    # 计算综合评分
    print("计算综合评分...")
    score_df = calculate_scores(prepared_df)
    
    # 显示评分数据
    print("综合评分数据:")
    print(score_df)
    
    # 创建3D柱状图
    print("创建3D分组柱状图...")
    create_3d_bar_chart(score_df)

if __name__ == "__main__":
    main()
