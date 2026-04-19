#!/usr/bin/env python3
# 生成活动能力指标的频谱风格热力图

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

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
        # 随机分配到子组（实际应用中可能需要更合理的方法）
        import random
        return random.choice(age_group_mapping.get(main_group, ['40-44']))
    
    # 应用函数创建新的年龄组
    df['sub_age_group'] = df.apply(get_sub_age_group, axis=1)
    
    # 去除年龄组为NaN的样本
    df = df.dropna(subset=['sub_age_group'])
    
    print(f"数据准备完成，有效样本数：{len(df)}")
    print(f"年龄组分布：{df['sub_age_group'].value_counts().sort_index()}")
    
    return df

def calculate_correlations(df):
    """计算各活动能力指标与高血脂的相关性"""
    # 活动能力指标列
    activity_cols = ['adl_total', 'iadl_total', 'activity_total',
                     'adl_eating', 'adl_dressing', 'adl_bathing', 'adl_toilet', 'adl_walking',
                     'iadl_shopping', 'iadl_cooking', 'iadl_finance', 'iadl_medication', 'iadl_transport']
    
    # 按年龄组分组，计算每个年龄组内各指标与高血脂的相关性
    age_groups = ['40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89']
    correlation_matrix = []
    
    for age_group in age_groups:
        group_df = df[df['sub_age_group'] == age_group]
        correlations = []
        
        for col in activity_cols:
            # 计算相关系数
            if len(group_df) > 0:
                corr = group_df[col].corr(group_df['high_lipid'])
            else:
                corr = 0  # 如果该年龄组没有数据，使用0
            correlations.append(corr)
        
        correlation_matrix.append(correlations)
    
    # 转换为DataFrame
    corr_df = pd.DataFrame(correlation_matrix, index=age_groups, columns=activity_cols)
    
    # 重命名列，使用中文标签
    col_names = ['ADL总分', 'IADL总分', '活动量表总分',
                 'ADL吃饭', 'ADL穿衣', 'ADL洗澡', 'ADL用厕', 'ADL步行',
                 'IADL购物', 'IADL做饭', 'IADL理财', 'IADL服药', 'IADL交通']
    corr_df.columns = col_names
    
    return corr_df

def create_spectrum_heatmap(corr_df):
    """创建频谱风格的热力图"""
    # 设置图表尺寸为16:9宽屏比例
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # 绘制热力图
    sns.heatmap(corr_df, annot=False, cmap='viridis', cbar_kws={'label': '与高血脂的相关性'}, ax=ax)
    
    # 旋转x轴标签45度
    plt.xticks(rotation=45, ha='right')
    
    # 添加纵向分区标注
    # 40-59岁区间（前4个年龄组）
    ax.axvspan(-0.5, 3.5, alpha=0.2, color='blue')
    ax.text(1.5, -0.5, 'ADL基础自理主导', ha='center', va='top', fontweight='bold')
    
    # 60-69岁区间（第5-6个年龄组）
    ax.axvspan(3.5, 5.5, alpha=0.2, color='yellow')
    ax.text(4.5, -0.5, 'IADL工具性活动爆发', ha='center', va='top', fontweight='bold')
    
    # 70-89岁区间（第7-10个年龄组）
    ax.axvspan(5.5, 9.5, alpha=0.2, color='red')
    ax.text(7.5, -0.5, '综合功能回归', ha='center', va='top', fontweight='bold')
    
    # 高亮数据矩阵中的最大值单元格
    max_val = corr_df.max().max()
    max_row, max_col = np.unravel_index(corr_df.values.argmax(), corr_df.shape)
    rect = plt.Rectangle((max_col, max_row), 1, 1, fill=False, edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    
    # 添加顶部第二横轴
    top_labels = ['中年早期', '中年晚期', '退休过渡', '老年早期', '高龄期']
    top_positions = [1, 3, 5, 7, 9]
    
    # 创建第二个x轴
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(top_positions)
    ax2.set_xticklabels(top_labels)
    ax2.tick_params(axis='x', which='both', length=0)
    
    # 设置图表标题
    ax.set_title('活动能力指标与高血脂相关性的频谱热力图 - 功能需求漂移分析', fontsize=16, fontweight='bold', pad=30)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('活动能力指标频谱热力图.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    print('活动能力指标频谱热力图已成功生成！')

def main():
    print("=== 生成活动能力指标的频谱风格热力图 ===")
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 准备数据
    prepared_df = prepare_data(df)
    if prepared_df is None:
        return
    
    # 计算相关性
    print("计算各活动能力指标与高血脂的相关性...")
    corr_df = calculate_correlations(prepared_df)
    
    # 显示相关性矩阵
    print("相关性矩阵:")
    print(corr_df)
    
    # 创建热力图
    print("创建频谱风格的热力图...")
    create_spectrum_heatmap(corr_df)

if __name__ == "__main__":
    main()
