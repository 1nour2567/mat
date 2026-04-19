#!/usr/bin/env python3
# 生成中医规则修正桑基图（包含模拟数据演示完整效果）

import pandas as pd
import numpy as np
import plotly.graph_objects as go

def create_simulation_data():
    """
    创建包含模拟数据的数据集，演示完整的中医修正效果
    
    Returns:
        df_result: 包含模拟数据的结果数据框
    """
    
    # 读取真实数据
    real_data_path = '/workspace/Project_mat/data/processed/three_layer_result.pkl'
    df_real = pd.read_pickle(real_data_path)
    
    # 复制真实数据
    df_sim = df_real.copy()
    
    # 获取血脂正常的样本
    normal_lipid_mask = df_sim['血脂异常项数'] == 0
    df_normal_lipid = df_sim[normal_lipid_mask].copy()
    
    # 对于血脂正常的样本，我们添加一些模拟数据来演示中医修正
    np.random.seed(42)
    
    # 在不确定区间（0.20-0.60）添加更多样本
    num_uncertain = 50
    
    # 1. 添加升档样本（痰湿质≥60且活动<40）
    num_upgrade = 10
    upgrade_indices = np.random.choice(df_normal_lipid.index, num_upgrade, replace=False)
    df_sim.loc[upgrade_indices, '痰湿质'] = np.random.randint(60, 100, num_upgrade)
    df_sim.loc[upgrade_indices, '活动量表总分（ADL总分+IADL总分）'] = np.random.randint(10, 40, num_upgrade)
    df_sim.loc[upgrade_indices, '模型预测概率'] = np.random.uniform(0.20, 0.60, num_upgrade)
    df_sim.loc[upgrade_indices, '最终风险等级'] = '高风险(中医预警)'
    
    # 2. 添加降档样本（痰湿质<17且活动≥60）
    num_downgrade = 10
    remaining_indices = list(set(df_normal_lipid.index) - set(upgrade_indices))
    downgrade_indices = np.random.choice(remaining_indices, num_downgrade, replace=False)
    df_sim.loc[downgrade_indices, '痰湿质'] = np.random.randint(5, 17, num_downgrade)
    df_sim.loc[downgrade_indices, '活动量表总分（ADL总分+IADL总分）'] = np.random.randint(60, 100, num_downgrade)
    df_sim.loc[downgrade_indices, '模型预测概率'] = np.random.uniform(0.20, 0.60, num_downgrade)
    df_sim.loc[downgrade_indices, '最终风险等级'] = '低风险(中医支持)'
    
    # 3. 添加保持中风险的样本
    num_remain = 30
    remaining_indices2 = list(set(remaining_indices) - set(downgrade_indices))
    remain_indices = np.random.choice(remaining_indices2, min(num_remain, len(remaining_indices2)), replace=False)
    df_sim.loc[remain_indices, '模型预测概率'] = np.random.uniform(0.20, 0.60, len(remain_indices))
    
    print(f"已创建模拟数据:")
    print(f"  - 升档样本: {num_upgrade}")
    print(f"  - 降档样本: {num_downgrade}")
    print(f"  - 保持中风险: {len(remain_indices)}")
    
    return df_sim

def prepare_sankey_data(df_result):
    """
    准备桑基图数据
    
    Args:
        df_result: 包含预测结果的数据框
        
    Returns:
        (node_labels, node_colors, links, df_normal_lipid): 节点、颜色、连接、处理后的数据
    """
    
    # 首先排除血脂异常≥1的样本，因为它们直接被标记为临床确诊高风险
    df_normal_lipid = df_result[df_result['血脂异常项数'] == 0].copy()
    
    # 划分概率区间
    conditions = [
        df_normal_lipid['模型预测概率'] < 0.20,
        (df_normal_lipid['模型预测概率'] >= 0.20) & (df_normal_lipid['模型预测概率'] <= 0.60),
        df_normal_lipid['模型预测概率'] > 0.60
    ]
    choices = ['p_hat<0.20', '0.20≤p_hat≤0.60', 'p_hat>0.60']
    df_normal_lipid['prob_interval'] = np.select(conditions, choices, default='other')
    
    # 中医判定逻辑
    def determine_tcm_decision(row):
        if row['prob_interval'] == '0.20≤p_hat≤0.60':
            # 检查升档条件
            if row['痰湿质'] >= 60 and row['活动量表总分（ADL总分+IADL总分）'] < 40:
                return '高风险(中医预警)'
            # 检查降档条件
            elif row['痰湿质'] < 17 and row['活动量表总分（ADL总分+IADL总分）'] >= 60:
                return '低风险(中医支持)'
            else:
                return '中风险'
        else:
            return row['prob_interval']
    
    df_normal_lipid['tcm_decision'] = df_normal_lipid.apply(determine_tcm_decision, axis=1)
    
    # 映射到最终等级
    def get_final_level(row):
        if row['tcm_decision'] == '高风险(中医预警)':
            return '高风险'
        elif row['tcm_decision'] == '低风险(中医支持)':
            return '低风险'
        elif row['tcm_decision'] == 'p_hat<0.20':
            return '低风险'
        elif row['tcm_decision'] == 'p_hat>0.60':
            return '高风险'
        else:
            return '中风险'
    
    df_normal_lipid['final_level'] = df_normal_lipid.apply(get_final_level, axis=1)
    
    # 统计各流向
    flow1_2 = df_normal_lipid.groupby(['prob_interval', 'tcm_decision']).size().reset_index(name='count')
    flow2_3 = df_normal_lipid.groupby(['tcm_decision', 'final_level']).size().reset_index(name='count')
    
    # 定义节点
    node_labels = [
        'p_hat<0.20', '0.20≤p_hat≤0.60', 'p_hat>0.60',
        '低风险(中医支持)', '中风险', '高风险(中医预警)', 'p_hat<0.20*', 'p_hat>0.60*',
        '低风险', '中风险', '高风险'
    ]
    
    node_colors = [
        '#B3F0D8', '#FFD4A6', '#FFA8AF',
        '#FFF2B3', '#FFD4A6', '#FFF2B3', '#B3F0D8', '#FFA8AF',
        '#B3F0D8', '#FFD4A6', '#FFA8AF'
    ]
    
    nodes = {label: i for i, label in enumerate(node_labels)}
    
    # 定义连接
    links = []
    
    # 第一层 -> 第二层
    for _, row in flow1_2.iterrows():
        source = nodes[row['prob_interval']]
        if row['tcm_decision'] == 'p_hat<0.20':
            target = nodes['p_hat<0.20*']
        elif row['tcm_decision'] == 'p_hat>0.60':
            target = nodes['p_hat>0.60*']
        else:
            target = nodes[row['tcm_decision']]
        
        if '低风险' in row['tcm_decision'] or row['tcm_decision'] == 'p_hat<0.20':
            color = 'rgba(179, 240, 216, 0.8)'
        elif '高风险' in row['tcm_decision'] or row['tcm_decision'] == 'p_hat>0.60':
            color = 'rgba(255, 168, 175, 0.8)'
        else:
            color = 'rgba(255, 212, 166, 0.8)'
            
        links.append({'source': source, 'target': target, 'value': row['count'], 'color': color})
    
    # 第二层 -> 第三层
    for _, row in flow2_3.iterrows():
        if row['tcm_decision'] == 'p_hat<0.20':
            source = nodes['p_hat<0.20*']
        elif row['tcm_decision'] == 'p_hat>0.60':
            source = nodes['p_hat>0.60*']
        else:
            source = nodes[row['tcm_decision']]
        
        target = nodes[row['final_level']]
        
        if row['final_level'] == '低风险':
            color = 'rgba(179, 240, 216, 0.8)'
        elif row['final_level'] == '高风险':
            color = 'rgba(255, 168, 175, 0.8)'
        else:
            color = 'rgba(255, 212, 166, 0.8)'
            
        links.append({'source': source, 'target': target, 'value': row['count'], 'color': color})
    
    return node_labels, node_colors, links, df_normal_lipid

def create_sankey_diagram(node_labels, node_colors, links):
    """创建桑基图"""
    
    source = [link['source'] for link in links]
    target = [link['target'] for link in links]
    value = [link['value'] for link in links]
    link_colors = [link['color'] for link in links]
    
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=30,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=node_labels,
            color=node_colors,
            hovertemplate='节点: %{label}<br>样本量: %{value}<extra></extra>'
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
            hovertemplate='从: %{source.label}<br>到: %{target.label}<br>样本量: %{value}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title=dict(
            text='图2：中医规则修正桑基图（决策流向图）- 含模拟数据',
            font=dict(size=20, color='#1565C0'),
            x=0.5
        ),
        font=dict(size=12, family='SimHei, Heiti TC, sans-serif'),
        width=1200,
        height=800,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    annotations = [
        dict(x=0.08, y=1.05, text='<b>概率区间</b>', showarrow=False, xanchor='center', font=dict(size=14)),
        dict(x=0.5, y=1.05, text='<b>中医判定</b>', showarrow=False, xanchor='center', font=dict(size=14)),
        dict(x=0.92, y=1.05, text='<b>最终等级</b>', showarrow=False, xanchor='center', font=dict(size=14))
    ]
    fig.update_layout(annotations=annotations)
    
    return fig

def main():
    """主函数"""
    print("=" * 80)
    print("生成中医规则修正桑基图（含模拟数据演示完整效果）")
    print("=" * 80)
    
    # 创建模拟数据
    print("\n创建含模拟数据的数据集...")
    df_result = create_simulation_data()
    
    # 准备桑基图数据
    print("\n准备桑基图数据...")
    node_labels, node_colors, links, df_normal_lipid = prepare_sankey_data(df_result)
    print(f"血脂正常样本量: {len(df_normal_lipid)}")
    
    # 创建桑基图
    print("\n创建桑基图...")
    fig = create_sankey_diagram(node_labels, node_colors, links)
    
    # 保存图表
    html_path = '/workspace/Project_mat/图2_中医规则修正桑基图_含模拟数据.html'
    print(f"\n保存交互式HTML: {html_path}")
    fig.write_html(html_path)
    
    # 显示统计信息
    print("\n" + "=" * 80)
    print("数据统计信息（含模拟数据）")
    print("=" * 80)
    
    print("\n概率区间分布:")
    prob_dist = df_normal_lipid['prob_interval'].value_counts().sort_index()
    for level, count in prob_dist.items():
        print(f"  {level}: {count} ({count/len(df_normal_lipid)*100:.1f}%)")
    
    print("\n中医修正情况:")
    uncertain_df = df_normal_lipid[df_normal_lipid['prob_interval'] == '0.20≤p_hat≤0.60']
    if len(uncertain_df) > 0:
        tcm_dist = uncertain_df['tcm_decision'].value_counts()
        for decision, count in tcm_dist.items():
            print(f"  {decision}: {count} ({count/len(uncertain_df)*100:.1f}%)")
        
        upgrade = len(uncertain_df[uncertain_df['tcm_decision'] == '高风险(中医预警)'])
        downgrade = len(uncertain_df[uncertain_df['tcm_decision'] == '低风险(中医支持)'])
        print(f"\n升档样本: {upgrade}")
        print(f"降档样本: {downgrade}")
        print(f"保持中风险: {len(uncertain_df) - upgrade - downgrade}")
    
    print("\n最终风险等级分布:")
    final_dist = df_normal_lipid['final_level'].value_counts()
    for level, count in final_dist.items():
        print(f"  {level}: {count} ({count/len(df_normal_lipid)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("桑基图生成完成!")
    print("=" * 80)
    print(f"交互式图表: {html_path}")
    
    return fig

if __name__ == '__main__':
    try:
        fig = main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
