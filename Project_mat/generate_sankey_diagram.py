#!/usr/bin/env python3
# 生成中医规则修正桑基图（决策流向图）

import pandas as pd
import numpy as np
try:
    import plotly.graph_objects as go
    from plotly.offline import plot
    print("成功导入Plotly!")
except ImportError:
    print("Plotly未安装，正在安装...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'plotly'])
    import plotly.graph_objects as go
    from plotly.offline import plot

# 设置中文字体（Plotly默认支持Unicode）
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'Heiti TC', 'SimHei', 'sans-serif']

def prepare_sankey_data(df_result):
    """
    准备桑基图数据
    
    Args:
        df_result: 包含预测结果的数据框
        
    Returns:
        (nodes, links): 节点和连接数据
    """
    
    # =============== 第一层：概率区间 ===============
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
    
    # =============== 第二层：中医判定 ===============
    # 对不确定区间的样本进行中医判定
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
    
    # =============== 第三层：最终等级 ===============
    # 这里的最终等级来自原始数据
    # 为了一致性，我们根据tcm_decision来映射
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
    
    # =============== 统计各流向的样本量 ===============
    # 第一层 -> 第二层
    flow1_2 = df_normal_lipid.groupby(['prob_interval', 'tcm_decision']).size().reset_index(name='count')
    
    # 第二层 -> 第三层
    flow2_3 = df_normal_lipid.groupby(['tcm_decision', 'final_level']).size().reset_index(name='count')
    
    # =============== 定义节点 ===============
    node_labels = [
        # 第一层：概率区间
        'p_hat<0.20', '0.20≤p_hat≤0.60', 'p_hat>0.60',
        # 第二层：中医判定
        '低风险(中医支持)', '中风险', '高风险(中医预警)', 'p_hat<0.20*', 'p_hat>0.60*',
        # 第三层：最终等级
        '低风险', '中风险', '高风险'
    ]
    
    # 节点颜色
    node_colors = [
        '#B3F0D8', '#FFD4A6', '#FFA8AF',  # 概率区间
        '#FFF2B3', '#FFD4A6', '#FFF2B3', '#B3F0D8', '#FFA8AF',  # 中医判定
        '#B3F0D8', '#FFD4A6', '#FFA8AF'   # 最终等级
    ]
    
    nodes = {label: i for i, label in enumerate(node_labels)}
    
    # =============== 定义连接 ===============
    links = []
    
    # 第一层 -> 第二层
    for _, row in flow1_2.iterrows():
        source = nodes[row['prob_interval']]
        # 特殊处理，因为tcm_decision中有些和第一层名称相同，但我们在第二层用了不同的标识
        if row['tcm_decision'] == 'p_hat<0.20':
            target = nodes['p_hat<0.20*']
        elif row['tcm_decision'] == 'p_hat>0.60':
            target = nodes['p_hat>0.60*']
        else:
            target = nodes[row['tcm_decision']]
        
        # 根据流向选择颜色
        if '低风险' in row['tcm_decision'] or row['tcm_decision'] == 'p_hat<0.20':
            color = 'rgba(179, 240, 216, 0.8)'
        elif '高风险' in row['tcm_decision'] or row['tcm_decision'] == 'p_hat>0.60':
            color = 'rgba(255, 168, 175, 0.8)'
        else:
            color = 'rgba(255, 212, 166, 0.8)'
            
        links.append({
            'source': source,
            'target': target,
            'value': row['count'],
            'color': color
        })
    
    # 第二层 -> 第三层
    for _, row in flow2_3.iterrows():
        # 映射源节点
        if row['tcm_decision'] == 'p_hat<0.20':
            source = nodes['p_hat<0.20*']
        elif row['tcm_decision'] == 'p_hat>0.60':
            source = nodes['p_hat>0.60*']
        else:
            source = nodes[row['tcm_decision']]
        
        target = nodes[row['final_level']]
        
        # 根据流向选择颜色
        if row['final_level'] == '低风险':
            color = 'rgba(179, 240, 216, 0.8)'
        elif row['final_level'] == '高风险':
            color = 'rgba(255, 168, 175, 0.8)'
        else:
            color = 'rgba(255, 212, 166, 0.8)'
            
        links.append({
            'source': source,
            'target': target,
            'value': row['count'],
            'color': color
        })
    
    return node_labels, node_colors, links, df_normal_lipid

def create_sankey_diagram(node_labels, node_colors, links):
    """
    创建桑基图
    
    Args:
        node_labels: 节点标签列表
        node_colors: 节点颜色列表
        links: 连接列表
        
    Returns:
        fig: Plotly图表对象
    """
    
    # 提取节点和连接数据
    source = [link['source'] for link in links]
    target = [link['target'] for link in links]
    value = [link['value'] for link in links]
    link_colors = [link['color'] for link in links]
    
    # 创建桑基图
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
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text='图2：中医规则修正桑基图（决策流向图）',
            font=dict(size=20, color='#1565C0'),
            x=0.5
        ),
        font=dict(size=12, family='SimHei, Heiti TC, sans-serif'),
        width=1200,
        height=800,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # 添加列标题注释
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
    print("生成中医规则修正桑基图")
    print("=" * 80)
    
    # 1. 读取结果数据
    data_path = '/workspace/Project_mat/data/processed/three_layer_result.pkl'
    print(f"\n读取数据: {data_path}")
    df_result = pd.read_pickle(data_path)
    print(f"数据加载完成，总样本量: {len(df_result)}")
    
    # 2. 准备桑基图数据
    print("\n准备桑基图数据...")
    node_labels, node_colors, links, df_normal_lipid = prepare_sankey_data(df_result)
    print(f"血脂正常样本量: {len(df_normal_lipid)}")
    
    # 3. 创建桑基图
    print("\n创建桑基图...")
    fig = create_sankey_diagram(node_labels, node_colors, links)
    
    # 4. 保存图表
    html_path = '/workspace/Project_mat/图2_中医规则修正桑基图.html'
    png_path = '/workspace/Project_mat/图2_中医规则修正桑基图.png'
    
    print(f"\n保存交互式HTML: {html_path}")
    fig.write_html(html_path)
    
    print(f"保存静态PNG: {png_path}")
    try:
        fig.write_image(png_path, width=1200, height=800, scale=2)
        print("PNG保存成功!")
    except Exception as e:
        print(f"PNG保存需要kaleido库: {e}")
        print("请安装: pip install -U kaleido")
    
    # 5. 显示统计信息
    print("\n" + "=" * 80)
    print("数据统计信息")
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
        
        # 统计升档和降档
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
    if 'png_path' in locals():
        print(f"静态图片: {png_path}")
    
    return fig

if __name__ == '__main__':
    try:
        fig = main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
