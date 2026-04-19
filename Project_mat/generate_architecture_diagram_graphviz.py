#!/usr/bin/env python3
# 使用Graphviz生成更美观的三层融合预警模型架构流程图

try:
    from graphviz import Digraph
    print("成功导入Graphviz!")
except ImportError:
    print("Graphviz未安装，尝试使用matplotlib版本")
    exit()

def create_architecture_diagram_graphviz():
    """使用Graphviz创建架构流程图"""
    
    # 创建有向图
    dot = Digraph('三层融合预警模型', format='png')
    dot.attr(rankdir='TB', size='16,10', dpi='200')
    dot.attr('node', fontname='SimHei', fontsize='11', shape='box', style='rounded,filled')
    dot.attr('edge', fontname='SimHei', fontsize='10')
    
    # 设置颜色主题
    colors = {
        'input1': ('#E8F5E9', '#2E7D32', '#1B5E20'),  # 中医体质
        'input2': ('#FFF3E0', '#EF6C00', '#E65100'),  # 活动能力
        'input3': ('#F3E5F5', '#7B1FA2', '#4A148C'),  # 代谢与人口学
        'feature': ('#F5F5F5', '#9E9E9E', '#616161'),  # 特征工程
        'layer1': ('#FFF3E0', '#F57C00', '#BF360C'),   # 临床规则层
        'layer2': ('#E3F2FD', '#1976D2', '#0D47A1'),   # LightGBM预测层
        'layer3': ('#E8F5E9', '#388E3C', '#1B5E20'),   # 中医功能层
        'output_low': ('#C8E6C9', '#2E7D32', '#1B5E20'),  # 低风险
        'output_mid': ('#FFF9C4', '#FBC02D', '#F57F17'),   # 中风险
        'output_high': ('#FFCDD2', '#D32F2F', '#B71C1C')   # 高风险
    }
    
    # =============== 子图：输入层 ===============
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入层', fontname='SimHei', fontsize='14', fontweight='bold', style='dashed')
        c.attr('node', shape='box')
        
        c.node('input1', '中医体质积分\n（9维）', 
               fillcolor=colors['input1'][0], color=colors['input1'][1], fontcolor=colors['input1'][2])
        c.node('input2', '活动能力量表\n（3维）', 
               fillcolor=colors['input2'][0], color=colors['input2'][1], fontcolor=colors['input2'][2])
        c.node('input3', '代谢与人口学\n（8维）', 
               fillcolor=colors['input3'][0], color=colors['input3'][1], fontcolor=colors['input3'][2])
    
    # =============== 子图：特征工程层 ===============
    with dot.subgraph(name='cluster_feature') as c:
        c.attr(label='特征工程层', fontname='SimHei', fontsize='12', style='dashed')
        
        feature_text = '中西医交叉特征生成:\n'
        feature_text += '痰湿质×BMI · 痰湿质×活动量表 · 痰湿质×血尿酸 ·\n'
        feature_text += '气虚质×BMI · 气虚质×活动量表'
        
        c.node('feature', feature_text,
               fillcolor=colors['feature'][0], color=colors['feature'][1], fontcolor=colors['feature'][2])
    
    # =============== 隔离墙 ===============
    dot.node('wall', '<<table border="0" cellborder="1" cellspacing="0" cellpadding="5"><tr><td bgcolor="#FFEBEE"><font color="#C62828" point-size="12">严格特征隔离墙<br/>（19项血脂特征严禁进入训练）</font></td></tr></table>>',
             shape='plaintext')
    
    # =============== 子图：模型层 ===============
    with dot.subgraph(name='cluster_models') as c:
        c.attr(label='三层预测模型', fontname='SimHei', fontsize='14', fontweight='bold', style='dashed')
        
        c.node('layer1', '临床规则层\n计算血脂异常项数\nN_i ≥1 → 临床确诊高风险',
               fillcolor=colors['layer1'][0], color=colors['layer1'][1], fontcolor=colors['layer1'][2])
        c.node('layer2', 'LightGBM预测层\n5折交叉验证\n25项安全特征\n输出p_hat ∈ [0,1]',
               fillcolor=colors['layer2'][0], color=colors['layer2'][1], fontcolor=colors['layer2'][2])
        c.node('layer3', '中医功能层\np_hat ∈ [0.20, 0.60] 触发修正\n痰湿质≥60且活动<40 → 升档\n痰湿质<17且活动≥60 → 降档',
               fillcolor=colors['layer3'][0], color=colors['layer3'][1], fontcolor=colors['layer3'][2])
    
    # =============== 子图：输出层 ===============
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出层', fontname='SimHei', fontsize='14', fontweight='bold', style='dashed')
        
        c.node('output_low', '低风险',
               fillcolor=colors['output_low'][0], color=colors['output_low'][1], fontcolor=colors['output_low'][2])
        c.node('output_mid', '中风险',
               fillcolor=colors['output_mid'][0], color=colors['output_mid'][1], fontcolor=colors['output_mid'][2])
        c.node('output_high', '高风险',
               fillcolor=colors['output_high'][0], color=colors['output_high'][1], fontcolor=colors['output_high'][2])
    
    # =============== 添加边（连接） ===============
    # 输入层 → 特征工程层
    dot.edge('input1', 'feature', color=colors['input1'][1], penwidth='2')
    dot.edge('input2', 'feature', color=colors['input2'][1], penwidth='2')
    dot.edge('input3', 'feature', color=colors['input3'][1], penwidth='2')
    
    # 特征工程层 → 隔离墙 → 模型层
    dot.edge('feature', 'wall', style='dashed', color='#9E9E9E', penwidth='1.5')
    dot.edge('wall', 'layer1', style='dashed', color='#9E9E9E', penwidth='1.5')
    dot.edge('wall', 'layer2', style='dashed', color='#9E9E9E', penwidth='1.5')
    
    # 模型层之间
    dot.edge('layer1', 'layer2', penwidth='2')
    dot.edge('layer1', 'layer3', penwidth='2')
    dot.edge('layer2', 'layer3', penwidth='2')
    
    # 模型层 → 输出层
    dot.edge('layer3', 'output_low', color=colors['output_low'][1], penwidth='2')
    dot.edge('layer3', 'output_mid', color=colors['output_mid'][1], penwidth='2')
    dot.edge('layer3', 'output_high', color=colors['output_high'][1], penwidth='2')
    
    # 第一层直接 → 高风险（绕过其他层）
    dot.edge('layer1', 'output_high', color=colors['output_high'][1], penwidth='3', 
             label='直接确诊\n（绕过模型）', fontcolor='#C62828', fontsize='10')
    
    # 设置等级（Rank）
    dot.attr(rank='same', nodes='input1 input2 input3')
    dot.attr(rank='same', nodes='output_low output_mid output_high')
    
    # 添加标题
    dot.attr(label='<<font point-size="20" color="#1565C0"><b>三层融合预警模型架构流程图</b></font><br/>' +
              '<font point-size="12" color="#546E7A">中西医结合 · 特征隔离 · 分层预测</font>>',
              labelloc='t', fontname='SimHei')
    
    # 渲染并保存
    output_path = '/workspace/Project_mat/三层融合预警模型架构图_graphviz'
    dot.render(output_path, cleanup=True, view=False)
    print(f"Graphviz版本架构图已保存到: {output_path}.png")
    
    return output_path

if __name__ == '__main__':
    try:
        create_architecture_diagram_graphviz()
    except Exception as e:
        print(f"Graphviz版本生成失败: {e}")
        print("请使用之前的matplotlib版本，图片已保存在: /workspace/Project_mat/三层融合预警模型架构图.png")
