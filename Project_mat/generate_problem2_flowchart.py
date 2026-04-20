#!/usr/bin/env python3
# 生成问题二的学术风格流程图

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# 设置中文字体和样式
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150

def draw_rectangle(ax, x, y, width, height, text, facecolor='#E3F2FD', edgecolor='#1976D2', 
                   linewidth=2, text_color='#0D47A1', boxstyle='round,pad=0.1', fontweight='bold'):
    """绘制带文字的圆角矩形"""
    box = FancyBboxPatch((x, y), width, height, boxstyle=boxstyle,
                        facecolor=facecolor, edgecolor=edgecolor,
                        linewidth=linewidth, alpha=0.9)
    ax.add_patch(box)
    
    # 添加文字
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', color=text_color,
            fontsize=10, fontweight=fontweight)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='#555555', linewidth=2, arrowstyle='->'):
    """绘制箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=arrowstyle, linewidth=linewidth,
                          color=color, mutation_scale=20)
    ax.add_patch(arrow)
    return arrow

def create_problem2_flowchart():
    """创建问题二流程图"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')
    
    # 标题
    ax.text(50, 57, '问题二：高血脂风险预警模型流程图',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color='#1565C0')
    
    # 子标题
    ax.text(50, 54, '图5 · 四层融合架构 · 中西医结合 · 特征隔离 · 核心特征识别',
            ha='center', va='center', fontsize=11, color='#546E7A')
    
    # ==================== 输入层 ====================
    input_y = 47
    input_height = 6
    
    # 中医体质积分
    draw_rectangle(ax, 10, input_y, 20, input_height,
                  '中医体质积分\n（9维）',
                  facecolor='#E8F5E9', edgecolor='#2E7D32',
                  text_color='#1B5E20')
    
    # 活动能力量表
    draw_rectangle(ax, 35, input_y, 20, input_height,
                  '活动能力量表\n（3维）',
                  facecolor='#FFF3E0', edgecolor='#EF6C00',
                  text_color='#E65100')
    
    # 代谢与人口学
    draw_rectangle(ax, 60, input_y, 20, input_height,
                  '代谢与人口学\n（8维）',
                  facecolor='#F3E5F5', edgecolor='#7B1FA2',
                  text_color='#4A148C')
    
    # 血脂指标（单独）
    draw_rectangle(ax, 85, input_y, 10, input_height,
                  '血脂指标\n（4维）',
                  facecolor='#FFEBEE', edgecolor='#D32F2F',
                  text_color='#B71C1C')
    
    # ==================== 特征工程层 ====================
    fe_y = 37
    fe_height = 8
    
    # 虚线框
    fe_box = FancyBboxPatch((15, fe_y), 70, fe_height,
                          boxstyle='round,pad=0.2',
                          facecolor='#F5F5F5', edgecolor='#9E9E9E',
                          linewidth=1.5, linestyle='--', alpha=0.8)
    ax.add_patch(fe_box)
    
    ax.text(50, fe_y + fe_height - 1, '中西医交叉特征生成',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='#616161')
    
    # 特征列表
    features = ['痰湿质×BMI', '痰湿质×活动量表', '痰湿质×血尿酸', 
                '气虚质×BMI', '气虚质×活动量表']
    feature_text = ' · '.join(features)
    ax.text(50, fe_y + 2, feature_text,
            ha='center', va='center', fontsize=9, color='#424242')
    
    # ==================== 隔离墙 ====================
    wall_y = 31
    wall_height = 4
    
    # 红色盾牌/虚线效果
    ax.plot([10, 90], [wall_y + wall_height/2, wall_y + wall_height/2],
            color='#D32F2F', linewidth=4, linestyle='--', alpha=0.8)
    
    # 文字
    ax.text(50, wall_y + wall_height/2, 
            '严格特征隔离墙（19项血脂特征严禁进入训练）',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='#C62828', bbox=dict(facecolor='#FFEBEE', edgecolor='#FFCDD2', pad=3))
    
    # ==================== 第一层：临床规则层 ====================
    layer1_y = 23
    layer_height = 6
    
    layer1 = draw_rectangle(ax, 15, layer1_y, 25, layer_height,
                          '临床规则层\n计算血脂异常项数\nN_i ≥1 → 临床确诊高风险',
                          facecolor='#FFF3E0', edgecolor='#F57C00',
                          text_color='#BF360C')
    
    # ==================== 第二层：LightGBM预测层 ====================
    layer2 = draw_rectangle(ax, 45, layer1_y, 25, layer_height,
                          'LightGBM预测层\n5折交叉验证\n25项安全特征\n输出p_hat ∈ [0,1]',
                          facecolor='#E3F2FD', edgecolor='#1976D2',
                          text_color='#0D47A1')
    
    # ==================== 第三层：中医功能层 ====================
    layer3_y = 13
    layer3 = draw_rectangle(ax, 30, layer3_y, 40, layer_height,
                          '中医功能层\np_hat ∈ [0.20, 0.60] 触发修正\n痰湿质≥60且活动<40 → 升档\n痰湿质<17且活动≥60 → 降档',
                          facecolor='#E8F5E9', edgecolor='#388E3C',
                          text_color='#1B5E20')
    
    # ==================== 输出层 ====================
    output_y = 3
    output_height = 5
    
    # 低风险（绿色）
    low_risk = draw_rectangle(ax, 10, output_y, 18, output_height,
                            '低风险',
                            facecolor='#C8E6C9', edgecolor='#2E7D32',
                            text_color='#1B5E20', boxstyle='round,pad=0.3')
    
    # 中风险（黄色）
    mid_risk = draw_rectangle(ax, 35, output_y, 18, output_height,
                            '中风险',
                            facecolor='#FFF9C4', edgecolor='#FBC02D',
                            text_color='#F57F17', boxstyle='round,pad=0.3')
    
    # 高风险（红色）
    high_risk = draw_rectangle(ax, 60, output_y, 18, output_height,
                             '高风险',
                             facecolor='#FFCDD2', edgecolor='#D32F2F',
                             text_color='#B71C1C', boxstyle='round,pad=0.3')
    
    # 核心特征识别（蓝色）
    combo_risk = draw_rectangle(ax, 82, output_y, 14, output_height,
                                '核心特征\n组合识别',
                                facecolor='#E3F2FD', edgecolor='#1976D2',
                                text_color='#0D47A1', boxstyle='round,pad=0.3')
    
    # ==================== 箭头连接 ====================
    
    # 输入层 → 特征工程层
    draw_arrow(ax, 20, input_y, 20, fe_y + fe_height, color='#2E7D32')
    draw_arrow(ax, 45, input_y, 45, fe_y + fe_height, color='#EF6C00')
    draw_arrow(ax, 70, input_y, 70, fe_y + fe_height, color='#7B1FA2')
    
    # 血脂指标 → 临床规则层（绕过隔离墙）
    draw_arrow(ax, 90, input_y, 27, layer1_y + layer_height, color='#D32F2F')
    
    # 特征工程层 → 各层
    draw_arrow(ax, 27, fe_y, 27, layer1_y + layer_height, color='#F57C00')
    draw_arrow(ax, 57, fe_y, 57, layer1_y + layer_height, color='#1976D2')
    
    # 第一层 → 第二层
    draw_arrow(ax, 40, layer1_y + layer_height/2, 45, layer1_y + layer_height/2)
    
    # 第一层、第二层 → 第三层
    draw_arrow(ax, 27, layer1_y, 50, layer3_y + layer_height)
    draw_arrow(ax, 57, layer1_y, 50, layer3_y + layer_height)
    
    # 第三层 → 输出层
    draw_arrow(ax, 35, layer3_y, 19, output_y + output_height, color='#2E7D32')
    draw_arrow(ax, 50, layer3_y, 44, output_y + output_height, color='#FBC02D')
    draw_arrow(ax, 65, layer3_y, 69, output_y + output_height, color='#D32F2F')
    
    # 第一层直接 → 高风险（绕过第二、三层）
    arrow1 = FancyArrowPatch((27, layer1_y + layer_height/2), (69, output_y + output_height),
                            arrowstyle='->', linewidth=2.5, color='#D32F2F',
                            connectionstyle="arc3,rad=-0.3", mutation_scale=25)
    ax.add_patch(arrow1)
    
    # 第三层 → 核心特征识别
    draw_arrow(ax, 70, layer3_y, 89, output_y + output_height, color='#1976D2')
    
    # 添加文字说明
    ax.text(55, 18, '直接确诊（绕过模型）',
            ha='center', va='center', fontsize=9, color='#C62828',
            bbox=dict(facecolor='#FFEBEE', edgecolor='none', pad=2))
    
    plt.tight_layout()
    
    # 保存图片
    output_path = '/workspace/Project_mat/problem2_flowchart.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"问题二流程图已保存到: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    create_problem2_flowchart()
