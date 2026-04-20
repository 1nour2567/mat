#!/usr/bin/env python3
# 生成问题一的学术风格流程图

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

def create_problem1_flowchart():
    """创建问题一流程图"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')
    
    # 标题
    ax.text(50, 57, '问题一：关键指标分析流程图',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color='#1565C0')
    
    # 子标题
    ax.text(50, 54, '图1 · 综合评分方法 · 性别年龄分组 · 关键指标识别',
            ha='center', va='center', fontsize=11, color='#546E7A')
    
    # ==================== 输入层 ====================
    input_y = 47
    input_height = 6
    
    # 原始数据
    draw_rectangle(ax, 10, input_y, 20, input_height,
                  '原始数据\n（中医体质、活动量表、血常规）',
                  facecolor='#E8F5E9', edgecolor='#2E7D32',
                  text_color='#1B5E20')
    
    # 数据预处理
    draw_rectangle(ax, 35, input_y, 20, input_height,
                  '数据预处理\n（清洗、标准化、特征衍生）',
                  facecolor='#FFF3E0', edgecolor='#EF6C00',
                  text_color='#E65100')
    
    # 特征选择
    draw_rectangle(ax, 60, input_y, 25, input_height,
                  '特征选择\n血常规指标：7项\n派生指标：5项',
                  facecolor='#F3E5F5', edgecolor='#7B1FA2',
                  text_color='#4A148C')
    
    # ==================== 分析方法层 ====================
    method_y = 37
    method_height = 8
    
    # 虚线框
    method_box = FancyBboxPatch((15, method_y), 70, method_height,
                          boxstyle='round,pad=0.2',
                          facecolor='#F5F5F5', edgecolor='#9E9E9E',
                          linewidth=1.5, linestyle='--', alpha=0.8)
    ax.add_patch(method_box)
    
    ax.text(50, method_y + method_height - 1, '三种特征重要性评估技术',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='#616161')
    
    # 三个分析方法
    draw_rectangle(ax, 20, method_y + 1, 20, 4,
                  'Spearman相关系数\n（痰湿表征能力）',
                  facecolor='#E3F2FD', edgecolor='#1976D2',
                  text_color='#0D47A1', fontweight='normal')
    
    draw_rectangle(ax, 45, method_y + 1, 20, 4,
                  '互信息\n（风险预警能力）',
                  facecolor='#FFF3E0', edgecolor='#F57C00',
                  text_color='#BF360C', fontweight='normal')
    
    draw_rectangle(ax, 70, method_y + 1, 20, 4,
                  'PLS联合结构载荷\n（双目标整合能力）',
                  facecolor='#E8F5E9', edgecolor='#388E3C',
                  text_color='#1B5E20', fontweight='normal')
    
    # ==================== 综合评分层 ====================
    score_y = 27
    score_height = 6
    
    score_box = draw_rectangle(ax, 30, score_y, 40, score_height,
                          '熵权法综合评分\n权重：\nSpearman: 0.1507\n互信息: 0.7091\nPLS: 0.1401',
                          facecolor='#E3F2FD', edgecolor='#1976D2',
                          text_color='#0D47A1')
    
    # ==================== 分组分析层 ====================
    group_y = 17
    group_height = 6
    
    # 性别分析
    draw_rectangle(ax, 15, group_y, 25, group_height,
                  '性别分组分析\n男性：TC、AIP、TG\n女性：TC、AIP、TG',
                  facecolor='#FFF3E0', edgecolor='#F57C00',
                  text_color='#BF360C')
    
    # 年龄组分析
    draw_rectangle(ax, 45, group_y, 25, group_height,
                  '年龄组分组分析\n40-89岁各年龄组\n关键指标差异分析',
                  facecolor='#E8F5E9', edgecolor='#388E3C',
                  text_color='#1B5E20')
    
    # 体质分析
    draw_rectangle(ax, 75, group_y, 15, group_height,
                  '体质贡献度分析',
                  facecolor='#F3E5F5', edgecolor='#7B1FA2',
                  text_color='#4A148C')
    
    # ==================== 输出层 ====================
    output_y = 3
    output_height = 5
    
    # 关键指标输出
    output_box = draw_rectangle(ax, 15, output_y, 30, output_height,
                            '关键指标输出\n第1名：TC（0.1588）\n第2名：TG（0.1484）\n第3名：血尿酸（0.1305）',
                            facecolor='#E3F2FD', edgecolor='#1976D2',
                            text_color='#0D47A1', boxstyle='round,pad=0.2')
    
    # 结论输出
    conclusion_box = draw_rectangle(ax, 50, output_y, 35, output_height,
                               '分析结论\n血常规指标表现更优\n活动量表有预测价值\n性别年龄组差异显著',
                               facecolor='#E8F5E9', edgecolor='#388E3C',
                               text_color='#1B5E20', boxstyle='round,pad=0.2')
    
    # ==================== 箭头连接 ====================
    
    # 输入层 → 分析方法层
    draw_arrow(ax, 20, input_y, 20, method_y + method_height, color='#2E7D32')
    draw_arrow(ax, 45, input_y, 45, method_y + method_height, color='#EF6C00')
    draw_arrow(ax, 72, input_y, 72, method_y + method_height, color='#7B1FA2')
    
    # 分析方法层 → 综合评分层
    draw_arrow(ax, 30, method_y, 50, score_y + score_height, color='#1976D2')
    draw_arrow(ax, 55, method_y, 50, score_y + score_height, color='#F57C00')
    draw_arrow(ax, 80, method_y, 50, score_y + score_height, color='#388E3C')
    
    # 综合评分层 → 分组分析层
    draw_arrow(ax, 50, score_y, 27, group_y + group_height, color='#1976D2')
    draw_arrow(ax, 50, score_y, 57, group_y + group_height, color='#388E3C')
    draw_arrow(ax, 50, score_y, 82, group_y + group_height, color='#7B1FA2')
    
    # 分组分析层 → 输出层
    draw_arrow(ax, 27, group_y, 30, output_y + output_height, color='#F57C00')
    draw_arrow(ax, 57, group_y, 67, output_y + output_height, color='#388E3C')
    draw_arrow(ax, 82, group_y, 67, output_y + output_height, color='#7B1FA2')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = '/workspace/Project_mat/problem1_flowchart.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"问题一流程图已保存到: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    create_problem1_flowchart()
