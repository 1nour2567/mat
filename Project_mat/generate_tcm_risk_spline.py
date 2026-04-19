#!/usr/bin/env python3
# 生成痰湿体质风险预警的“限制性立方样条+活动分层曲线”

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 模拟痰湿积分与OR值的关系
def generate_or_values():
    """生成模拟的OR值数据"""
    # 痰湿积分范围
    tan_shi_scores = np.linspace(0, 100, 100)
    
    # 低活动量组（活动量表总分<40）的OR值
    # 模拟OR值随痰湿积分增加而增加，尤其是在积分≥60后快速上升
    or_low_activity = np.exp(0.03 * (tan_shi_scores - 40))
    or_low_activity[tan_shi_scores < 40] = 1.0
    or_low_activity[tan_shi_scores >= 60] = np.exp(0.05 * (tan_shi_scores[tan_shi_scores >= 60] - 40))
    
    # 高活动量组（活动量表总分≥60）的OR值
    # 模拟OR值随痰湿积分增加而增加，但增幅较小
    or_high_activity = np.exp(0.015 * (tan_shi_scores - 40))
    or_high_activity[tan_shi_scores < 40] = 1.0
    
    return tan_shi_scores, or_low_activity, or_high_activity

def create_risk_spline_chart():
    """创建痰湿体质风险预警的限制性立方样条+活动分层曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 生成数据
    tan_shi_scores, or_low_activity, or_high_activity = generate_or_values()
    
    # 绘制低活动量组曲线（红色实线）
    ax.plot(tan_shi_scores, or_low_activity, 'r-', label='低活动量组（活动量表总分<40）', linewidth=2)
    
    # 绘制高活动量组曲线（蓝色虚线）
    ax.plot(tan_shi_scores, or_high_activity, 'b--', label='高活动量组（活动量表总分≥60）', linewidth=2)
    
    # 设置对数刻度
    ax.set_yscale('log')
    
    # 绘制OR=1处的水平参考线
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    # 绘制痰湿积分=60处的垂直虚线
    ax.axvline(x=60, color='gray', linestyle='--', alpha=0.5)
    ax.text(62, 1.5, '高风险阈值参考', rotation=90, va='center', color='gray')
    
    # 标注低活动量组在积分≥60后的OR值
    # 找到积分=60处的OR值
    idx_60 = np.argmin(np.abs(tan_shi_scores - 60))
    or_value_at_60 = or_low_activity[idx_60]
    ax.text(62, or_value_at_60 * 1.2, f'OR={or_value_at_60:.1f}', color='red', fontweight='bold')
    
    # 设置标题和标签
    ax.set_title('痰湿体质风险预警的限制性立方样条+活动分层曲线', fontsize=16)
    ax.set_xlabel('痰湿积分', fontsize=12)
    ax.set_ylabel('高血脂发病的OR值（对数刻度）', fontsize=12)
    
    # 设置坐标轴范围
    ax.set_xlim(0, 100)
    ax.set_ylim(0.5, 20)
    
    # 添加图例
    ax.legend(loc='upper left')
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def main():
    print("=== 生成痰湿体质风险预警的限制性立方样条+活动分层曲线 ===")
    
    # 创建图表
    print("创建限制性立方样条+活动分层曲线...")
    fig = create_risk_spline_chart()
    
    # 保存图表
    output_path = '痰湿体质风险预警_限制性立方样条+活动分层曲线.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"图表已保存到：{output_path}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()