#!/usr/bin/env python3
# 修正的九种体质性别背靠背条形图和年龄组热图
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# Figure 2: 九种体质性别背靠背条形图
def create_sex_back_to_back():
    # 九种体质（按贡献度排序）
    constitutions = ['平和质', '气虚质', '阳虚质', '痰湿质', '湿热质', 
                     '血瘀质', '气郁质', '阴虚质', '特禀质']
    
    # 真实数据（来自分析报告）
    male_values = np.array([0.0282, 0.1305, 0.1628, 0.0725, 0.0517, 
                            0.0553, 0.0063, 0.0581, 0.0367])
    female_values = np.array([0.1578, 0.1370, 0.1095, 0.0754, 0.0672, 
                              0.0409, 0.0339, 0.0330, 0.0080])
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 位置索引
    y = np.arange(len(constitutions))
    bar_width = 0.4
    
    # 绘制条形图（左侧男性，右侧女性）
    ax.barh(y - bar_width/2, -male_values, bar_width, label='男性', color='#1f77b4', alpha=0.8)
    ax.barh(y + bar_width/2, female_values, bar_width, label='女性', color='#ff7f0e', alpha=0.8)
    
    # 找到痰湿质的位置并高亮
    tan_shi_idx = constitutions.index('痰湿质')
    # 给痰湿质的行添加橙色边框
    rect1 = plt.Rectangle((-max(male_values)*1.05, tan_shi_idx - bar_width*0.8), 
                          max(male_values)*2.1, bar_width*1.6, 
                          fill=False, color='#ff9500', linewidth=2)
    ax.add_patch(rect1)
    
    # 标注数值
    for i, (m_val, f_val) in enumerate(zip(male_values, female_values)):
        ax.text(-m_val - 0.005, i - bar_width/2, f'{m_val:.4f}', 
                va='center', ha='right', color='#1f77b4', fontweight='bold')
        ax.text(f_val + 0.005, i + bar_width/2, f'{f_val:.4f}', 
                va='center', ha='left', color='#ff7f0e', fontweight='bold')
    
    # 给平和质添加脚注
    pinghe_idx = constitutions.index('平和质')
    ax.annotate('*女性平和质高贡献可能源于围绝经期激素波动\n掩盖体质偏颇，提示「虚假平和」现象。',
                xy=(female_values[pinghe_idx] + 0.01, pinghe_idx + bar_width/2),
                xytext=(0.18, pinghe_idx),
                arrowprops=dict(arrowstyle='->', color='black'),
                ha='left', va='center', fontsize=9, bbox=dict(boxstyle='round', alpha=0.1))
    
    # 设置标题和标签
    ax.set_title('Figure 2: 九种体质对发病风险贡献度 - 性别差异', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('贡献度', fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(constitutions, fontsize=11)
    
    # 设置x轴范围
    xlim = max(max(male_values), max(female_values)) * 1.15
    ax.set_xlim(-xlim, xlim)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=11)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

# Figure 3: 年龄组体质贡献度热图
def create_age_heatmap():
    # 九种体质（按总贡献度排序）
    constitutions = ['气虚质', '平和质', '阳虚质', '特禀质', '血瘀质', 
                     '气郁质', '湿热质', '阴虚质', '痰湿质']
    
    # 年龄组
    age_groups = ['40-49', '50-59', '60-69', '70-79', '80-89']
    
    # 真实数据（来自分析报告，完全使用原始数据）
    data = np.array([
        [0.6112, 0.0116, 0.1842, 0.0491, 0.4127],  # 气虚质
        [0.3298, 0.0655, 0.0495, 0.2928, 0.0564],  # 平和质
        [0.0441, 0.2158, 0.2094, 0.0447, 0.1177],  # 阳虚质
        [0.0264, 0.0679, 0.3347, 0.1152, 0.0827],  # 特禀质
        [0.0678, 0.1546, 0.0310, 0.2794, 0.2548],  # 血瘀质
        [0.0219, 0.1897, 0.0277, 0.0647, 0.1721],  # 气郁质
        [0.0207, 0.1067, 0.0330, 0.0255, 0.0756],  # 湿热质
        [0.2648, 0.0676, 0.0505, 0.0094, 0.0304],  # 阴虚质
        [0.0343, 0.0455, 0.0811, 0.0042, 0.0233]   # 痰湿质
    ])
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制热图（使用深色调，0.6112应该是最深的颜色）
    im = ax.imshow(data, cmap='Reds', aspect='auto', vmin=0, vmax=0.65)
    
    # 设置标题（在顶部添加生命周期规律标注）
    ax.set_title('青年期（气虚）→ 中年期（阳虚）→ 老年前期（特禀）→ 老年期（平和）→ 高龄期（气虚）\n\nFigure 3: 九种体质对发病风险贡献度 - 年龄组差异', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(age_groups)))
    ax.set_yticks(np.arange(len(constitutions)))
    ax.set_xticklabels(age_groups, fontsize=11)
    ax.set_yticklabels(constitutions, fontsize=11)
    
    # 在格子内标注数值（保留4位小数）
    for i in range(len(constitutions)):
        for j in range(len(age_groups)):
            text = ax.text(j, i, f'{data[i, j]:.4f}', 
                           ha='center', va='center', 
                           color='black' if data[i, j] < 0.4 else 'white', 
                           fontweight='bold', fontsize=10)
    
    # 用黑色粗框标出每个年龄组的Top1
    for j in range(len(age_groups)):
        top_i = np.argmax(data[:, j])
        # 画框
        rect = plt.Rectangle((j - 0.5, top_i - 0.5), 1, 1, 
                             fill=False, color='black', linewidth=3)
        ax.add_patch(rect)
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('贡献度', rotation=-90, va='bottom', fontsize=11)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def main():
    print("=== 生成修正的体质相关图表 ===")
    
    # Figure 2
    print("生成Figure 2: 性别背靠背条形图...")
    fig2 = create_sex_back_to_back()
    fig2.savefig('Figure_2_体质性别差异_修正版.png', dpi=300, bbox_inches='tight')
    
    # Figure 3
    print("生成Figure 3: 年龄组热图...")
    fig3 = create_age_heatmap()
    fig3.savefig('Figure_3_体质年龄差异_修正版.png', dpi=300, bbox_inches='tight')
    
    print("图表保存完成！")
    plt.show()

if __name__ == "__main__":
    main()
