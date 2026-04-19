#!/usr/bin/env python3
# Figure 2: 九种体质性别背靠背条形图
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 来自问题一分析报告的真实数据
constitutions = ['平和质', '气虚质', '阳虚质', '阴虚质',
                 '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
male_values = [0.0282, 0.1305, 0.1628, 0.0581,
               0.0725, 0.0517, 0.0553, 0.0063, 0.0367]
female_values = [0.1578, 0.1370, 0.1095, 0.0330,
                 0.0754, 0.0672, 0.0409, 0.0339, 0.0080]

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))

y = np.arange(len(constitutions))
bar_width = 0.4

# 绘制男性 (左侧，蓝色)
male_bars = ax.barh(y - bar_width/2, [-v for v in male_values],
                    bar_width, label='男性', color='#1f77b4', edgecolor='white')

# 绘制女性 (右侧，红色)
female_bars = ax.barh(y + bar_width/2, female_values,
                      bar_width, label='女性', color='#d62728', edgecolor='white')

# 高亮痰湿质所在行
tan_shi_idx = constitutions.index('痰湿质')
male_bars[tan_shi_idx].set_edgecolor('#ff7f0e')
male_bars[tan_shi_idx].set_linewidth(3)
female_bars[tan_shi_idx].set_edgecolor('#ff7f0e')
female_bars[tan_shi_idx].set_linewidth(3)

# 添加数值标签
for i, (m_val, f_val) in enumerate(zip(male_values, female_values)):
    ax.text(-m_val - 0.005, y[i] - bar_width/2, f'{m_val:.4f}',
            va='center', ha='right', fontsize=10)
    ax.text(f_val + 0.005, y[i] + bar_width/2, f'{f_val:.4f}',
            va='center', ha='left', fontsize=10)

# 设置标签和标题
ax.set_yticks(y)
ax.set_yticklabels(constitutions, fontsize=11)
ax.set_xlabel('贡献度', fontsize=12)
ax.set_title('Figure 2: 九种体质对发病风险的贡献度 - 性别差异',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower center', ncol=2, fontsize=11)

# 设置x轴范围
ax.set_xlim(-0.2, 0.2)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

# 在平和质行右侧加脚注
pinghe_idx = constitutions.index('平和质')
ax.text(0.16, y[pinghe_idx] + bar_width/2 + 0.1,
        '*女性平和质高贡献可能源于围绝经期激素波动掩盖体质偏颇，提示"虚假平和"现象。',
        ha='left', va='center', color='gray', fontsize=9, wrap=True)

# 添加说明文本
ax.text(0.5, 1.02, '左侧蓝色条：男性贡献度  |  右侧红色条：女性贡献度',
        ha='center', va='bottom', color='black', fontsize=10,
        transform=ax.get_xaxis_transform())

plt.tight_layout()
plt.savefig('/workspace/Project_mat/Figure_2_性别背靠背条形图.png',
            dpi=300, bbox_inches='tight')
print('Figure 2 saved successfully!')
plt.show()

