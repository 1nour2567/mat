#!/usr/bin/env python3
# 生成九种体质贡献度的饼图

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def create_pie_chart():
    # 体质名称
    constitutions = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
    
    # 男性贡献度
    male_contributions = [0.0282, 0.1305, 0.1628, 0.0581, 0.0725, 0.0517, 0.0553, 0.0063, 0.0367]
    
    # 女性贡献度
    female_contributions = [0.1578, 0.1370, 0.1095, 0.0330, 0.0754, 0.0672, 0.0409, 0.0339, 0.0080]
    
    # 总体贡献度（男女平均）
    total_contributions = [(m + f) / 2 for m, f in zip(male_contributions, female_contributions)]
    
    # 颜色设置
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666', '#c2f0c2', '#ffdb4d']
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 男性贡献度饼图
    axes[0].pie(male_contributions, labels=constitutions, autopct='%1.2f%%', startangle=90, colors=colors)
    axes[0].set_title('男性九种体质贡献度', fontsize=14, fontweight='bold')
    axes[0].axis('equal')  # 确保饼图是圆的
    
    # 女性贡献度饼图
    axes[1].pie(female_contributions, labels=constitutions, autopct='%1.2f%%', startangle=90, colors=colors)
    axes[1].set_title('女性九种体质贡献度', fontsize=14, fontweight='bold')
    axes[1].axis('equal')
    
    # 总体贡献度饼图
    axes[2].pie(total_contributions, labels=constitutions, autopct='%1.2f%%', startangle=90, colors=colors)
    axes[2].set_title('总体九种体质贡献度', fontsize=14, fontweight='bold')
    axes[2].axis('equal')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('九种体质贡献度饼图.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    print('九种体质贡献度饼图已成功生成！')

if __name__ == "__main__":
    create_pie_chart()
