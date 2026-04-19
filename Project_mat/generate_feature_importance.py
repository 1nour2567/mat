#!/usr/bin/env python3
# 生成LightGBM特征重要性Top15条形图

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150

def generate_feature_importance():
    """生成特征重要性图表"""
    print("=" * 80)
    print("生成LightGBM特征重要性Top15条形图")
    print("=" * 80)
    
    # 导入模型
    try:
        from src.three_layer_architecture import TripleLayerPredictor
        print("成功导入TripleLayerPredictor")
    except ImportError as e:
        print(f"导入失败: {e}")
        return
    
    # 加载数据
    try:
        print("\n加载数据...")
        df = pd.read_pickle('data/processed/preprocessed_data.pkl')
        print(f"数据加载成功，形状: {df.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 训练模型
    print("\n训练模型...")
    predictor = TripleLayerPredictor()
    try:
        predictor.fit(df, target_col='高血脂症二分类标签')
        print("模型训练成功！")
    except Exception as e:
        print(f"模型训练失败: {e}")
        return
    
    # 提取特征重要性
    print("\n提取特征重要性...")
    try:
        all_importances = np.array([m.feature_importances_ for m in predictor.model_layer.models])
        mean_imp = all_importances.mean(axis=0)
        feature_names = predictor.model_layer.feature_names
        print(f"特征数量: {len(feature_names)}")
        print(f"特征重要性形状: {mean_imp.shape}")
    except Exception as e:
        print(f"提取特征重要性失败: {e}")
        return
    
    # 排序取Top15
    idx = np.argsort(mean_imp)[-15:]
    top_features = [feature_names[i] for i in idx][::-1]
    top_importances = mean_imp[idx][::-1]
    
    # 标记交叉特征
    cross_features = ['痰湿质×BMI', '痰湿质×活动量表', '痰湿质×血尿酸', '气虚质×BMI', '气虚质×活动量表']
    is_cross = [1 if feat in cross_features else 0 for feat in top_features]
    
    # 分配颜色
    colors = ['#E65100' if is_cross[i] else '#1976D2' for i in range(len(top_features))]
    
    # 检查重点交叉特征是否进入前5
    top5_features = top_features[:5]
    tan_shi_activity = '痰湿质×活动量表' in top5_features
    tan_shi_bmi = '痰湿质×BMI' in top5_features
    
    print(f"\n重点交叉特征检查:")
    print(f"  痰湿质×活动量表 在前5: {tan_shi_activity}")
    print(f"  痰湿质×BMI 在前5: {tan_shi_bmi}")
    
    # 绘制图表
    print("\n绘制特征重要性图...")
    plt.figure(figsize=(12, 8))
    
    # 创建水平条形图
    bars = plt.barh(top_features, top_importances, color=colors)
    
    # 添加特征重要性数值标签
    for bar, imp in zip(bars, top_importances):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.4f}', ha='left', va='center', fontsize=9)
    
    # 添加标题和标签
    plt.xlabel('特征重要性 (Mean Gain)', fontsize=12, fontweight='bold')
    plt.ylabel('特征名称', fontsize=12, fontweight='bold')
    plt.title('LightGBM预测层 - Top 15 重要特征', fontsize=14, fontweight='bold')
    
    # 反转y轴，使重要性高的在上面
    plt.gca().invert_yaxis()
    
    # 添加图例
    cross_patch = plt.Rectangle((0,0), 1, 1, color='#E65100')
    normal_patch = plt.Rectangle((0,0), 1, 1, color='#1976D2')
    plt.legend([cross_patch, normal_patch], ['中西医交叉特征', '普通特征'], loc='lower right')
    
    # 添加网格线
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = '/workspace/Project_mat/LightGBM特征重要性Top15.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n特征重要性图已保存到: {output_path}")
    
    # 打印Top15特征列表
    print("\n=== Top 15 重要特征 ===")
    for i, (feat, imp, is_c) in enumerate(zip(top_features, top_importances, is_cross)):
        cross_mark = " ⭐" if is_c else ""
        print(f"{i+1:2d}. {feat:<20} {imp:.4f}{cross_mark}")
    
    plt.show()
    
    return output_path

if __name__ == '__main__':
    generate_feature_importance()
