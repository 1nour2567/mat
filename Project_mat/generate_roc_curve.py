#!/usr/bin/env python3
# 生成5折交叉验证ROC曲线图

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150

def generate_roc_curve():
    """生成5折交叉验证ROC曲线图"""
    print("=" * 80)
    print("生成5折交叉验证ROC曲线图")
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
    
    # 提取预测结果
    print("\n提取5折预测结果...")
    try:
        model_layer = predictor.model_layer
        oof_preds_dict = model_layer.oof_preds_dict
        val_indices = model_layer.val_indices
        y_true = model_layer.y_true
        print(f"成功提取 {len(oof_preds_dict)} 折的预测结果")
    except Exception as e:
        print(f"提取预测结果失败: {e}")
        return
    
    # 计算ROC曲线
    print("\n计算ROC曲线...")
    
    # 创建统一的FPR网格
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    for fold in range(len(oof_preds_dict)):
        # 获取该折的验证集索引和预测
        val_idx = val_indices[fold]
        y_val = y_true[val_idx]
        val_pred = oof_preds_dict[fold]
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_val, val_pred)
        
        # 插值到统一的FPR网格
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0  # 确保起点为(0,0)
        tprs.append(tpr_interp)
        
        # 计算AUC
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)
        print(f"Fold {fold+1} AUC: {fold_auc:.4f}")
    
    # 计算平均TPR和平均AUC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # 确保终点为(1,1)
    mean_auc = auc(mean_fpr, mean_tpr)
    print(f"\n平均AUC: {mean_auc:.4f}")
    
    # 绘制ROC曲线
    print("\n绘制ROC曲线...")
    plt.figure(figsize=(10, 8))
    
    # 绘制各折的ROC曲线
    for i, (tpr_interp, fold_auc) in enumerate(zip(tprs, aucs)):
        plt.plot(mean_fpr, tpr_interp, lw=1, alpha=0.3, 
                 label=f'Fold {i+1} (AUC = {fold_auc:.4f})')
    
    # 绘制平均ROC曲线
    plt.plot(mean_fpr, mean_tpr, 'b-', lw=2, 
             label=f'Mean ROC (AUC = {mean_auc:.4f})')
    
    # 绘制对角线参考线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    
    # 添加标题和标签
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('5-Fold Cross Validation ROC Curve', fontsize=14, fontweight='bold')
    
    # 添加图例
    plt.legend(loc='lower right', fontsize=10)
    
    # 添加网格线
    plt.grid(alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = '/workspace/Project_mat/5折交叉验证ROC曲线.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nROC曲线图已保存到: {output_path}")
    
    plt.show()
    
    return output_path

if __name__ == '__main__':
    generate_roc_curve()
