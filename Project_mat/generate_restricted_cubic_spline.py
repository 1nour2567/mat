#!/usr/bin/env python3
# 生成痰湿体质风险预警的限制性立方样条+活动分层曲线

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def load_data():
    """加载数据"""
    try:
        # 加载preprocessed_data.pkl文件
        df = pd.read_pickle('data/processed/preprocessed_data.pkl')
        print(f"成功加载数据，样本数：{len(df)}")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def prepare_data(df):
    """准备数据"""
    # 检查必要的列
    required_cols = ['痰湿质', '活动量表总分（ADL总分+IADL总分）', '高血脂症二分类标签']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"缺少必要的列：{missing_cols}")
        return None
    
    # 重命名列，方便后续处理
    df = df.rename(columns={
        '痰湿质': 'tan_shi_score',
        '活动量表总分（ADL总分+IADL总分）': 'activity_score',
        '高血脂症二分类标签': 'high_lipid'
    })
    
    # 分层：低活动量组（<40）和高活动量组（≥60）
    df['activity_group'] = np.where(df['activity_score'] < 40, 'low', 
                                   np.where(df['activity_score'] >= 60, 'high', 'medium'))
    
    # 只保留低活动量和高活动量组
    df = df[df['activity_group'].isin(['low', 'high'])].copy()
    
    print(f"数据准备完成，低活动量组样本数：{len(df[df['activity_group'] == 'low'])}")
    print(f"高活动量组样本数：{len(df[df['activity_group'] == 'high'])}")
    
    return df

def fit_restricted_cubic_spline(df):
    """拟合限制性立方样条模型"""
    # 为低活动量组和高活动量组分别拟合模型
    models = {}
    predictions = {}
    
    for group in ['low', 'high']:
        group_df = df[df['activity_group'] == group].copy()
        
        # 创建限制性立方样条
        formula = 'high_lipid ~ cr(tan_shi_score, df=3)'
        model = smf.glm(formula=formula, data=group_df, family=sm.families.Binomial()).fit()
        models[group] = model
        
        # 生成预测数据
        tan_shi_range = np.linspace(0, 100, 100)
        pred_df = pd.DataFrame({'tan_shi_score': tan_shi_range})
        
        # 计算OR值（优势比）
        pred_prob = model.predict(pred_df)
        pred_or = pred_prob / (1 - pred_prob)
        predictions[group] = (tan_shi_range, pred_or)
    
    return models, predictions

def plot_restricted_cubic_spline(predictions, models, df):
    """绘制限制性立方样条曲线"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制低活动量组曲线（红色实线）
    tan_shi_low, or_low = predictions['low']
    ax.plot(tan_shi_low, or_low, 'r-', linewidth=2, label='低活动量组（活动量表总分<40）')
    
    # 绘制高活动量组曲线（蓝色虚线）
    tan_shi_high, or_high = predictions['high']
    ax.plot(tan_shi_high, or_high, 'b--', linewidth=2, label='高活动量组（活动量表总分≥60）')
    
    # 绘制OR=1参考线
    ax.axhline(y=1, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # 绘制痰湿积分=60垂直线
    ax.axvline(x=60, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(61, ax.get_ylim()[0] + 0.1, '高风险阈值参考', rotation=90, verticalalignment='bottom')
    
    # 标注低活动量组在积分≥60后的OR值
    low_or_at_60 = or_low[np.argmin(np.abs(tan_shi_low - 60))]
    ax.text(65, low_or_at_60, f'OR≈{low_or_at_60:.1f}', color='red', fontweight='bold')
    
    # 设置坐标轴
    ax.set_xlabel('痰湿积分（0-100）', fontsize=14)
    ax.set_ylabel('高血脂发病的OR值', fontsize=14)
    ax.set_title('痰湿体质风险预警的限制性立方样条+活动分层曲线', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    
    # 计算并标注交互作用p值
    # 这里使用简化的方法，实际应该使用包含交互项的模型
    # 创建交互项模型
    formula_interact = 'high_lipid ~ cr(tan_shi_score, df=3) * activity_group'
    model_interact = smf.glm(formula=formula_interact, data=df, family=sm.families.Binomial()).fit()
    
    # 提取交互项的p值
    interact_pvalue = model_interact.pvalues.get('cr(tan_shi_score, df=3)[0]:activity_group[T.high]', np.nan)
    
    # 添加图注
    ax.text(0.05, 0.95, f'交互作用p值: {interact_pvalue:.4f}', 
            transform=ax.transAxes, verticalalignment='top', fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('痰湿体质风险预警的限制性立方样条+活动分层曲线.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    print('限制性立方样条曲线已成功生成！')

def main():
    print("=== 生成痰湿体质风险预警的限制性立方样条+活动分层曲线 ===")
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 准备数据
    prepared_df = prepare_data(df)
    if prepared_df is None:
        return
    
    # 拟合限制性立方样条模型
    print("拟合限制性立方样条模型...")
    models, predictions = fit_restricted_cubic_spline(prepared_df)
    
    # 绘制曲线
    print("绘制限制性立方样条曲线...")
    plot_restricted_cubic_spline(predictions, models, prepared_df)

if __name__ == "__main__":
    main()
