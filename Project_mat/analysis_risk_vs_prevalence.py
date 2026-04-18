# 分析最终风险等级与真实患病率的交叉统计
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.three_layer_architecture import TripleLayerPredictor

def main():
    print("=" * 80)
    print("最终风险等级与真实患病率交叉统计分析")
    print("=" * 80)
    
    # 1. 加载预处理后的数据
    print("\n[步骤1] 加载数据...")
    data_path = "data/processed/preprocessed_data.pkl"
    df = pd.read_pickle(data_path)
    print(f"数据加载完成，形状: {df.shape}")
    
    # 2. 初始化并训练三层预测器
    print("\n[步骤2] 训练三层模型...")
    predictor = TripleLayerPredictor()
    predictor.fit(df)
    
    # 3. 预测
    print("\n[步骤3] 预测风险等级...")
    df_result = predictor.predict(df)
    
    # 4. 生成交叉统计表
    print("\n[步骤4] 生成交叉统计表...")
    generate_cross_table(df_result)
    
    print("\n" + "=" * 80)
    print("交叉统计分析完成！")
    print("=" * 80)

def generate_cross_table(df):
    """
    生成最终风险等级与真实患病率的交叉统计表
    """
    # 确保标签列存在
    if '高血脂症二分类标签' not in df.columns:
        print("错误：数据中缺少 '高血脂症二分类标签' 列")
        return
    
    # 确保最终风险等级列存在
    if '最终风险等级' not in df.columns:
        print("错误：数据中缺少 '最终风险等级' 列")
        return
    
    # 生成交叉表
    cross_table = pd.crosstab(
        df['最终风险等级'],
        df['高血脂症二分类标签'],
        rownames=['最终风险等级'],
        colnames=['真实患病率 (0=无, 1=有)'],
        margins=True,
        margins_name='总计'
    )
    
    # 计算百分比
    cross_table_percent = cross_table.div(cross_table['总计'], axis=0) * 100
    
    print("\n=== 最终风险等级与真实患病率交叉统计表 ===")
    print("\n频数统计:")
    print(cross_table)
    
    print("\n百分比统计 (%):")
    print(cross_table_percent.round(2))
    
    # 计算各风险等级的患病比例
    print("\n=== 各风险等级患病比例 ===")
    risk_levels = df['最终风险等级'].unique()
    for level in risk_levels:
        subset = df[df['最终风险等级'] == level]
        if len(subset) > 0:
            prevalence = subset['高血脂症二分类标签'].mean() * 100
            print(f"{level}: {prevalence:.2f}% ({subset['高血脂症二分类标签'].sum()}/{len(subset)})")
    
    # 保存结果
    output_path = "data/processed/risk_vs_prevalence_cross_table.pkl"
    cross_table.to_pickle(output_path)
    print(f"\n交叉表已保存到: {output_path}")

if __name__ == "__main__":
    main()