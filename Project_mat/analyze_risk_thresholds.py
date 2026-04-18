# 分析风险等级阈值和分布
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.three_layer_architecture import TripleLayerPredictor

def main():
    print("=" * 80)
    print("高血脂症风险等级分析")
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
    
    # 4. 分析风险等级分布
    print("\n[步骤4] 风险等级分布分析...")
    risk_dist = df_result['最终风险等级'].value_counts()
    print("\n风险等级分布:")
    for level, count in risk_dist.items():
        print(f"  {level}: {count} ({count/len(df)*100:.1f}%)")
    
    # 5. 分析各风险等级的特征阈值
    print("\n[步骤5] 特征分层阈值分析...")
    analyze_thresholds(df_result)
    
    # 6. 输出详细的阈值选取依据
    print("\n[步骤6] 风险等级阈值选取依据...")
    print_threshold_basis()
    
    # 7. 保存结果
    output_path = "data/processed/risk_threshold_analysis.pkl"
    df_result.to_pickle(output_path)
    print(f"\n结果已保存到: {output_path}")
    
    print("\n" + "=" * 80)
    print("风险等级分析完成！")
    print("=" * 80)

def analyze_thresholds(df):
    """
    分析各风险等级的特征阈值
    """
    # 按风险等级分组分析
    risk_levels = df['最终风险等级'].unique()
    
    for level in risk_levels:
        print(f"\n--- {level} 特征分析 ---")
        subset = df[df['最终风险等级'] == level]
        
        # 分析关键特征
        key_features = ['痰湿质', '活动量表总分（ADL总分+IADL总分）', '血脂异常项数', '模型预测概率']
        
        for feature in key_features:
            if feature in subset.columns:
                mean_val = subset[feature].mean()
                min_val = subset[feature].min()
                max_val = subset[feature].max()
                print(f"  {feature}:")
                print(f"    均值: {mean_val:.2f}")
                print(f"    范围: {min_val:.2f} - {max_val:.2f}")
        
        # 分析血脂异常情况
        if '血脂异常项数' in subset.columns:
            lipid_dist = subset['血脂异常项数'].value_counts()
            print("  血脂异常项数分布:")
            for count, num in lipid_dist.items():
                print(f"    {count}项异常: {num}人 ({num/len(subset)*100:.1f}%)")

def print_threshold_basis():
    """
    输出风险等级阈值选取依据
    """
    print("\n=== 三级风险阈值选取依据 ===")
    
    print("\n1. 高风险（临床确诊高风险）:")
    print("   - 依据：血脂异常项数 N_i ≥ 1")
    print("   - 具体标准：")
    print("     * TC（总胆固醇）异常：> 6.2 或 < 3.1")
    print("     * TG（甘油三酯）异常：> 1.7 或 < 0.56")
    print("     * LDL-C（低密度脂蛋白）异常：> 3.1 或 < 2.07")
    print("     * HDL-C（高密度脂蛋白）异常：< 1.04")
    
    print("\n2. 高风险（中医预警高风险）:")
    print("   - 依据：模型预测概率在不确定区间 [0.25, 0.5]，且符合中医升档规则")
    print("   - 具体标准：痰湿质 ≥ 60 且 活动量表总分 < 40")
    
    print("\n3. 中风险:")
    print("   - 依据：模型预测概率在 [0.25, 0.5] 区间")
    print("   - 未触发中医升档或降档规则")
    
    print("\n4. 低风险:")
    print("   - 依据：模型预测概率 < 0.25")
    
    print("\n5. 低风险（中医支持）:")
    print("   - 依据：模型预测概率在不确定区间 [0.25, 0.5]，且符合中医降档规则")
    print("   - 具体标准：痰湿质 < 60 且 活动量表总分 ≥ 60")
    
    print("\n=== 阈值选取说明 ===")
    print("1. 血脂异常标准：严格执行赛题给出的临床阈值")
    print("2. 模型概率阈值：基于用户要求，设置为 0.25 和 0.5")
    print("3. 中医规则阈值：")
    print("   - 升档阈值 (60分/40分)：根据用户要求调整")
    print("   - 降档阈值 (60分/60分)：基于赛题附表中的基础调理和高强度活动分界线")

if __name__ == "__main__":
    main()
