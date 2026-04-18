# 分析临床确诊高风险但实际无高血脂症的样本
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.three_layer_architecture import TripleLayerPredictor

def main():
    print("=" * 80)
    print("分析临床确诊高风险但无高血脂症的样本")
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
    
    # 4. 找出被误诊的样本
    print("\n[步骤4] 识别被误诊的样本...")
    analyze_misdiagnosed(df_result)
    
    print("\n" + "=" * 80)
    print("误诊分析完成！")
    print("=" * 80)

def analyze_misdiagnosed(df):
    """
    分析临床确诊高风险但实际无高血脂症的样本
    """
    # 找出被误诊的样本：临床确诊高风险但实际无高血脂症
    misdiagnosed = df[(df['最终风险等级'] == '临床确诊高风险') & (df['高血脂症二分类标签'] == 0)]
    
    print(f"\n=== 被误诊样本分析 ===")
    print(f"被误诊样本数量: {len(misdiagnosed)}")
    print(f"占临床确诊高风险的比例: {len(misdiagnosed)/len(df[df['最终风险等级'] == '临床确诊高风险']) * 100:.2f}%")
    
    if len(misdiagnosed) > 0:
        # 分析被误诊样本的血脂异常情况
        print("\n被误诊样本的血脂异常情况:")
        lipid_abnormal_counts = misdiagnosed['血脂异常项数'].value_counts()
        for count, num in lipid_abnormal_counts.items():
            print(f"  {count}项血脂异常: {num}人 ({num/len(misdiagnosed)*100:.1f}%)")
        
        # 分析被误诊样本的具体血脂指标
        print("\n被误诊样本的血脂指标统计:")
        lipid_cols = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）']
        for col in lipid_cols:
            if col in misdiagnosed.columns:
                mean_val = misdiagnosed[col].mean()
                min_val = misdiagnosed[col].min()
                max_val = misdiagnosed[col].max()
                print(f"  {col}:")
                print(f"    均值: {mean_val:.2f}")
                print(f"    范围: {min_val:.2f} - {max_val:.2f}")
        
        # 分析被误诊样本的其他特征
        print("\n被误诊样本的其他特征统计:")
        other_cols = ['痰湿质', '活动量表总分（ADL总分+IADL总分）', '模型预测概率', '年龄组', '性别']
        for col in other_cols:
            if col in misdiagnosed.columns:
                if col in ['年龄组', '性别']:
                    # 分类变量
                    print(f"  {col}分布:")
                    distribution = misdiagnosed[col].value_counts()
                    for val, count in distribution.items():
                        print(f"    {val}: {count}人 ({count/len(misdiagnosed)*100:.1f}%)")
                else:
                    # 连续变量
                    mean_val = misdiagnosed[col].mean()
                    min_val = misdiagnosed[col].min()
                    max_val = misdiagnosed[col].max()
                    print(f"  {col}:")
                    print(f"    均值: {mean_val:.2f}")
                    print(f"    范围: {min_val:.2f} - {max_val:.2f}")
        
        # 分析误诊原因
        print("\n=== 误诊原因分析 ===")
        print("1. 血脂异常标准问题:")
        print("   - 临床规则层使用的是固定阈值，可能对某些个体不适用")
        print("   - 例如：HDL-C < 1.04 被视为异常，但可能对某些人群来说是正常的")
        
        print("\n2. 血脂指标波动:")
        print("   - 血脂指标可能受到临时因素影响（如饮食、运动等）")
        print("   - 单次检测结果可能不能完全反映真实情况")
        
        print("\n3. 其他因素:")
        print("   - 可能存在其他保护因素，如良好的生活习惯、遗传因素等")
        print("   - 模型预测概率较低，说明综合其他特征判断风险较低")
        
        # 保存被误诊样本
        output_path = "data/processed/misdiagnosed_samples.pkl"
        misdiagnosed.to_pickle(output_path)
        print(f"\n被误诊样本已保存到: {output_path}")
    else:
        print("\n没有发现被误诊的样本")

if __name__ == "__main__":
    main()