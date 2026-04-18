# 分析被误诊样本的血脂异常情况
import pandas as pd
import numpy as np

def main():
    print("=" * 80)
    print("分析被误诊样本的血脂异常情况")
    print("=" * 80)
    
    # 加载被误诊样本
    print("\n[步骤1] 加载被误诊样本数据...")
    pkl_path = "data/processed/misdiagnosed_samples.pkl"
    df = pd.read_pickle(pkl_path)
    print(f"数据加载完成，形状: {df.shape}")
    
    # 重新计算血脂异常项数，验证计算是否正确
    print("\n[步骤2] 验证血脂异常项数计算...")
    verify_lipid_abnormal_count(df)
    
    # 分析具体的血脂异常情况
    print("\n[步骤3] 分析具体血脂异常指标...")
    analyze_specific_abnormalities(df)
    
    # 分析边界值情况
    print("\n[步骤4] 分析边界值附近的异常...")
    analyze_boundary_cases(df)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

def verify_lipid_abnormal_count(df):
    """
    验证血脂异常项数的计算是否正确
    """
    # 重新计算血脂异常项数
    def recalculate_abnormal_count(row):
        checks = [
            row['TC（总胆固醇）'] > 6.2 or row['TC（总胆固醇）'] < 3.1,
            row['TG（甘油三酯）'] > 1.7 or row['TG（甘油三酯）'] < 0.56,
            row['LDL-C（低密度脂蛋白）'] > 3.1 or row['LDL-C（低密度脂蛋白）'] < 2.07,
            row['HDL-C（高密度脂蛋白）'] < 1.04
        ]
        return sum(checks)
    
    # 计算新的异常项数
    df['重新计算的血脂异常项数'] = df.apply(recalculate_abnormal_count, axis=1)
    
    # 比较原始计算和重新计算的结果
    mismatch = (df['血脂异常项数'] != df['重新计算的血脂异常项数']).sum()
    print(f"重新计算的血脂异常项数与原始计算的匹配情况: {len(df) - mismatch}/{len(df)} 匹配")
    
    if mismatch > 0:
        print(f"存在 {mismatch} 个不匹配的样本")
    else:
        print("所有样本的血脂异常项数计算正确")

def analyze_specific_abnormalities(df):
    """
    分析具体的血脂异常指标
    """
    # 定义异常判断函数
    def is_tc_abnormal(val):
        return val > 6.2 or val < 3.1
    
    def is_tg_abnormal(val):
        return val > 1.7 or val < 0.56
    
    def is_ldl_abnormal(val):
        return val > 3.1 or val < 2.07
    
    def is_hdl_abnormal(val):
        return val < 1.04
    
    # 计算各项异常情况
    df['TC异常'] = df['TC（总胆固醇）'].apply(is_tc_abnormal)
    df['TG异常'] = df['TG（甘油三酯）'].apply(is_tg_abnormal)
    df['LDL异常'] = df['LDL-C（低密度脂蛋白）'].apply(is_ldl_abnormal)
    df['HDL异常'] = df['HDL-C（高密度脂蛋白）'].apply(is_hdl_abnormal)
    
    # 统计各项异常的分布
    print("\n各血脂指标异常分布:")
    abnormal_counts = {
        'TC异常': df['TC异常'].sum(),
        'TG异常': df['TG异常'].sum(),
        'LDL异常': df['LDL异常'].sum(),
        'HDL异常': df['HDL异常'].sum()
    }
    
    for key, count in abnormal_counts.items():
        print(f"  {key}: {count}人 ({count/len(df)*100:.1f}%)")
    
    # 分析异常组合
    print("\n异常组合分析:")
    # 创建异常组合字符串
    def create_abnormal_combination(row):
        abnormal = []
        if row['TC异常']:
            abnormal.append('TC')
        if row['TG异常']:
            abnormal.append('TG')
        if row['LDL异常']:
            abnormal.append('LDL')
        if row['HDL异常']:
            abnormal.append('HDL')
        return '+'.join(abnormal) if abnormal else '无'
    
    df['异常组合'] = df.apply(create_abnormal_combination, axis=1)
    combination_counts = df['异常组合'].value_counts()
    for combo, count in combination_counts.items():
        if combo != '无':
            print(f"  {combo}: {count}人 ({count/len(df)*100:.1f}%)")

def analyze_boundary_cases(df):
    """
    分析边界值附近的异常情况
    """
    # 定义边界值范围
    boundaries = {
        'TC（总胆固醇）': [(3.0, 3.2), (6.1, 6.3)],
        'TG（甘油三酯）': [(0.5, 0.6), (1.6, 1.8)],
        'LDL-C（低密度脂蛋白）': [(2.0, 2.1), (3.0, 3.2)],
        'HDL-C（高密度脂蛋白）': [(1.0, 1.1)]
    }
    
    print("\n边界值附近的异常情况:")
    for lipid, ranges in boundaries.items():
        boundary_count = 0
        for min_val, max_val in ranges:
            # 找到在边界范围内的值
            boundary_cases = df[(df[lipid] >= min_val) & (df[lipid] <= max_val)]
            boundary_count += len(boundary_cases)
        
        if boundary_count > 0:
            print(f"  {lipid}: {boundary_count}人在边界值附近 ({boundary_count/len(df)*100:.1f}%)")
    
    # 分析具体边界值案例
    print("\n具体边界值案例:")
    # 找出接近边界的样本
    boundary_samples = []
    for idx, row in df.iterrows():
        is_boundary = False
        boundary_reasons = []
        
        # TC边界
        if (row['TC（总胆固醇）'] >= 3.0 and row['TC（总胆固醇）'] <= 3.2) or (row['TC（总胆固醇）'] >= 6.1 and row['TC（总胆固醇）'] <= 6.3):
            is_boundary = True
            boundary_reasons.append(f"TC={row['TC（总胆固醇）']:.2f}")
        
        # TG边界
        if (row['TG（甘油三酯）'] >= 0.5 and row['TG（甘油三酯）'] <= 0.6) or (row['TG（甘油三酯）'] >= 1.6 and row['TG（甘油三酯）'] <= 1.8):
            is_boundary = True
            boundary_reasons.append(f"TG={row['TG（甘油三酯）']:.2f}")
        
        # LDL边界
        if (row['LDL-C（低密度脂蛋白）'] >= 2.0 and row['LDL-C（低密度脂蛋白）'] <= 2.1) or (row['LDL-C（低密度脂蛋白）'] >= 3.0 and row['LDL-C（低密度脂蛋白）'] <= 3.2):
            is_boundary = True
            boundary_reasons.append(f"LDL={row['LDL-C（低密度脂蛋白）']:.2f}")
        
        # HDL边界
        if row['HDL-C（高密度脂蛋白）'] >= 1.0 and row['HDL-C（高密度脂蛋白）'] <= 1.1:
            is_boundary = True
            boundary_reasons.append(f"HDL={row['HDL-C（高密度脂蛋白）']:.2f}")
        
        if is_boundary:
            boundary_samples.append({"样本索引": idx, "边界原因": ", ".join(boundary_reasons)})
    
    # 打印前5个边界案例
    print(f"找到 {len(boundary_samples)} 个边界值附近的样本")
    for i, sample in enumerate(boundary_samples[:5]):
        print(f"  样本{i+1}: {sample['边界原因']}")

if __name__ == "__main__":
    main()