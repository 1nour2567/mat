# 数据清洗与初始衍生模块
import pandas as pd
import numpy as np
from config.constants import AGE_CONSTRAINTS, STRATIFICATION

def load_raw_data(file_path):
    """加载原始数据"""
    return pd.read_excel(file_path)

def clean_data(df):
    """数据清洗"""
    # 5.0.1 数据质量诊断
    print("\n=== 5.0.1 数据质量诊断 ===")
    print(f"样本数量: {len(df)}")
    print(f"字段数量: {len(df.columns)}")
    
    # 数据完整性检查
    missing_values = df.isnull().sum().sum()
    duplicate_samples = df.duplicated().sum()
    print(f"缺失值数量: {missing_values}")
    print(f"重复样本数量: {duplicate_samples}")
    
    # 标签分布检查
    if '高血脂症二分类标签' in df.columns:
        positive_samples = df['高血脂症二分类标签'].sum()
        positive_ratio = positive_samples / len(df) * 100
        print(f"高血脂确诊样本: {positive_samples} ({positive_ratio:.1f}%)")
    
    if '痰湿质' in df.columns:
        phlegm_samples = (df['痰湿质'] > 60).sum()  # 假设痰湿质得分>60为痰湿质
        phlegm_ratio = phlegm_samples / len(df) * 100
        print(f"痰湿质样本: {phlegm_samples} ({phlegm_ratio:.1f}%)")
    
    # 体质标签与九种体质积分一致性检查（考虑平和质特殊规则）
    if '体质标签' in df.columns:
        constitution_cols = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
        inconsistent_count = 0
        
        # 体质判定规则
        def determine_constitution(row):
            # 平和质判定：平和分达到阈值（假设60），同时其他失衡体质分数不能太高（假设<40）
            if row['平和质'] >= 60:
                other_scores = [row[col] for col in constitution_cols if col != '平和质']
                if all(score < 40 for score in other_scores):
                    return '平和质'
            
            # 偏颇体质判定：分数最高且达到成立阈值（假设40）
            max_score = max(row[constitution_cols])
            if max_score >= 40:
                max_constitution = row[constitution_cols].idxmax()
                return max_constitution
            
            # 无明显体质倾向
            return None
        
        for idx, row in df.iterrows():
            # 实际体质标签
            label_mapping = {
                1: '平和质', 2: '气虚质', 3: '阳虚质', 4: '阴虚质',
                5: '痰湿质', 6: '湿热质', 7: '血瘀质', 8: '气郁质', 9: '特禀质'
            }
            actual_label = label_mapping.get(row['体质标签'])
            
            # 计算应该的体质
            predicted_label = determine_constitution(row)
            
            if actual_label != predicted_label:
                inconsistent_count += 1
        
        print(f"体质标签与积分不一致样本: {inconsistent_count}")
        print("注：平和质判定需满足'平和分达到阈值，同时其他失衡体质分数不能太高'")
        print("注：偏颇体质判定需满足'分数最高且达到成立阈值'")
    
    # 血脂异常与诊断标签对应关系检查
    if '高血脂症二分类标签' in df.columns and '血脂异常项数' in df.columns:
        matched = ((df['高血脂症二分类标签'] == 1) & (df['血脂异常项数'] > 0)).sum()
        total_positive = df['高血脂症二分类标签'].sum()
        if total_positive > 0:
            match_ratio = matched / total_positive * 100
            print(f"血脂异常与诊断标签匹配率: {match_ratio:.1f}%")
    
    # 处理缺失值（根据诊断结论，无缺失值）
    df = df.dropna()
    
    return df

def feature_derivation(df):
    """初始特征衍生"""
    # 5.0.2 预处理执行方案
    print("\n=== 5.0.2 预处理执行方案 ===")
    print("1. 基础字段保留：保留全部37个原始字段")
    
    # 代谢派生指标
    print("\n2. 生成代谢派生指标：")
    
    if 'TC（总胆固醇）' in df.columns and 'HDL-C（高密度脂蛋白）' in df.columns:
        df['non-HDL-C'] = df['TC（总胆固醇）'] - df['HDL-C（高密度脂蛋白）']
        print("   - non-HDL-C")
    
    if 'TG（甘油三酯）' in df.columns and 'HDL-C（高密度脂蛋白）' in df.columns:
        df['AIP'] = np.log10(df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）'])
        print("   - AIP")
    
    if 'TC（总胆固醇）' in df.columns and 'HDL-C（高密度脂蛋白）' in df.columns:
        df['TC/HDL比值'] = df['TC（总胆固醇）'] / df['HDL-C（高密度脂蛋白）']
        print("   - TC/HDL比值")
    
    if 'LDL-C（低密度脂蛋白）' in df.columns and 'HDL-C（高密度脂蛋白）' in df.columns:
        df['LDL/HDL比值'] = df['LDL-C（低密度脂蛋白）'] / df['HDL-C（高密度脂蛋白）']
        print("   - LDL/HDL比值")
    
    if 'TG（甘油三酯）' in df.columns and 'HDL-C（高密度脂蛋白）' in df.columns:
        df['TG/HDL比值'] = df['TG（甘油三酯）'] / df['HDL-C（高密度脂蛋白）']
        print("   - TG/HDL比值")
    
    # 1%-99%缩尾版本
    print("\n3. 生成1%-99%缩尾版本：")
    for col in ['non-HDL-C', 'AIP', 'TC/HDL比值', 'LDL/HDL比值', 'TG/HDL比值']:
        if col in df.columns:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[f'{col}_缩尾'] = df[col].clip(lower=q1, upper=q99)
            print(f"   - {col}_缩尾")
    
    # 临床异常标志
    print("\n4. 生成临床异常标志：")
    
    if 'TC（总胆固醇）' in df.columns:
        df['TC异常'] = (df['TC（总胆固醇）'] > 5.2).astype(int)
        print("   - TC异常")
    
    if 'TG（甘油三酯）' in df.columns:
        df['TG异常'] = (df['TG（甘油三酯）'] > 1.7).astype(int)
        print("   - TG异常")
    
    if 'LDL-C（低密度脂蛋白）' in df.columns:
        df['LDL-C异常'] = (df['LDL-C（低密度脂蛋白）'] > 3.4).astype(int)
        print("   - LDL-C异常")
    
    if 'HDL-C（高密度脂蛋白）' in df.columns:
        df['HDL-C异常'] = (df['HDL-C（高密度脂蛋白）'] < 1.0).astype(int)
        print("   - HDL-C异常")
    
    # 血脂异常项数
    lipid_abnormalities = ['TC异常', 'TG异常', 'LDL-C异常', 'HDL-C异常']
    # 只选择存在的列
    existing_lipid_abnormalities = [col for col in lipid_abnormalities if col in df.columns]
    if existing_lipid_abnormalities:
        df['血脂异常项数'] = df[existing_lipid_abnormalities].sum(axis=1)
        print("   - 血脂异常项数")
    else:
        df['血脂异常项数'] = 0
    
    if '血尿酸' in df.columns:
        df['尿酸异常'] = (df['血尿酸'] > 420).astype(int)
        print("   - 尿酸异常")
    
    # 年龄组已经存在，不需要映射
    if '年龄组' not in df.columns:
        # 如果年龄组不存在，可以根据其他信息生成
        pass
    
    # 赛题约束分层变量
    print("\n5. 生成赛题约束分层变量：")
    
    if 'ADL总分' in df.columns:
        # 活动能力分层（<40/40-59/≥60）
        df['活动能力分层'] = pd.cut(df['ADL总分'], bins=[-float('inf'), 39, 59, float('inf')], labels=['<40', '40-59', '≥60'])
        print("   - 活动能力分层（<40/40-59/≥60）")
    
    if '痰湿质' in df.columns:
        # 痰湿积分分层（≤58/59-61/≥62）
        df['痰湿积分分层'] = pd.cut(df['痰湿质'], bins=[-float('inf'), 58, 61, float('inf')], labels=['≤58', '59-61', '≥62'])
        print("   - 痰湿积分分层（≤58/59-61/≥62）")
    
    print("\n6. 类别不平衡处理方案：")
    print("   - 采样：SMOTE-Tomek算法")
    print("   - 模型：Focal Loss损失函数")
    print("   - 评价：Macro-F1、AUC-ROC、PR曲线、召回率")
    
    return df

def preprocess_data(raw_data_path, output_path):
    """完整预处理流程"""
    # 加载数据
    df = load_raw_data(raw_data_path)
    
    # 清洗数据
    df = clean_data(df)
    
    # 特征衍生
    df = feature_derivation(df)
    
    # 保存处理后的数据
    df.to_pickle(output_path)
    
    return df