# 分析预处理后数据
import pandas as pd
import numpy as np
import sys

# 添加Project_mat到路径
sys.path.append('/workspace/Project_mat')

from src.feature_engineering import (
    build_feature_pool,
    calculate_spearman_correlation,
    calculate_mutual_info,
    calculate_pls_loadings,
    entropy_weight_method,
    select_features,
    analyze_constitution_contribution
)
from src.preprocessing import clean_data

def load_preprocessed_data(file_path):
    """加载预处理后的数据"""
    # 尝试不同编码读取
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
    
    for encoding in encodings:
        try:
            # 读取文本文件
            df = pd.read_csv(file_path, sep='\s+', encoding=encoding)
            # 清理列名（去除空格）
            df.columns = df.columns.str.strip()
            print(f"成功使用 {encoding} 编码读取文件")
            return df
        except Exception as e:
            print(f"尝试 {encoding} 编码失败: {e}")
    
    # 如果所有编码都失败，尝试二进制读取
    print("尝试二进制读取...")
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # 检测编码
    import chardet
    result = chardet.detect(content)
    encoding = result['encoding']
    confidence = result['confidence']
    print(f"检测到编码: {encoding} (置信度: {confidence})")
    
    # 使用检测到的编码
    try:
        df = pd.read_csv(file_path, sep='\s+', encoding=encoding)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"使用检测到的编码 {encoding} 失败: {e}")
        raise

def main():
    """主分析函数"""
    # 加载数据
    file_path = '/workspace/预处理后数据.txt'
    print("正在加载预处理后的数据...")
    df = load_preprocessed_data(file_path)
    
    print(f"数据维度: {df.shape}")
    print(f"列数: {len(df.columns)}")
    print(f"列名: {df.columns.tolist()}")
    
    # 数据清洗
    print("\n正在进行数据清洗...")
    df = clean_data(df)
    
    # 处理NaN值
    df = df.fillna(0)
    
    # 处理分层变量
    if '活动能力分层' in df.columns:
        activity_mapping = {'低活动能力(<40)': 0, '中活动能力[40,60)': 1, '高活动能力(>=60)': 2,
                          '<40': 0, '40-59': 1, '≥60': 2,
                          '低活动能力': 0, '中活动能力': 1, '高活动能力': 2}
        df['活动能力分层'] = df['活动能力分层'].map(activity_mapping).fillna(0)
    
    if '痰湿积分分层' in df.columns:
        phlegm_mapping = {'轻度痰湿(≤58)': 0, '中度痰湿(59-61)': 1, '重度痰湿(>=62)': 2,
                         '≤58': 0, '59-61': 1, '≥62': 2,
                         '轻度痰湿': 0, '中度痰湿': 1, '重度痰湿': 2}
        df['痰湿积分分层'] = df['痰湿积分分层'].map(phlegm_mapping).fillna(0)
    
    # 5.1.2 构建特征池
    print("\n=== 5.1.2 构建三级候选特征池 ===")
    df, features = build_feature_pool(df)
    
    # 目标变量
    target = '高血脂症二分类标签'
    if target not in df.columns:
        # 尝试其他可能的目标变量名
        possible_targets = ['高血脂诊断', '高血脂', '诊断']
        for t in possible_targets:
            if t in df.columns:
                target = t
                break
    
    print(f"目标变量: {target}")
    
    # 5.1.3 基于Spearman-互信息-PLS的双目标联合筛选
    print("\n=== 5.1.3 基于Spearman-互信息-PLS的双目标联合筛选 ===")
    selected_features = select_features(df, features, target, phlegm_score_col='痰湿质')
    
    # 5.1.4 九种体质风险贡献度分析
    print("\n=== 5.1.4 九种体质风险贡献度分析 ===")
    analyze_constitution_contribution(df, target)
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()
