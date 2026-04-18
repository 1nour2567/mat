# 将 pickle 文件转换为 CSV 格式
import pandas as pd
import sys
import os

def main():
    print("=" * 80)
    print("Pickle 转换为 CSV")
    print("=" * 80)
    
    # 输入文件路径
    input_path = "data/processed/pu_learning_results.pkl"
    output_path = "data/processed/pu_learning_results.csv"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：文件 {input_path} 不存在")
        return
    
    # 读取 pickle 文件
    print(f"\n正在读取 {input_path} ...")
    df = pd.read_pickle(input_path)
    print(f"数据加载完成，形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 保存为 CSV 文件
    print(f"\n正在保存到 {output_path} ...")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n文件成功保存！")
    print(f"输出路径: {os.path.abspath(output_path)}")
    print(f"大小: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    print("\n" + "=" * 80)
    print("转换完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()