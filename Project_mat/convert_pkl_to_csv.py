# 将 PKL 文件转换为 CSV 文件
import pandas as pd
import os

def main():
    print("=" * 80)
    print("PKL 转 CSV 转换工具")
    print("=" * 80)
    
    # 输入文件路径
    pkl_file = "data/processed/pu_learning_results.pkl"
    
    # 检查文件是否存在
    if not os.path.exists(pkl_file):
        print(f"错误: 文件 {pkl_file} 不存在!")
        return
    
    print(f"\n读取文件: {pkl_file}")
    
    # 读取 PKL 文件
    df = pd.read_pickle(pkl_file)
    print(f"数据形状: {df.shape}")
    
    # 输出文件路径
    csv_file = "data/processed/pu_learning_results.csv"
    
    # 保存为 CSV 文件
    print(f"\n保存文件: {csv_file}")
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    print(f"\n转换成功! CSV 文件已保存至: {csv_file}")
    
    # 显示前几行数据
    print("\n数据预览:")
    print(df.head())
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()