# 将被误诊样本转换为CSV格式
import pandas as pd
import os

def main():
    print("=" * 80)
    print("将被误诊样本转换为CSV格式")
    print("=" * 80)
    
    # 检查文件是否存在
    pkl_path = "data/processed/misdiagnosed_samples.pkl"
    if not os.path.exists(pkl_path):
        print(f"错误：文件 {pkl_path} 不存在")
        return
    
    # 加载数据
    print("\n[步骤1] 加载被误诊样本数据...")
    df = pd.read_pickle(pkl_path)
    print(f"数据加载完成，形状: {df.shape}")
    
    # 转换为CSV格式
    print("\n[步骤2] 转换为CSV格式...")
    csv_path = "data/processed/misdiagnosed_samples.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV文件已保存到: {csv_path}")
    
    # 验证文件是否创建成功
    if os.path.exists(csv_path):
        print(f"\n[步骤3] 验证文件创建成功")
        file_size = os.path.getsize(csv_path)
        print(f"CSV文件大小: {file_size} 字节")
    else:
        print("\n错误：CSV文件创建失败")
    
    print("\n" + "=" * 80)
    print("转换完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()