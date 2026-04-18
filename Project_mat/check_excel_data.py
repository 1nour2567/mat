import pandas as pd

# 读取Excel文件
df = pd.read_excel('data/raw/附件1：样例数据.xlsx')

print("=== 数据形状 ===")
print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")

print("\n=== 列名列表 ===")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

print("\n=== 前5行数据预览 ===")
print(df.head())

print("\n=== 数据类型 ===")
print(df.dtypes)

print("\n=== 数值型列的统计信息 ===")
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:
    print(df[numeric_cols].describe())

print("\n=== 类别型列的统计信息 ===")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head(10))
