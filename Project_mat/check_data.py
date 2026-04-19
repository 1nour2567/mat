import pandas as pd

# 加载数据
df = pd.read_excel('data/raw/附件1：样例数据.xlsx')

# 打印列名
print("列名:")
print(df.columns.tolist())

# 打印前5行
print("\n前5行数据:")
print(df.head())

# 打印数据类型
print("\n数据类型:")
print(df.dtypes)
