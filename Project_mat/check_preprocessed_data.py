import pandas as pd

# 读取预处理后的数据
df = pd.read_pickle('data/processed/preprocessed_data.pkl')

print("=== 预处理后数据的所有列：")
for i, col in enumerate(sorted(df.columns)):
    print(f"{i+1}. {col}")

print("\n=== 与血脂相关的列：")
lipid_related = [col for col in df.columns if any(keyword in col for keyword in ['TC', 'TG', 'HDL', 'LDL', '血脂', '胆固醇', '甘油', '脂蛋白', 'non-HDL', 'AIP'])]
for col in sorted(lipid_related):
    print(f"- {col}")
