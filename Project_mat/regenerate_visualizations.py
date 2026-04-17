import pandas as pd
import pickle
from src._05_visualization import visualize_results

FINAL_DATA_PATH = 'data/processed/final_data.pkl'
MODEL_OUTPUT_PATH = 'data/processed/models.pkl'

print("加载数据和模型...")
models = pickle.load(open(MODEL_OUTPUT_PATH, 'rb'))

print("重新生成可视化图表...")
visualize_results(FINAL_DATA_PATH, models)

print("完成！所有图表已重新生成。")
