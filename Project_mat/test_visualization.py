from src.visualization import visualize_results
import pandas as pd
import numpy as np
import tempfile
import os

# 创建一个简单的测试数据框
data = {
    'risk_level': [1, 2, 3, 1, 2],
    'risk_probability': [0.2, 0.5, 0.8, 0.3, 0.6],
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [2, 3, 4, 5, 6],
    'feature3': [3, 4, 5, 6, 7]
}
df = pd.DataFrame(data)

# 保存为临时文件
temp_dir = tempfile.mkdtemp()
test_file = os.path.join(temp_dir, 'test_data.pkl')
df.to_pickle(test_file)

print('Testing visualization...')
try:
    result = visualize_results(test_file)
    print('SUCCESS: Visualization completed without errors')
except Exception as e:
    print('ERROR:', str(e))
finally:
    import shutil
    shutil.rmtree(temp_dir)