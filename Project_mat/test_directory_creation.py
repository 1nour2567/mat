from src.preprocessing import preprocess_data
import os
import tempfile
import shutil

temp_dir = tempfile.mkdtemp()
output_path = os.path.join(temp_dir, 'subdir', 'test_output.pkl')
print('Testing directory creation...')
print('Output path:', output_path)
try:
    preprocess_data('data/raw/附件1：样例数据.xlsx', output_path)
    print('SUCCESS: Directory created and file saved')
    shutil.rmtree(temp_dir)
except Exception as e:
    print('ERROR:', str(e))
    shutil.rmtree(temp_dir)