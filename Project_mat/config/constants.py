# 赛题约束常量定义

# 成本相关（附表2）
COST_MAPPING = {
    'intervention_type_1': 100,  # 假设值，需要根据实际附表2填写
    'intervention_type_2': 200,
    'intervention_type_3': 300
}

# 阈值相关（附表3）
THRESHOLDS = {
    'risk_level_1': 0.3,  # 假设值，需要根据实际附表3填写
    'risk_level_2': 0.6,
    'risk_level_3': 0.8
}

# 年龄约束
AGE_CONSTRAINTS = {
    'min_age': 18,  # 假设值，需要根据实际约束填写
    'max_age': 65
}

# 其他常量
DEFAULT_TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42