# 赛题约束常量定义

# 成本相关（附表2）
COST_MAPPING = {
    'intervention_type_1': 50,   # 低强度干预
    'intervention_type_2': 150,  # 中强度干预
    'intervention_type_3': 300   # 高强度干预
}

# 阈值相关（附表3）
THRESHOLDS = {
    'risk_level_1': 0.25,  # 低风险阈值
    'risk_level_2': 0.5,   # 中风险阈值
    'risk_level_3': 0.75   # 高风险阈值
}

# 年龄约束
AGE_CONSTRAINTS = {
    'min_age': 18,  # 最小年龄
    'max_age': 65   # 最大年龄
}

# 活动能力分层
ACTIVITY_LEVELS = {
    'low': '<40',      # 低活动能力
    'medium': '40-59', # 中等活动能力
    'high': '≥60'      # 高活动能力
}

# 痰湿积分分层
PHLEGM_DAMP_SCORES = {
    'low': '≤58',      # 低痰湿积分
    'medium': '59-61', # 中等痰湿积分
    'high': '≥62'      # 高痰湿积分
}

# 其他常量
DEFAULT_TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42