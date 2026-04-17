# 赛题约束常量定义

# 成本相关（附表2）
COST_MAPPING = {
    # 中医调理成本（月）
    'traditional_chinese_medicine_1': 30,  # 1级调理
    'traditional_chinese_medicine_2': 80,  # 2级调理
    'traditional_chinese_medicine_3': 130,  # 3级调理
    # 活动干预成本（单次）
    'activity_intervention_1': 3,  # 1级强度
    'activity_intervention_2': 5,  # 2级强度
    'activity_intervention_3': 8   # 3级强度
}

# 阈值相关（附表3）
THRESHOLDS = {
    # 风险等级阈值
    'risk_probability': 0.35,  # ROC曲线约登指数确定的最优阈值
    # 高风险规则
    'high_risk': {
        'lipid_abnormality_count': 2,  # 血脂异常项数≥2
        'lipid_abnormality_count_1痰湿': 62,  # 血脂异常项数=1且痰湿积分≥62
        'normal_lipid_痰湿': 80,  # 血脂正常且痰湿积分≥80
        'normal_lipid_活动': 40  # 血脂正常且活动总分<40
    },
    # 低风险规则
    'low_risk': {
        '痰湿积分': 60,  # 痰湿积分<60
        '活动总分': 40,  # 活动总分≥40
        'risk_probability': 0.35  # 模型预测概率<0.35
    }
}

# 年龄约束
AGE_CONSTRAINTS = {
    'min_age': 18,  # 最小年龄
    'max_age': 65   # 最大年龄
}

# 分层变量
STRATIFICATION = {
    # 活动能力分层
    'activity_level': {
        'low': 40,    # <40
        'medium': 59, # 40-59
        'high': 60    # ≥60
    },
    # 痰湿积分分层
    'phlegm_dampness': {
        'low': 58,    # ≤58
        'medium': 61, # 59-61
        'high': 62    # ≥62
    }
}

# 干预参数
INTERVENTION_PARAMS = {
    # 中医调理降幅（月）
    'tcm_reduction': {
        'level_1': 0.01,  # 1级调理：1%
        'level_2': 0.02,  # 2级调理：2%
        'level_3': 0.03   # 3级调理：3%
    },
    # 活动干预参数
    'activity_params': {
        'min_frequency': 5,  # 每周≥5次起效
        'intensity_reduction': 0.03,  # 每升1级强度月增3%降幅
        'frequency_reduction': 0.01   # 每增1次频次月增1%降幅
    },
    # 约束
    'constraints': {
        'max_monthly_reduction': 0.20,  # 月总降幅≤20%
        'max_total_cost': 2000,  # 单人6个月总成本≤2000元
        'max_months': 6  # 干预周期6个月
    }
}

# 其他常量
DEFAULT_TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42