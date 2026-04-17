# 三级融合预警模型模块
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from config.constants import THRESHOLDS, DEFAULT_TRAIN_TEST_SPLIT, RANDOM_SEED
from src.02_feature_engineering import handle_class_imbalance

def focal_loss(labels, logits, gamma=2.0, alpha=0.25):
    """Focal Loss损失函数"""
    epsilon = 1e-7
    logits = np.clip(logits, epsilon, 1.0 - epsilon)
    p_t = logits * labels + (1 - logits) * (1 - labels)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    loss = -alpha_t * np.power(1 - p_t, gamma) * np.log(p_t)
    return loss.mean()

def train_base_models(X_train, y_train):
    """训练基础模型"""
    # 处理类别不平衡
    X_res, y_res = handle_class_imbalance(X_train, y_train)
    
    # 模型1：随机森林
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf_model.fit(X_res, y_res)
    
    # 模型2：梯度提升树
    gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
    gbm_model.fit(X_res, y_res)
    
    # 模型3：逻辑回归
    lr_model = LogisticRegression(random_state=RANDOM_SEED)
    lr_model.fit(X_res, y_res)
    
    return rf_model, gbm_model, lr_model

def ensemble_predict(models, X):
    """融合预测"""
    # 获取各模型的预测概率
    rf_proba = models[0].predict_proba(X)[:, 1]
    gbm_proba = models[1].predict_proba(X)[:, 1]
    lr_proba = models[2].predict_proba(X)[:, 1]
    
    # 融合概率（简单平均）
    ensemble_proba = (rf_proba + gbm_proba + lr_proba) / 3
    
    return ensemble_proba

def classify_risk_level(prob, lipid_abnormality_count=None, phlegm_score=None, activity_score=None):
    """根据概率和规则划分风险等级"""
    # 高风险规则
    if lipid_abnormality_count is not None:
        if lipid_abnormality_count >= THRESHOLDS['high_risk']['lipid_abnormality_count']:
            return 3
        elif lipid_abnormality_count == 1 and phlegm_score is not None and phlegm_score >= THRESHOLDS['high_risk']['lipid_abnormality_count_1痰湿']:
            return 3
        elif lipid_abnormality_count == 0 and phlegm_score is not None and activity_score is not None:
            if phlegm_score >= THRESHOLDS['high_risk']['normal_lipid_痰湿'] and activity_score < THRESHOLDS['high_risk']['normal_lipid_活动']:
                return 3
    
    # 低风险规则
    if lipid_abnormality_count == 0 and phlegm_score is not None and activity_score is not None:
        if phlegm_score < THRESHOLDS['low_risk']['痰湿积分'] and activity_score >= THRESHOLDS['low_risk']['活动总分'] and prob < THRESHOLDS['low_risk']['risk_probability']:
            return 1
    
    # 中风险
    return 2

def train_risk_model(input_path, model_output_path, target):
    """完整风险模型训练流程"""
    # 加载特征工程后的数据
    df = pd.read_pickle(input_path)
    
    # 准备特征和目标
    features = [col for col in df.columns if col != target]
    X = df[features]
    y = df[target]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-DEFAULT_TRAIN_TEST_SPLIT, random_state=RANDOM_SEED)
    
    # 训练基础模型
    models = train_base_models(X_train, y_train)
    
    # 融合预测
    y_test_proba = ensemble_predict(models, X_test)
    
    # 评估模型
    y_test_pred = [1 if p >= THRESHOLDS['risk_probability'] else 0 for p in y_test_proba]
    precision = precision_score(y_test, y_test_pred, average='macro')
    recall = recall_score(y_test, y_test_pred, average='macro')
    f1 = f1_score(y_test, y_test_pred, average='macro')
    auc_roc = roc_auc_score(y_test, y_test_proba)
    pr_auc = average_precision_score(y_test, y_test_proba)
    
    print(f"模型评估指标：")
    print(f"Macro-F1: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"PR曲线下面积: {pr_auc:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"精确率: {precision:.4f}")
    
    # 保存模型（实际项目中可使用joblib或pickle）
    # import joblib
    # joblib.dump(models, model_output_path)
    
    return models, (f1, auc_roc, pr_auc, recall, precision)