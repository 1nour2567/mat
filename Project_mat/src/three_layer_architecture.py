# 三层融合预警模型架构
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from config.constants import THRESHOLDS, RANDOM_SEED

# 定义特征列表（构建隔离墙）
# 定义血脂屏蔽清单 (禁止进入模型训练)
LIPID_FEATURES = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 'TC（总胆固醇）', 'AIP', 'TC/HDL比值', 'non-HDL-C'] 

# 定义模型可用特征 (中西医融合表型)
MODEL_FEATURES = [
    '平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质',
    'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）', '年龄组', '性别', '吸烟史', '饮酒史',
    '空腹血糖', '血尿酸', 'BMI'
]

# 标签
TARGET = '高血脂症二分类标签'


class ClinicalRuleLayer:
    """第一层：临床规则层 (Clinical Rule Layer)"""
    
    def __init__(self):
        self.name = "临床规则层"
    
    @staticmethod
    def apply_clinical_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        应用临床规则，识别临床确诊高风险
        
        Args:
            df: 输入数据框
            
        Returns:
            (更新后的数据框, 临床高风险标记数组)
        """
        df = df.copy()
        
        # 由于移除了血脂特征，临床规则层无法使用血脂异常项数
        # 设置血脂异常项数为0
        df['血脂异常项数'] = 0
        
        # 临床规则：由于没有血脂数据，无法判定临床确诊高风险
        clinical_high_risk = np.zeros(len(df), dtype=int)
        
        # 添加标记列
        df['临床确诊高风险'] = clinical_high_risk
        
        return df, clinical_high_risk


# 第二层：统计模型层 (LightGBM Prediction Layer)
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class LightGBMPredictionLayer:
    """第二层：统计模型层 (LightGBM Prediction Layer)"""
    
    def __init__(self, n_splits=5):
        self.name = "LightGBM预测层"
        self.n_splits = n_splits
        self.models = []
        self.feature_names = MODEL_FEATURES
    
    def train(self, df: pd.DataFrame, target_col: str = TARGET):
        """
        训练LightGBM模型（使用5折交叉验证）
        
        Args:
            df: 输入数据框
            target_col: 目标变量列名
        """
        # 使用模型可用特征
        X = df[self.feature_names]
        y = df[target_col]
        
        # 5折交叉验证
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        self.models = []
        oof_preds = np.zeros(len(df))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            train_x, val_x = X.iloc[train_idx], X.iloc[val_idx]
            train_y, val_y = y.iloc[train_idx], y.iloc[val_idx]
            
            # 处理类别不平衡：使用 is_unbalance=True 或设置 scale_pos_weight
            model = lgb.LGBMClassifier(
                objective='binary',
                metric='auc',
                is_unbalance=True, 
                learning_rate=0.05,
                n_estimators=1000
            )
            model.fit(train_x, train_y)
            
            self.models.append(model)
            
            # 验证集预测
            val_pred = model.predict_proba(val_x)[:, 1]
            oof_preds[val_idx] = val_pred
            
            # 打印验证集AUC
            val_auc = roc_auc_score(val_y, val_pred)
            print(f"Fold {fold + 1} Validation AUC: {val_auc:.4f}")
        
        # 打印整体OOF AUC
        overall_auc = roc_auc_score(y, oof_preds)
        print(f"Overall OOF AUC: {overall_auc:.4f}")
        
        return self
    
    def predict_probability(self, df: pd.DataFrame) -> np.ndarray:
        """
        预测风险概率
        
        Args:
            df: 输入数据框
            
        Returns:
            预测概率数组
        """
        if not self.models:
            raise ValueError("模型未训练，请先调用train()方法")
        
        # 使用模型可用特征
        X = df[self.feature_names]
        
        # 平均所有折模型的预测
        predictions = np.zeros(len(df))
        for model in self.models:
            predictions += model.predict_proba(X)[:, 1]
        
        predictions /= len(self.models)
        return predictions


# 第三层：中医功能层 (TCM Functional Layer)
class TCMFunctionalLayer:
    """第三层：中医功能层 (TCM Functional Layer)"""
    
    def __init__(self):
        self.name = "中医功能层"
        self.uncertainty_low = 0.25
        self.uncertainty_high = 0.5
    
    def apply_tcm_rules(self, df: pd.DataFrame, predicted_probs: np.ndarray) -> pd.DataFrame:
        """
        应用中医规则修正
        
        Args:
            df: 输入数据框
            predicted_probs: 第二层模型输出的预测概率
            
        Returns:
            添加最终风险等级的数据框
        """
        df = df.copy()
        
        # 确保有必要的列
        required_cols = ['痰湿质', '活动量表总分（ADL总分+IADL总分）']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要列：{col}")
        
        # 初始化风险等级和概率
        risk_levels = []
        
        for i in range(len(df)):
            p_hat = predicted_probs[i]
            row = df.iloc[i]
            
            # --- 第一层：临床规则层 (西医金标准) ---
            n_i = row['血脂异常项数']
            if n_i >= 1:
                risk_levels.append("临床确诊高风险")
                continue
            
            # --- 第二层：统计模型层 (潜在风险概率) ---
            # 得到各折模型的平均预测概率 p_hat
            
            # --- 第三层：中医功能层 (边界修正逻辑) ---
            # 设置初步等级
            if p_hat >= 0.5:
                final_risk = "高风险"
            elif p_hat < 0.25:
                final_risk = "低风险"
            else:
                final_risk = "中风险"
            
            # 触发专家规则干预 (仅对中风险及临界区)
            tcm_tan_shi = row['痰湿质']
            activity_score = row['活动量表总分（ADL总分+IADL总分）']
            
            # 【升档逻辑】邪盛正衰：痰湿极重且不动，即便模型概率不高也强行升至高风险
            if final_risk == "中风险" and (tcm_tan_shi >= 60 and activity_score < 40):
                final_risk = "高风险(中医预警)"
            
            # 【降档逻辑】正盛邪微：痰湿轻且运动极强
            elif final_risk == "中风险" and (tcm_tan_shi < 60 and activity_score >= 60):
                final_risk = "低风险(中医支持)"
            
            risk_levels.append(final_risk)
        
        df['最终风险等级'] = risk_levels
        
        return df


# 整合三层架构
class TripleLayerPredictor:
    """三层整合风险预测器"""
    
    def __init__(self):
        self.clinical_layer = ClinicalRuleLayer()
        self.model_layer = LightGBMPredictionLayer()
        self.tcm_layer = TCMFunctionalLayer()
        self.is_trained = False
    
    def fit(self, df: pd.DataFrame, target_col: str = TARGET):
        """
        训练完整流程
        
        Args:
            df: 输入数据框
            target_col: 目标变量列名
        """
        # 第一层：临床规则层（不影响训练，仅用于评估）
        df_processed, clinical_high_risk = self.clinical_layer.apply_clinical_rules(df)
        
        # 第二层：统计模型层训练
        self.model_layer.train(df_processed, target_col)
        
        self.is_trained = True
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预测完整流程
        
        Args:
            df: 输入数据框
            
        Returns:
            包含预测结果的数据框
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        df_result = df.copy()
        
        # 第一层：临床规则层
        df_result, clinical_high_risk = self.clinical_layer.apply_clinical_rules(df_result)
        
        # 第二层：统计模型层预测
        predicted_probs = self.model_layer.predict_probability(df_result)
        df_result['模型预测概率'] = predicted_probs
        
        # 第三层：中医功能层修正
        df_result = self.tcm_layer.apply_tcm_rules(df_result, predicted_probs)
        
        return df_result

    def predict_instance(self, row):
        """
        预测单个实例
        
        Args:
            row: 数据行
            
        Returns:
            (风险等级, 预测概率)
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        # --- 第一层：临床规则层 (西医金标准) ---
        # 由于移除了血脂特征，无法计算血脂异常项数
        n_i = 0
        if n_i >= 1:
            return "临床确诊高风险", 1.0
        
        # --- 第二层：统计模型层 (潜在风险概率) ---
        input_data = row[MODEL_FEATURES].values.reshape(1, -1)
        p_hat = np.mean([m.predict_proba(input_data)[0][1] for m in self.model_layer.models])
        
        # --- 第三层：中医功能层 (边界修正逻辑) ---
        # 设置初步等级
        if p_hat >= 0.5:
            final_risk = "高风险"
        elif p_hat < 0.25:
            final_risk = "低风险"
        else:
            final_risk = "中风险"
        
        # 触发专家规则干预 (仅对中风险及临界区)
        tcm_tan_shi = row['痰湿质']
        activity_score = row['活动量表总分（ADL总分+IADL总分）']
        
        # 【升档逻辑】邪盛正衰：痰湿极重且不动，即便模型概率不高也强行升至高风险
        if final_risk == "中风险" and (tcm_tan_shi >= 60 and activity_score < 40):
            final_risk = "高风险(中医预警)"
        
        # 【降档逻辑】正盛邪微：痰湿轻且运动极强
        elif final_risk == "中风险" and (tcm_tan_shi < 60 and activity_score >= 60):
            final_risk = "低风险(中医支持)"
        
        return final_risk, p_hat
