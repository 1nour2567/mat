# 三层融合预警模型架构
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from config.constants import THRESHOLDS, RANDOM_SEED


class ClinicalRuleLayer:
    """第一层：临床规则层 (Clinical Rule Layer)"""
    
    def __init__(self):
        self.name = "临床规则层"
    
    @staticmethod
    def calculate_lipid_abnormality_count(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算血脂异常项数
        
        Args:
            df: 输入数据框
            
        Returns:
            添加血脂异常项数的数据框
        """
        df = df.copy()
        
        # 血脂异常判定（根据标准定义）
        if 'TC（总胆固醇）' in df.columns:
            df['TC异常'] = ((df['TC（总胆固醇）'] < 3.1) | (df['TC（总胆固醇）'] > 6.2)).astype(int)
        
        if 'TG（甘油三酯）' in df.columns:
            df['TG异常'] = ((df['TG（甘油三酯）'] < 0.56) | (df['TG（甘油三酯）'] > 1.7)).astype(int)
        
        if 'LDL-C（低密度脂蛋白）' in df.columns:
            df['LDL-C异常'] = ((df['LDL-C（低密度脂蛋白）'] < 2.07) | (df['LDL-C（低密度脂蛋白）'] > 3.1)).astype(int)
        
        if 'HDL-C（高密度脂蛋白）' in df.columns:
            df['HDL-C异常'] = ((df['HDL-C（高密度脂蛋白）'] < 1.04) | (df['HDL-C（高密度脂蛋白）'] > 1.55)).astype(int)
        
        # 计算血脂异常项数
        lipid_abnormalities = ['TC异常', 'TG异常', 'LDL-C异常', 'HDL-C异常']
        existing_lipid_abnormalities = [col for col in lipid_abnormalities if col in df.columns]
        
        if existing_lipid_abnormalities:
            df['血脂异常项数'] = df[existing_lipid_abnormalities].sum(axis=1)
        else:
            df['血脂异常项数'] = 0
        
        return df
    
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
        
        # 确保有血脂异常项数
        if '血脂异常项数' not in df.columns:
            df = ClinicalRuleLayer.calculate_lipid_abnormality_count(df)
        
        # 临床规则：血脂异常项数 ≥ 1 判定为临床确诊高风险
        clinical_high_risk = (df['血脂异常项数'] >= 1).astype(int).values
        
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
        self.feature_names = None
    
    @staticmethod
    def get_allowed_features(df: pd.DataFrame) -> list:
        """
        获取允许使用的特征（屏蔽血脂相关指标）
        
        Args:
            df: 输入数据框
            
        Returns:
            允许使用的特征列表
        """
        all_cols = df.columns.tolist()
        
        # 严格屏蔽的特征
        blocked_features = [
            # 原始四项血脂指标
            'TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）',
            # 血脂异常相关指标
            '血脂异常项数', 'TC异常', 'TG异常', 'LDL-C异常', 'HDL-C异常',
            # 血脂派生指标
            'AIP', 'TC/HDL比值', 'LDL/HDL比值', 'TG/HDL比值', 'non-HDL-C',
            # 带缩尾的血脂派生指标
            'non-HDL-C_缩尾', 'AIP_缩尾', 'TC/HDL比值_缩尾', 
            'LDL/HDL比值_缩尾', 'TG/HDL比值_缩尾',
            # 标签和输出
            '高血脂症二分类标签', '体质标签', '临床确诊高风险',
            # 其他不需要的
            '尿酸异常', '活动能力分层', '痰湿积分分层'
        ]
        
        # 允许使用的特征
        allowed_features = [col for col in all_cols if col not in blocked_features]
        
        return allowed_features
    
    @staticmethod
    def focal_loss_lgb(y_true, y_pred):
        """
        LightGBM Focal Loss 实现
        """
        gamma = 2.0
        alpha = 0.25
        
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        loss = -alpha_t * np.power(1 - p_t, gamma) * np.log(p_t)
        
        grad = -alpha_t * np.power(1 - p_t, gamma) * (1 - y_true - y_pred + 2 * y_true * y_pred)
        hess = alpha_t * (np.power(1 - p_t, gamma - 1) * (gamma * p_t * (1 - p_t) + np.power(1 - p_t, 2)))
        
        return loss, grad, hess
    
    def train(self, df: pd.DataFrame, target_col: str = '高血脂症二分类标签'):
        """
        训练LightGBM模型（使用5折交叉验证）
        
        Args:
            df: 输入数据框
            target_col: 目标变量列名
        """
        # 获取允许使用的特征
        allowed_features = self.get_allowed_features(df)
        self.feature_names = allowed_features
        
        X = df[allowed_features].values
        y = df[target_col].values
        
        # 5折交叉验证
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_SEED)
        
        self.models = []
        oof_preds = np.zeros(len(df))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # LightGBM参数
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 6,
                'num_leaves': 31,
                'verbose': -1,
                'random_state': RANDOM_SEED
            }
            
            # 使用自定义Focal Loss（可选，这里为简化使用标准loss）
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            self.models.append(model)
            
            # 验证集预测
            val_pred = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_pred
            
            # 打印验证集AUC
            val_auc = roc_auc_score(y_val, val_pred)
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
        
        # 获取允许使用的特征
        allowed_features = self.feature_names
        X = df[allowed_features].values
        
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
        self.uncertainty_low = 0.35
        self.uncertainty_high = 0.65
    
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
        
        # 初始化风险等级
        risk_levels = []
        
        for i in range(len(df)):
            p_hat = predicted_probs[i]
            phlegm_score = df['痰湿质'].iloc[i]
            activity_score = df['活动量表总分（ADL总分+IADL总分）'].iloc[i]
            
            # 判断是否在不确定区间 [0.35, 0.65]
            if self.uncertainty_low <= p_hat <= self.uncertainty_high:
                # 【升档规则 - 邪盛正衰】
                if phlegm_score >= 80 and activity_score < 40:
                    risk_level = "高风险"  # 中医预警高风险
                # 【降档规则 - 正盛邪微】
                elif phlegm_score < 60 and activity_score >= 60:
                    risk_level = "低风险"
                else:
                    # 维持原始概率对应的等级
                    risk_level = self._prob_to_level(p_hat)
            else:
                # 模型确定区，信任模型
                risk_level = self._prob_to_level(p_hat)
            
            risk_levels.append(risk_level)
        
        df['最终风险等级'] = risk_levels
        
        # 如果有临床确诊高风险，标记为特殊类型
        if '临床确诊高风险' in df.columns:
            df['最终风险等级'] = df.apply(
                lambda row: "临床确诊高风险" if row['临床确诊高风险'] == 1 and row['最终风险等级'] == "高风险" 
                else row['最终风险等级'],
                axis=1
            )
        
        return df
    
    @staticmethod
    def _prob_to_level(prob: float) -> str:
        """
        概率转等级
        
        Args:
            prob: 预测概率
            
        Returns:
            风险等级字符串
        """
        if prob < 0.35:
            return "低风险"
        elif prob < 0.65:
            return "中风险"
        else:
            return "高风险"


# 整合三层架构
class ThreeLayerRiskPredictor:
    """三层整合风险预测器"""
    
    def __init__(self):
        self.clinical_layer = ClinicalRuleLayer()
        self.model_layer = LightGBMPredictionLayer()
        self.tcm_layer = TCMFunctionalLayer()
        self.is_trained = False
    
    def fit(self, df: pd.DataFrame, target_col: str = '高血脂症二分类标签'):
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
