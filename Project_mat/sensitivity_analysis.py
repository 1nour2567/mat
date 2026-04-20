#!/usr/bin/env python3
# 问题二模型敏感性分析

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.three_layer_architecture import TripleLayerPredictor, create_tcm_interactions

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150

class SensitivityAnalyzer:
    """模型敏感性分析类"""
    
    def __init__(self):
        self.predictor = None
        self.data = None
        self.base_results = None
    
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        self.data = pd.read_pickle('data/processed/preprocessed_data.pkl')
        print(f"数据加载成功，形状: {self.data.shape}")
    
    def train_base_model(self):
        """训练基础模型"""
        print("训练基础模型...")
        self.predictor = TripleLayerPredictor()
        self.predictor.fit(self.data, target_col='高血脂症二分类标签')
        self.base_results = self.predictor.predict(self.data)
        print("基础模型训练完成")
    
    def analyze_probability_thresholds(self):
        """分析概率阈值敏感性"""
        print("\n=== 概率阈值敏感性分析 ===")
        
        # 测试不同的概率阈值
        thresholds = np.linspace(0.1, 0.8, 15)
        results = []
        
        for threshold in thresholds:
            # 重新预测（使用修改后的阈值）
            # 由于TCMFunctionalLayer的阈值是硬编码的，我们需要创建一个临时的预测函数
            def apply_tcm_rules_with_thresholds(df, predicted_probs):
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
                    
                    # --- 第二层：临床规则层 (西医金标准) ---
                    n_i = row['血脂异常项数']
                    if n_i >= 1:
                        risk_levels.append("临床确诊高风险")
                        continue
                    
                    # --- 第一层：统计模型层 (潜在风险概率) ---
                    # 得到各折模型的平均预测概率 p_hat
                    
                    # --- 第三层：中医功能层 (边界修正逻辑) ---
                    # 设置初步等级
                    medium_threshold = threshold * 0.5
                    if p_hat >= threshold:
                        final_risk = "高风险"
                    elif p_hat < medium_threshold:
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
                    elif final_risk == "中风险" and (tcm_tan_shi < 17 and activity_score >= 60):
                        final_risk = "低风险(中医支持)"
                    
                    risk_levels.append(final_risk)
                
                df['最终风险等级'] = risk_levels
                return df
            
            # 执行预测流程
            df_result = self.data.copy()
            df_result = create_tcm_interactions(df_result)
            df_result, clinical_high_risk = self.predictor.clinical_layer.apply_clinical_rules(df_result)
            predicted_probs = self.predictor.model_layer.predict_probability(df_result)
            df_result['模型预测概率'] = predicted_probs
            current_results = apply_tcm_rules_with_thresholds(df_result, predicted_probs)
            
            risk_distribution = current_results['最终风险等级'].value_counts().to_dict()
            
            # 计算各风险等级比例
            total = len(current_results)
            distribution = {
                '低风险': (risk_distribution.get('低风险', 0) + risk_distribution.get('低风险(中医支持)', 0)) / total,
                '中风险': risk_distribution.get('中风险', 0) / total,
                '高风险': (risk_distribution.get('临床确诊高风险', 0) + risk_distribution.get('高风险', 0) + risk_distribution.get('高风险(中医预警)', 0)) / total
            }
            
            results.append({
                '阈值': threshold,
                '低风险比例': distribution['低风险'],
                '中风险比例': distribution['中风险'],
                '高风险比例': distribution['高风险']
            })
            
            print(f"高风险阈值={threshold:.2f}, 中风险阈值={threshold*0.5:.2f}: 低风险比例={distribution['低风险']:.4f}, 中风险比例={distribution['中风险']:.4f}, 高风险比例={distribution['高风险']:.4f}")
        
        # 转换为DataFrame并绘制图表
        df_thresholds = pd.DataFrame(results)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df_thresholds['阈值'], df_thresholds['低风险比例'], 'g-', label='低风险', linewidth=2)
        plt.plot(df_thresholds['阈值'], df_thresholds['中风险比例'], 'y-', label='中风险', linewidth=2)
        plt.plot(df_thresholds['阈值'], df_thresholds['高风险比例'], 'r-', label='高风险', linewidth=2)
        plt.xlabel('高风险概率阈值', fontsize=12, fontweight='bold')
        plt.ylabel('比例', fontsize=12, fontweight='bold')
        plt.title('概率阈值敏感性分析', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('sensitivity_analysis_thresholds.png', dpi=200, bbox_inches='tight')
        print("概率阈值敏感性分析完成，结果已保存")
        
        return df_thresholds
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("\n=== 特征重要性分析 ===")
        
        # 提取特征重要性
        all_importances = np.array([m.feature_importances_ for m in self.predictor.model_layer.models])
        mean_imp = all_importances.mean(axis=0)
        feature_names = self.predictor.model_layer.feature_names
        
        # 排序
        idx = np.argsort(mean_imp)[-15:]
        top_features = [feature_names[i] for i in idx][::-1]
        top_importances = mean_imp[idx][::-1]
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 6))
        bars = plt.barh(top_features, top_importances, color='steelblue')
        plt.xlabel('Feature Importance (Mean Gain)', fontsize=12, fontweight='bold')
        plt.title('Top 15 Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('sensitivity_analysis_feature_importance.png', dpi=200, bbox_inches='tight')
        print("特征重要性分析完成，结果已保存")
        
        return top_features, top_importances
    
    def analyze_tcm_parameters(self):
        """分析中医参数敏感性"""
        print("\n=== 中医参数敏感性分析 ===")
        
        # 测试不同的痰湿质阈值
        tan_shi_thresholds = np.linspace(50, 80, 7)
        activity_thresholds = np.linspace(30, 50, 5)
        
        results = []
        
        for tan_shi_thresh in tan_shi_thresholds:
            for activity_thresh in activity_thresholds:
                # 重新预测（使用修改后的阈值）
                # 由于TCMFunctionalLayer的阈值是硬编码的，我们需要创建一个临时的预测函数
                def apply_tcm_rules_with_thresholds(df, predicted_probs):
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
                        
                        # --- 第二层：临床规则层 (西医金标准) ---
                        n_i = row['血脂异常项数']
                        if n_i >= 1:
                            risk_levels.append("临床确诊高风险")
                            continue
                        
                        # --- 第一层：统计模型层 (潜在风险概率) ---
                        # 得到各折模型的平均预测概率 p_hat
                        
                        # --- 第三层：中医功能层 (边界修正逻辑) ---
                        # 设置初步等级
                        if p_hat >= 0.60:
                            final_risk = "高风险"
                        elif p_hat < 0.20:
                            final_risk = "低风险"
                        else:
                            final_risk = "中风险"
                        
                        # 触发专家规则干预 (仅对中风险及临界区)
                        tcm_tan_shi = row['痰湿质']
                        activity_score = row['活动量表总分（ADL总分+IADL总分）']
                        
                        # 【升档逻辑】邪盛正衰：痰湿极重且不动，即便模型概率不高也强行升至高风险
                        if final_risk == "中风险" and (tcm_tan_shi >= tan_shi_thresh and activity_score < activity_thresh):
                            final_risk = "高风险(中医预警)"
                        
                        # 【降档逻辑】正盛邪微：痰湿轻且运动极强
                        elif final_risk == "中风险" and (tcm_tan_shi < 17 and activity_score >= 60):
                            final_risk = "低风险(中医支持)"
                        
                        risk_levels.append(final_risk)
                    
                    df['最终风险等级'] = risk_levels
                    return df
                
                # 执行预测流程
                df_result = self.data.copy()
                df_result = create_tcm_interactions(df_result)
                df_result, clinical_high_risk = self.predictor.clinical_layer.apply_clinical_rules(df_result)
                predicted_probs = self.predictor.model_layer.predict_probability(df_result)
                df_result['模型预测概率'] = predicted_probs
                current_results = apply_tcm_rules_with_thresholds(df_result, predicted_probs)
                
                risk_distribution = current_results['最终风险等级'].value_counts().to_dict()
                
                # 计算各风险等级比例
                total = len(current_results)
                high_risk_count = risk_distribution.get('临床确诊高风险', 0) + risk_distribution.get('高风险', 0) + risk_distribution.get('高风险(中医预警)', 0)
                high_risk_ratio = high_risk_count / total
                
                results.append({
                    '痰湿质阈值': tan_shi_thresh,
                    '活动能力阈值': activity_thresh,
                    '高风险比例': high_risk_ratio
                })
                
                print(f"痰湿质阈值={tan_shi_thresh:.0f}, 活动能力阈值={activity_thresh:.0f}: 高风险比例={high_risk_ratio:.4f}")
        
        # 转换为DataFrame并绘制热图
        df_tcm = pd.DataFrame(results)
        pivot_df = df_tcm.pivot(index='痰湿质阈值', columns='活动能力阈值', values='高风险比例')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.xlabel('活动能力阈值', fontsize=12, fontweight='bold')
        plt.ylabel('痰湿质阈值', fontsize=12, fontweight='bold')
        plt.title('中医参数敏感性分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sensitivity_analysis_tcm_parameters.png', dpi=200, bbox_inches='tight')
        print("中医参数敏感性分析完成，结果已保存")
        
        return pivot_df
    
    def analyze_data_noise(self):
        """分析数据噪声敏感性"""
        print("\n=== 数据噪声敏感性分析 ===")
        
        # 测试不同程度的噪声
        noise_levels = [0, 0.01, 0.05, 0.1, 0.15, 0.2]
        results = []
        
        for noise in noise_levels:
            # 创建带噪声的数据
            noisy_data = self.data.copy()
            
            # 对数值特征添加噪声
            numeric_features = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', 
                              '血瘀质', '气郁质', '特禀质', 'ADL总分', 'IADL总分', 
                              '活动量表总分（ADL总分+IADL总分）', '空腹血糖', '血尿酸', 'BMI']
            
            for feature in numeric_features:
                if feature in noisy_data.columns:
                    std = noisy_data[feature].std()
                    noise_values = np.random.normal(0, std * noise, len(noisy_data))
                    noisy_data[feature] = noisy_data[feature] + noise_values
                    # 确保值在合理范围内
                    if feature.endswith('质'):
                        noisy_data[feature] = np.clip(noisy_data[feature], 0, 100)
                    elif feature in ['ADL总分', 'IADL总分']:
                        noisy_data[feature] = np.clip(noisy_data[feature], 0, 50)
                    elif feature == '活动量表总分（ADL总分+IADL总分）':
                        noisy_data[feature] = np.clip(noisy_data[feature], 0, 100)
            
            # 重新训练模型并预测
            noisy_predictor = TripleLayerPredictor()
            noisy_predictor.fit(noisy_data, target_col='高血脂症二分类标签')
            noisy_results = noisy_predictor.predict(noisy_data)
            
            # 计算与基础模型的一致性
            base_risk = self.base_results['最终风险等级']
            noisy_risk = noisy_results['最终风险等级']
            consistency = (base_risk == noisy_risk).mean()
            
            results.append({
                '噪声水平': noise,
                '一致性': consistency
            })
            
            print(f"噪声水平 {noise:.2f}: 一致性 {consistency:.4f}")
        
        # 绘制噪声敏感性图
        df_noise = pd.DataFrame(results)
        plt.figure(figsize=(10, 6))
        plt.plot(df_noise['噪声水平'], df_noise['一致性'], 'b-o', linewidth=2, markersize=8)
        plt.xlabel('噪声水平', fontsize=12, fontweight='bold')
        plt.ylabel('一致性', fontsize=12, fontweight='bold')
        plt.title('数据噪声敏感性分析', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('sensitivity_analysis_noise.png', dpi=200, bbox_inches='tight')
        print("数据噪声敏感性分析完成，结果已保存")
        
        return df_noise
    
    def run_full_analysis(self):
        """运行完整的敏感性分析"""
        print("=" * 80)
        print("问题二模型敏感性分析")
        print("=" * 80)
        
        # 加载数据
        self.load_data()
        
        # 训练基础模型
        self.train_base_model()
        
        # 分析概率阈值敏感性
        threshold_results = self.analyze_probability_thresholds()
        
        # 分析特征重要性
        top_features, top_importances = self.analyze_feature_importance()
        
        # 分析中医参数敏感性
        tcm_results = self.analyze_tcm_parameters()
        
        # 分析数据噪声敏感性
        noise_results = self.analyze_data_noise()
        
        print("\n" + "=" * 80)
        print("敏感性分析完成")
        print("=" * 80)
        
        return {
            'threshold_results': threshold_results,
            'top_features': top_features,
            'top_importances': top_importances,
            'tcm_results': tcm_results,
            'noise_results': noise_results
        }

if __name__ == '__main__':
    analyzer = SensitivityAnalyzer()
    results = analyzer.run_full_analysis()
