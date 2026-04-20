#!/usr/bin/env python3
# 合并敏感性分析结果并美化图表

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.three_layer_architecture import TripleLayerPredictor, create_tcm_interactions

# 设置中文字体和美化参数
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.facecolor"] = "#f8f9fa"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["legend.framealpha"] = 1.0
plt.rcParams["legend.edgecolor"] = "#333"
plt.rcParams["legend.fontsize"] = 10

class MergedSensitivityAnalyzer:
    """合并敏感性分析类"""
    
    def __init__(self):
        self.predictor = None
        self.data = None
    
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
        print("基础模型训练完成")
    
    def get_probability_threshold_results(self):
        """获取概率阈值敏感性分析结果"""
        print("\n=== 概率阈值敏感性分析 ===")
        
        # 测试不同的概率阈值
        thresholds = np.linspace(0.1, 0.8, 15)
        results = []
        
        for threshold in thresholds:
            # 重新预测（使用修改后的阈值）
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
        
        return pd.DataFrame(results)
    
    def get_data_noise_results(self):
        """获取数据噪声敏感性分析结果"""
        print("\n=== 数据噪声敏感性分析 ===")
        
        # 测试不同程度的噪声
        noise_levels = [0, 0.01, 0.05, 0.1, 0.15, 0.2]
        results = []
        
        # 先获取基础模型结果
        base_results = self.predictor.predict(self.data)
        base_risk = base_results['最终风险等级']
        
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
            noisy_risk = noisy_results['最终风险等级']
            consistency = (base_risk == noisy_risk).mean()
            
            results.append({
                '噪声水平': noise,
                '一致性': consistency
            })
            
            print(f"噪声水平 {noise:.2f}: 一致性 {consistency:.4f}")
        
        return pd.DataFrame(results)
    
    def create_merged_chart(self):
        """创建合并的敏感性分析图表"""
        print("\n=== 创建合并敏感性分析图表 ===")
        
        # 获取分析结果
        threshold_df = self.get_probability_threshold_results()
        noise_df = self.get_data_noise_results()
        
        # 创建合并图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # 图表1：概率阈值敏感性分析
        ax1.plot(threshold_df['阈值'], threshold_df['低风险比例'], 'g-', label='低风险', linewidth=2, marker='o', markersize=6)
        ax1.plot(threshold_df['阈值'], threshold_df['中风险比例'], 'y-', label='中风险', linewidth=2, marker='s', markersize=6)
        ax1.plot(threshold_df['阈值'], threshold_df['高风险比例'], 'r-', label='高风险', linewidth=2, marker='^', markersize=6)
        ax1.set_xlabel('高风险概率阈值', fontsize=14, fontweight='bold')
        ax1.set_ylabel('比例', fontsize=14, fontweight='bold')
        ax1.set_title('概率阈值敏感性分析', fontsize=16, fontweight='bold')
        ax1.legend(loc='best', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 添加阈值参考线
        ax1.axvline(x=0.6, color='gray', linestyle='--', alpha=0.7, label='当前阈值')
        ax1.text(0.61, 0.95, '当前高风险阈值', transform=ax1.transData, rotation=90, verticalalignment='top', fontsize=10)
        
        # 图表2：数据噪声敏感性分析
        ax2.plot(noise_df['噪声水平'], noise_df['一致性'], 'b-', linewidth=3, marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax2.set_xlabel('噪声水平', fontsize=14, fontweight='bold')
        ax2.set_ylabel('一致性', fontsize=14, fontweight='bold')
        ax2.set_title('数据噪声敏感性分析', fontsize=16, fontweight='bold')
        ax2.set_ylim(0.98, 1.001)
        ax2.grid(True, alpha=0.3)
        
        # 添加噪声水平标签
        for i, row in noise_df.iterrows():
            ax2.text(row['噪声水平'] + 0.005, row['一致性'] - 0.001, f'{row["一致性"]:.4f}', fontsize=10)
        
        # 整体标题
        fig.suptitle('模型敏感性分析', fontsize=18, fontweight='bold', y=1.02)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        output_path = 'sensitivity_analysis_merged.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"合并敏感性分析图表已保存到: {output_path}")
        
        return output_path

if __name__ == '__main__':
    analyzer = MergedSensitivityAnalyzer()
    analyzer.load_data()
    analyzer.train_base_model()
    analyzer.create_merged_chart()
