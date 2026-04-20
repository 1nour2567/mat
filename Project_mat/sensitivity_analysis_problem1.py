#!/usr/bin/env python3
# 问题一模型敏感性分析

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

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

class Problem1SensitivityAnalyzer:
    """问题一模型敏感性分析类"""
    
    def __init__(self):
        self.data = None
        self.blood_indicators = ['TC（总胆固醇）', 'TG（甘油三酯）', 'LDL-C（低密度脂蛋白）', 
                               'HDL-C（高密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI',
                               'non-HDL-C_缩尾', 'AIP_缩尾', 'TC/HDL比值_缩尾', 'LDL/HDL比值_缩尾', 'TG/HDL比值_缩尾']
        self.activity_indicators = ['ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）']
        self.constitution_types = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
        self.target_phlegm = '痰湿质'
        self.target_risk = '高血脂症二分类标签'
    
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        self.data = pd.read_pickle('data/processed/preprocessed_data.pkl')
        print(f"数据加载成功，形状: {self.data.shape}")
        
        # 确保必要的列存在
        required_cols = self.blood_indicators + self.activity_indicators + \
                      self.constitution_types + [self.target_phlegm, self.target_risk]
        for col in required_cols:
            if col not in self.data.columns:
                print(f"警告：缺少列 {col}")
    
    def calculate_spearman_correlation(self, features, target):
        """计算Spearman相关系数"""
        scores = {}
        for feature in features:
            if feature in self.data.columns:
                corr, _ = spearmanr(self.data[feature], self.data[target])
                scores[feature] = abs(corr)
            else:
                scores[feature] = 0
        return scores
    
    def calculate_mutual_info(self, features, target):
        """计算互信息"""
        scores = {}
        valid_features = [f for f in features if f in self.data.columns]
        if valid_features:
            X = self.data[valid_features]
            y = self.data[target]
            mi_scores = mutual_info_classif(X, y, random_state=42)
            for i, feature in enumerate(valid_features):
                scores[feature] = mi_scores[i]
        for feature in features:
            if feature not in scores:
                scores[feature] = 0
        return scores
    
    def calculate_pls_loadings(self, features, target1, target2):
        """计算PLS联合结构载荷"""
        scores = {}
        valid_features = [f for f in features if f in self.data.columns]
        if valid_features and target1 in self.data.columns and target2 in self.data.columns:
            X = self.data[valid_features]
            y1 = self.data[target1]
            y2 = self.data[target2]
            y_combined = np.column_stack((y1, y2))
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pls = PLSRegression(n_components=2, scale=False)
            pls.fit(X_scaled, y_combined)
            
            for i, feature in enumerate(valid_features):
                corr1 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 0])[0, 1]
                corr2 = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, 1])[0, 1]
                scores[feature] = (abs(corr1) + abs(corr2)) / 2
        for feature in features:
            if feature not in scores:
                scores[feature] = 0
        return scores
    
    def entropy_weight_method(self, spearman_scores, mi_scores, pls_scores):
        """熵权法计算权重"""
        # 构建原始矩阵
        features = list(spearman_scores.keys())
        n = len(features)
        m = 3
        
        X_matrix = np.zeros((n, m))
        for i, feature in enumerate(features):
            X_matrix[i, 0] = spearman_scores.get(feature, 0)
            X_matrix[i, 1] = mi_scores.get(feature, 0)
            X_matrix[i, 2] = pls_scores.get(feature, 0)
        
        # 矩阵标准化
        X_norm = np.zeros_like(X_matrix)
        for j in range(m):
            min_val = np.min(X_matrix[:, j])
            max_val = np.max(X_matrix[:, j])
            if max_val - min_val == 0:
                X_norm[:, j] = 0
            else:
                X_norm[:, j] = (X_matrix[:, j] - min_val) / (max_val - min_val)
        X_norm = X_norm + 0.0001
        
        # 计算比重
        P = np.zeros_like(X_norm)
        for j in range(m):
            col_sum = np.sum(X_norm[:, j])
            if col_sum == 0:
                P[:, j] = 0
            else:
                P[:, j] = X_norm[:, j] / col_sum
        
        # 计算信息熵
        e = np.zeros(m)
        k = 1 / np.log(n) if n > 1 else 0
        for j in range(m):
            entropy = 0
            for i in range(n):
                if P[i, j] > 0:
                    entropy += P[i, j] * np.log(P[i, j])
            e[j] = -k * entropy
        
        # 计算信息冗余度
        d = 1 - e
        
        # 计算最终权重
        if np.sum(d) == 0:
            w = np.ones(m) / m
        else:
            w = d / np.sum(d)
        
        return {
            'spearman': w[0],
            'mi': w[1],
            'pls': w[2]
        }
    
    def analyze_key_indicators(self, weights=None):
        """分析关键指标"""
        all_indicators = self.blood_indicators + self.activity_indicators
        
        # 计算各项评分
        spearman_scores = self.calculate_spearman_correlation(all_indicators, self.target_phlegm)
        mi_scores = self.calculate_mutual_info(all_indicators, self.target_risk)
        pls_scores = self.calculate_pls_loadings(all_indicators, self.target_phlegm, self.target_risk)
        
        # 使用熵权法或指定权重
        if weights is None:
            weights = self.entropy_weight_method(spearman_scores, mi_scores, pls_scores)
        
        # 计算综合得分
        combined_scores = {}
        for feature in all_indicators:
            score = (weights['spearman'] * spearman_scores.get(feature, 0) +
                    weights['mi'] * mi_scores.get(feature, 0) +
                    weights['pls'] * pls_scores.get(feature, 0))
            combined_scores[feature] = score
        
        # 排序
        sorted_indicators = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_indicators, weights
    
    def analyze_constitution_contribution(self):
        """分析体质贡献度"""
        from sklearn.linear_model import LogisticRegression
        
        # 控制变量
        control_vars = self.blood_indicators + self.activity_indicators
        valid_control_vars = [var for var in control_vars if var in self.data.columns]
        
        # 构建特征矩阵
        features = self.constitution_types + valid_control_vars
        valid_features = [f for f in features if f in self.data.columns]
        
        if not valid_features:
            return []
        
        X = self.data[valid_features]
        y = self.data[self.target_risk]
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 拟合模型
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, y)
        
        # 提取体质相关的系数
        coefficients = {}
        for feature in self.constitution_types:
            if feature in valid_features:
                idx = valid_features.index(feature)
                coefficients[feature] = abs(model.coef_[0][idx])
            else:
                coefficients[feature] = 0
        
        # 排序
        sorted_constitutions = sorted(coefficients.items(), key=lambda x: x[1], reverse=True)
        return sorted_constitutions
    
    def sensitivity_analysis_weights(self):
        """权重敏感性分析"""
        print("\n=== 权重敏感性分析 ===")
        
        # 测试不同权重组合
        weight_combinations = [
            {'spearman': 0.5, 'mi': 0.3, 'pls': 0.2},
            {'spearman': 0.3, 'mi': 0.5, 'pls': 0.2},
            {'spearman': 0.3, 'mi': 0.3, 'pls': 0.4},
            {'spearman': 0.6, 'mi': 0.2, 'pls': 0.2},
            {'spearman': 0.2, 'mi': 0.6, 'pls': 0.2},
            {'spearman': 0.2, 'mi': 0.2, 'pls': 0.6}
        ]
        
        results = []
        base_result, base_weights = self.analyze_key_indicators()
        base_top10 = [item[0] for item in base_result[:10]]
        
        print(f"基准权重: Spearman={base_weights['spearman']:.4f}, MI={base_weights['mi']:.4f}, PLS={base_weights['pls']:.4f}")
        print(f"基准前10指标: {base_top10}")
        
        for i, weights in enumerate(weight_combinations):
            result, _ = self.analyze_key_indicators(weights)
            top10 = [item[0] for item in result[:10]]
            # 计算与基准的相似度：前10指标的交集比例
            similarity = len(set(base_top10) & set(top10)) / 10
            results.append({
                'weights': weights,
                'top10': top10,
                'similarity': similarity
            })
            print(f"组合{i+1}权重: Spearman={weights['spearman']:.2f}, MI={weights['mi']:.2f}, PLS={weights['pls']:.2f}, 相似度={similarity:.2f}")
        
        return results
    
    def sensitivity_analysis_features(self):
        """特征敏感性分析"""
        print("\n=== 特征敏感性分析 ===")
        
        # 测试删除单个特征的影响
        all_indicators = self.blood_indicators + self.activity_indicators
        base_result, _ = self.analyze_key_indicators()
        base_top10 = [item[0] for item in base_result[:10]]
        base_scores = dict(base_result)
        
        results = []
        for feature in all_indicators:
            if feature in self.data.columns:
                # 临时删除该特征
                original_data = self.data.copy()
                self.data = self.data.drop(columns=[feature])
                
                # 重新分析
                result, _ = self.analyze_key_indicators()
                top10 = [item[0] for item in result[:10]]
                new_scores = dict(result)
                
                # 计算影响
                similarity = len(set(base_top10) & set(top10)) / 10
                score_change = {}
                for indicator in all_indicators:
                    if indicator != feature and indicator in base_scores and indicator in new_scores:
                        score_change[indicator] = new_scores[indicator] - base_scores[indicator]
                
                results.append({
                    'removed_feature': feature,
                    'top10': top10,
                    'similarity': similarity,
                    'score_change': score_change
                })
                
                # 恢复数据
                self.data = original_data
                
                print(f"删除特征 {feature}: 相似度={similarity:.2f}")
        
        # 识别敏感特征
        if results:
            results.sort(key=lambda x: x['similarity'])
            print("\n敏感特征排序（按相似度从低到高）:")
            for i, result in enumerate(results):
                print(f"{i+1}. {result['removed_feature']}: 相似度={result['similarity']:.2f}")
        
        return results
    
    def sensitivity_analysis_gender_age(self):
        """性别和年龄敏感性分析"""
        print("\n=== 性别和年龄敏感性分析 ===")
        
        # 性别分析
        genders = [0, 1]
        gender_names = ['女', '男']
        gender_results = []
        
        for gender, name in zip(genders, gender_names):
            # 筛选数据
            gender_data = self.data[self.data['性别'] == gender]
            if len(gender_data) > 0:
                # 临时替换数据
                original_data = self.data.copy()
                self.data = gender_data
                
                # 分析
                result, weights = self.analyze_key_indicators()
                top10 = [item[0] for item in result[:10]]
                gender_results.append({
                    'gender': name,
                    'top10': top10,
                    'weights': weights
                })
                
                # 恢复数据
                self.data = original_data
                
                print(f"性别 {name}: 前10指标={top10}")
        
        # 年龄组分析
        age_groups = [1, 2, 3, 4, 5]
        age_names = ['40-49岁', '50-59岁', '60-69岁', '70-79岁', '80-89岁']
        age_results = []
        
        for age_group, name in zip(age_groups, age_names):
            # 筛选数据
            age_data = self.data[self.data['年龄组'] == age_group]
            if len(age_data) > 0:
                # 临时替换数据
                original_data = self.data.copy()
                self.data = age_data
                
                # 分析
                result, weights = self.analyze_key_indicators()
                top10 = [item[0] for item in result[:10]]
                age_results.append({
                    'age_group': name,
                    'top10': top10,
                    'weights': weights
                })
                
                # 恢复数据
                self.data = original_data
                
                print(f"年龄组 {name}: 前10指标={top10}")
        
        return gender_results, age_results
    
    def create_sensitivity_charts(self):
        """创建敏感性分析图表"""
        print("\n=== 创建敏感性分析图表 ===")
        
        # 1. 权重敏感性分析图表
        weight_results = self.sensitivity_analysis_weights()
        
        plt.figure(figsize=(12, 8))
        similarities = [r['similarity'] for r in weight_results]
        labels = [f"Spearman={r['weights']['spearman']:.2f}\nMI={r['weights']['mi']:.2f}\nPLS={r['weights']['pls']:.2f}" for r in weight_results]
        
        bars = plt.bar(range(len(similarities)), similarities, color='steelblue')
        plt.xlabel('权重组合', fontsize=14)
        plt.ylabel('与基准结果相似度', fontsize=14)
        plt.title('权重敏感性分析', fontsize=16)
        plt.xticks(range(len(similarities)), labels, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('sensitivity_analysis_weights.png', dpi=200, bbox_inches='tight')
        print("权重敏感性分析图表已保存")
        
        # 2. 特征敏感性分析图表
        feature_results = self.sensitivity_analysis_features()
        
        plt.figure(figsize=(14, 8))
        features = [r['removed_feature'] for r in feature_results]
        similarities = [r['similarity'] for r in feature_results]
        
        bars = plt.bar(range(len(similarities)), similarities, color='forestgreen')
        plt.xlabel('删除的特征', fontsize=14)
        plt.ylabel('与基准结果相似度', fontsize=14)
        plt.title('特征敏感性分析', fontsize=16)
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('sensitivity_analysis_features.png', dpi=200, bbox_inches='tight')
        print("特征敏感性分析图表已保存")
        
        # 3. 性别和年龄敏感性分析图表
        gender_results, age_results = self.sensitivity_analysis_gender_age()
        
        # 性别对比
        plt.figure(figsize=(16, 10))
        
        # 左侧：女性
        plt.subplot(1, 2, 1)
        if gender_results:
            female_top10 = gender_results[0]['top10']
            female_weights = gender_results[0]['weights']
            plt.barh(range(len(female_top10)), range(len(female_top10), 0, -1), color='pink')
            plt.yticks(range(len(female_top10)), female_top10)
            plt.title(f'女性前10关键指标\n权重: Spearman={female_weights['spearman']:.3f}, MI={female_weights['mi']:.3f}, PLS={female_weights['pls']:.3f}', fontsize=14)
            plt.gca().invert_yaxis()
        
        # 右侧：男性
        plt.subplot(1, 2, 2)
        if len(gender_results) > 1:
            male_top10 = gender_results[1]['top10']
            male_weights = gender_results[1]['weights']
            plt.barh(range(len(male_top10)), range(len(male_top10), 0, -1), color='lightblue')
            plt.yticks(range(len(male_top10)), male_top10)
            plt.title(f'男性前10关键指标\n权重: Spearman={male_weights['spearman']:.3f}, MI={male_weights['mi']:.3f}, PLS={male_weights['pls']:.3f}', fontsize=14)
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('sensitivity_analysis_gender.png', dpi=200, bbox_inches='tight')
        print("性别敏感性分析图表已保存")
        
        # 年龄组对比
        plt.figure(figsize=(18, 12))
        for i, age_result in enumerate(age_results):
            plt.subplot(2, 3, i+1)
            top10 = age_result['top10']
            weights = age_result['weights']
            plt.barh(range(len(top10)), range(len(top10), 0, -1), color='orange')
            plt.yticks(range(len(top10)), top10, fontsize=9)
            plt.title(f'{age_result['age_group']}\n权重: S={weights['spearman']:.2f}, M={weights['mi']:.2f}, P={weights['pls']:.2f}', fontsize=12)
            plt.gca().invert_yaxis()
            plt.xlabel('排名')
        
        plt.tight_layout()
        plt.savefig('sensitivity_analysis_age.png', dpi=200, bbox_inches='tight')
        print("年龄敏感性分析图表已保存")
    
    def run_full_analysis(self):
        """运行完整的敏感性分析"""
        print("=" * 80)
        print("问题一模型敏感性分析")
        print("=" * 80)
        
        # 加载数据
        self.load_data()
        
        # 基准分析
        print("\n=== 基准分析 ===")
        base_indicators, base_weights = self.analyze_key_indicators()
        print("基准权重:")
        print(f"  Spearman: {base_weights['spearman']:.4f}")
        print(f"  互信息: {base_weights['mi']:.4f}")
        print(f"  PLS载荷: {base_weights['pls']:.4f}")
        print("\n基准前10关键指标:")
        for i, (indicator, score) in enumerate(base_indicators[:10]):
            print(f"  {i+1}. {indicator}: {score:.4f}")
        
        # 体质贡献度分析
        print("\n=== 体质贡献度分析 ===")
        constitution_contribution = self.analyze_constitution_contribution()
        print("体质贡献度排序:")
        for i, (constitution, contribution) in enumerate(constitution_contribution):
            print(f"  {i+1}. {constitution}: {contribution:.4f}")
        
        # 敏感性分析
        self.create_sensitivity_charts()
        
        print("\n" + "=" * 80)
        print("敏感性分析完成")
        print("=" * 80)

if __name__ == '__main__':
    analyzer = Problem1SensitivityAnalyzer()
    analyzer.run_full_analysis()
