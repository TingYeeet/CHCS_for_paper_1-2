#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PM2.5 門檻值分析
基於 cubic spline 找出轉折點作為PM2.5濃度門檻，分析門檻前後看診率變化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.signal import find_peaks, argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

class PM25ThresholdAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.results = []
        self.all_threshold_effects = []  # 存儲所有門檻效應的詳細資訊
        
    def load_disease_data(self, disease_file):
        """載入疾病資料"""
        file_path = f"{self.data_path}/{disease_file}"
        df = pd.read_csv(file_path)
        return df
    
    def fit_cubic_model(self, x, y):
        """擬合四次多項式（保證至少兩個轉折點）"""
        # 排序資料以確保擬合穩定
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        # 四次多項式特徵
        quartic_features = PolynomialFeatures(degree=4)
        x_quartic = quartic_features.fit_transform(x_sorted.reshape(-1, 1))

        # 初次擬合
        quartic_reg = LinearRegression()
        quartic_reg.fit(x_quartic, y_sorted)
        coef = quartic_reg.coef_
        a, b, c, d = coef[4], coef[3], coef[2], coef[1]

        # --- 定義三次導數的判別式（確保至少兩轉折點）---
        def cubic_discriminant(a, b, c, d):
            # 對三次方程 4a x^3 + 3b x^2 + 2c x + d = 0 的判別式
            A = 4*a
            B = 3*b
            C = 2*c
            D = d
            Δ = 18*A*B*C*D - 4*(B**3)*D + (B**2)*(C**2) - 4*A*(C**3) - 27*(A**2)*(D**2)
            return Δ

        Δ = cubic_discriminant(a, b, c, d)

        # 若不滿足至少兩轉折條件，做微調嘗試
        attempt = 0
        while Δ <= 0 and attempt < 5:
            noise = np.random.normal(scale=0.01 * np.std(y_sorted), size=y_sorted.shape)
            y_perturbed = y_sorted + noise
            quartic_reg.fit(x_quartic, y_perturbed)
            coef = quartic_reg.coef_
            a, b, c, d = coef[4], coef[3], coef[2], coef[1]
            Δ = cubic_discriminant(a, b, c, d)
            attempt += 1

        # 建立更密集的 x 用於平滑曲線
        x_dense = np.linspace(x_sorted.min(), x_sorted.max(), 1000)
        x_dense_quartic = quartic_features.transform(x_dense.reshape(-1, 1))
        y_dense_pred = quartic_reg.predict(x_dense_quartic)

        # 額外記錄 quartic fit 的轉折確認
        if Δ > 0:
            print(f"✅ quartic 擬合具有至少兩個轉折點 (Δ = {Δ:.3e})")
        else:
            print(f"⚠️ quartic 擬合轉折不足 (Δ = {Δ:.3e})，可能為單峰或單調")

        return x_sorted, y_sorted, x_dense, y_dense_pred, quartic_reg, quartic_features
    
    def find_inflection_points(self, x_dense, y_dense_pred):
        """找出拐點（二階導數為零的點）"""
        # 計算一階導數
        dy_dx = np.gradient(y_dense_pred, x_dense)
        
        # 計算二階導數
        d2y_dx2 = np.gradient(dy_dx, x_dense)
        
        # 找出二階導數過零點（拐點）
        # 尋找符號變化的點
        sign_changes = np.diff(np.sign(d2y_dx2))
        inflection_indices = np.where(sign_changes != 0)[0]
        
        # 過濾掉太接近邊界的點
        boundary_margin = len(x_dense) // 20  # 5%的邊界
        inflection_indices = inflection_indices[
            (inflection_indices > boundary_margin) & 
            (inflection_indices < len(x_dense) - boundary_margin)
        ]
        
        inflection_points = []
        for idx in inflection_indices:
            # 使用線性插值找到更精確的拐點
            if idx < len(x_dense) - 1:
                x_inflection = x_dense[idx]
                y_inflection = y_dense_pred[idx]
                inflection_points.append((x_inflection, y_inflection))
        
        return inflection_points, dy_dx, d2y_dx2
    
    def find_extreme_points(self, x_dense, y_dense_pred):
        """找出極值點（一階導數為零的點）"""
        # 計算一階導數
        dy_dx = np.gradient(y_dense_pred, x_dense)
        
        # 找出一階導數過零點（極值點）
        sign_changes = np.diff(np.sign(dy_dx))
        extreme_indices = np.where(sign_changes != 0)[0]
        
        # 過濾掉太接近邊界的點
        boundary_margin = len(x_dense) // 20
        extreme_indices = extreme_indices[
            (extreme_indices > boundary_margin) & 
            (extreme_indices < len(x_dense) - boundary_margin)
        ]
        
        extreme_points = []
        for idx in extreme_indices:
            if idx < len(x_dense) - 1:
                x_extreme = x_dense[idx]
                y_extreme = y_dense_pred[idx]
                extreme_points.append((x_extreme, y_extreme))
        
        return extreme_points
    
    def calculate_threshold_effects(self, x, y, thresholds):
        """計算門檻前後的效應"""
        threshold_effects = []
        
        for threshold_pm25, _ in thresholds:
            # 分割資料
            below_mask = x <= threshold_pm25
            above_mask = x > threshold_pm25
            
            below_x = x[below_mask]
            below_y = y[below_mask]
            above_x = x[above_mask]
            above_y = y[above_mask]
            
            if len(below_y) < 3 or len(above_y) < 3:
                continue
            
            # 計算平均值和標準差
            below_mean = np.mean(below_y)
            below_std = np.std(below_y)
            above_mean = np.mean(above_y)
            above_std = np.std(above_y)
            
            # 計算變化
            absolute_change = above_mean - below_mean
            relative_change = (absolute_change / below_mean) * 100 if below_mean != 0 else 0
            
            # 計算效應大小 (Cohen's d)
            pooled_std = np.sqrt(((len(below_y) - 1) * below_std**2 + (len(above_y) - 1) * above_std**2) / 
                               (len(below_y) + len(above_y) - 2))
            cohens_d = absolute_change / pooled_std if pooled_std != 0 else 0
            
            threshold_effects.append({
                'threshold_pm25': threshold_pm25,
                'below_count': len(below_y),
                'above_count': len(above_y),
                'below_mean': below_mean,
                'below_std': below_std,
                'above_mean': above_mean,
                'above_std': above_std,
                'absolute_change': absolute_change,
                'relative_change': relative_change,
                'cohens_d': cohens_d
            })
        
        return threshold_effects
    
    def analyze_disease_threshold(self, disease_file):
        """分析單一疾病的門檻效應"""
        try:
            # 載入資料
            df = self.load_disease_data(disease_file)
            
            # 準備資料
            x = df['PM25_lag0'].values
            y = df['case_per_capita(‰)'].values
            
            # 移除缺失值
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]
            
            if len(x) < 20:  # 資料點太少則跳過
                return None
            
            # 擬合三次模型
            x_sorted, y_sorted, x_dense, y_dense_pred, cubic_reg, cubic_features = self.fit_cubic_model(x, y)
            
            # 計算R²
            x_cubic = cubic_features.transform(x_sorted.reshape(-1, 1))
            y_pred = cubic_reg.predict(x_cubic)
            cubic_r2 = r2_score(y_sorted, y_pred)
            
            # 找出拐點和極值點
            inflection_points, dy_dx, d2y_dx2 = self.find_inflection_points(x_dense, y_dense_pred)
            extreme_points = self.find_extreme_points(x_dense, y_dense_pred)
            
            # 合併所有潛在門檻點
            all_thresholds = inflection_points + extreme_points
            
            if not all_thresholds:
                # 如果沒有找到明顯的轉折點，使用分位數作為門檻
                percentiles = [25, 50, 75]
                all_thresholds = [(np.percentile(x, p), np.percentile(y, p)) for p in percentiles]
                threshold_type = "Percentile-based"
            else:
                # 按PM2.5濃度排序
                all_thresholds = sorted(all_thresholds, key=lambda x: x[0])
                threshold_type = "Inflection/Extreme points"
            
            # 計算門檻效應
            threshold_effects = self.calculate_threshold_effects(x_sorted, y_sorted, all_thresholds)
            
            # 疾病名稱處理
            disease_name = disease_file.replace('.csv', '')
            
            # 選擇最顯著的門檻（基於效應大小）
            if threshold_effects:
                best_threshold = max(threshold_effects, key=lambda x: abs(x['cohens_d']))
                
                result = {
                    '疾病名稱': disease_name,
                    '門檻類型': threshold_type,
                    'Cubic_R2': f"{cubic_r2:.4f}",
                    '最佳門檻PM2.5': f"{best_threshold['threshold_pm25']:.2f}",
                    '門檻下樣本數': best_threshold['below_count'],
                    '門檻上樣本數': best_threshold['above_count'],
                    '門檻下平均看診率': f"{best_threshold['below_mean']:.4f}",
                    '門檻上平均看診率': f"{best_threshold['above_mean']:.4f}",
                    '絕對變化': f"{best_threshold['absolute_change']:.4f}",
                    '相對變化': f"{best_threshold['relative_change']:.2f}%",
                    'Cohen_d': f"{best_threshold['cohens_d']:.4f}",
                    '門檻點數量': len(all_thresholds)
                }
                
                return result, x_sorted, y_sorted, x_dense, y_dense_pred, all_thresholds, threshold_effects, dy_dx, d2y_dx2, cubic_r2
            else:
                return None
                
        except Exception as e:
            print(f"分析 {disease_file} 時發生錯誤: {str(e)}")
            return None
    
    def plot_threshold_analysis(self, disease_name, x_sorted, y_sorted, x_dense, y_dense_pred, 
                               all_thresholds, threshold_effects, dy_dx, d2y_dx2, save_path=None):
        """繪製門檻分析圖"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{disease_name} - PM2.5 Threshold Analysis', fontsize=16, fontweight='bold')
        
        # 第一個子圖：原始資料和擬合曲線
        ax1 = axes[0, 0]
        ax1.scatter(x_sorted, y_sorted, alpha=0.6, color='lightblue', s=30, label='Original Data')
        ax1.plot(x_dense, y_dense_pred, 'r-', linewidth=2, label='Cubic Fit')
        
        # 標示門檻點
        for i, (x_thresh, y_thresh) in enumerate(all_thresholds):
            ax1.axvline(x=x_thresh, color='orange', linestyle='--', alpha=0.7)
            ax1.plot(x_thresh, y_thresh, 'ro', markersize=8, label=f'Threshold {i+1}' if i < 3 else "")
        
        ax1.set_title('Data and Cubic Fit with Thresholds')
        ax1.set_xlabel('PM2.5 Concentration (μg/m³)')
        ax1.set_ylabel('Consultation Rate (‰)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 第二個子圖：一階導數
        ax2 = axes[0, 1]
        ax2.plot(x_dense, dy_dx, 'g-', linewidth=2, label='First Derivative')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        for x_thresh, _ in all_thresholds:
            ax2.axvline(x=x_thresh, color='orange', linestyle='--', alpha=0.7)
        ax2.set_title('First Derivative')
        ax2.set_xlabel('PM2.5 Concentration (μg/m³)')
        ax2.set_ylabel('dy/dx')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 第三個子圖：二階導數
        ax3 = axes[1, 0]
        ax3.plot(x_dense, d2y_dx2, 'purple', linewidth=2, label='Second Derivative')
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        for x_thresh, _ in all_thresholds:
            ax3.axvline(x=x_thresh, color='orange', linestyle='--', alpha=0.7)
        ax3.set_title('Second Derivative (Inflection Points)')
        ax3.set_xlabel('PM2.5 Concentration (μg/m³)')
        ax3.set_ylabel('d²y/dx²')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 第四個子圖：門檻效應比較
        ax4 = axes[1, 1]
        if threshold_effects:
            thresholds_pm25 = [te['threshold_pm25'] for te in threshold_effects]
            relative_changes = [te['relative_change'] for te in threshold_effects]
            cohens_d = [te['cohens_d'] for te in threshold_effects]
            
            # 雙軸圖
            ax4_twin = ax4.twinx()
            
            bars1 = ax4.bar([f'{pm:.1f}' for pm in thresholds_pm25], relative_changes, 
                           alpha=0.7, color='skyblue', label='Relative Change (%)')
            bars2 = ax4_twin.bar([f'{pm:.1f}' for pm in thresholds_pm25], cohens_d, 
                                alpha=0.7, color='salmon', width=0.5, label="Cohen's d")
            
            ax4.set_title('Threshold Effects Comparison')
            ax4.set_xlabel('PM2.5 Threshold (μg/m³)')
            ax4.set_ylabel('Relative Change (%)', color='blue')
            ax4_twin.set_ylabel("Cohen's d", color='red')
            ax4.tick_params(axis='y', labelcolor='blue')
            ax4_twin.tick_params(axis='y', labelcolor='red')
            
            # 添加數值標籤
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                height1 = bar1.get_height()
                height2 = bar2.get_height()
                ax4.text(bar1.get_x() + bar1.get_width()/2., height1,
                        f'{height1:.1f}%', ha='center', va='bottom', fontsize=8)
                ax4_twin.text(bar2.get_x() + bar2.get_width()/2., height2,
                             f'{height2:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/{disease_name}_threshold_analysis.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def run_analysis(self):
        """執行完整門檻分析"""
        # 獲取所有疾病檔案
        disease_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        
        print("開始進行 PM2.5 門檻值分析...")
        print("=" * 80)
        
        save_path = "./6-2_Threshold_Analysis"
        
        for disease_file in disease_files:
            print(f"\n分析疾病: {disease_file}")
            
            try:
                result_data = self.analyze_disease_threshold(disease_file)
                
                if result_data is None:
                    print(f"跳過 {disease_file} - 資料不足或無明顯門檻")
                    continue
                
                result, x_sorted, y_sorted, x_dense, y_dense_pred, all_thresholds, threshold_effects, dy_dx, d2y_dx2, cubic_r2 = result_data
                self.results.append(result)
                
                # 保存所有門檻效應到詳細列表
                for i, te in enumerate(threshold_effects, 1):
                    threshold_detail = {
                        '疾病名稱': result['疾病名稱'],
                        'Cubic_R2': result['Cubic_R2'],
                        '門檻編號': i,
                        '門檻PM2.5': f"{te['threshold_pm25']:.2f}",
                        '門檻下樣本數': te['below_count'],
                        '門檻上樣本數': te['above_count'],
                        '門檻下平均看診率': f"{te['below_mean']:.4f}",
                        '門檻下標準差': f"{te['below_std']:.4f}",
                        '門檻上平均看診率': f"{te['above_mean']:.4f}",
                        '門檻上標準差': f"{te['above_std']:.4f}",
                        '絕對變化': f"{te['absolute_change']:.4f}",
                        '相對變化': f"{te['relative_change']:.2f}%",
                        'Cohen_d': f"{te['cohens_d']:.4f}"
                    }
                    self.all_threshold_effects.append(threshold_detail)
                
                # 顯示結果
                print(f"門檻類型: {result['門檻類型']}")
                print(f"Cubic R²: {result['Cubic_R2']}")
                print(f"最佳PM2.5門檻: {result['最佳門檻PM2.5']} μg/m³")
                print(f"門檻下平均看診率: {result['門檻下平均看診率']} ‰")
                print(f"門檻上平均看診率: {result['門檻上平均看診率']} ‰")
                print(f"相對變化: {result['相對變化']}")
                print(f"效應大小 (Cohen's d): {result['Cohen_d']}")
                
                # 顯示所有門檻點的詳細資訊
                print(f"\n所有門檻點詳細資訊 (共{len(threshold_effects)}個):")
                for i, te in enumerate(threshold_effects, 1):
                    print(f"  門檻{i}: {te['threshold_pm25']:.2f} μg/m³")
                    print(f"    門檻下: {te['below_mean']:.4f}‰ (n={te['below_count']})")
                    print(f"    門檻上: {te['above_mean']:.4f}‰ (n={te['above_count']})")
                    print(f"    相對變化: {te['relative_change']:.2f}%")
                    print(f"    Cohen's d: {te['cohens_d']:.4f}")
                    print()
                
                # 繪製分析圖
                self.plot_threshold_analysis(result['疾病名稱'], x_sorted, y_sorted, x_dense, y_dense_pred, 
                                           all_thresholds, threshold_effects, dy_dx, d2y_dx2, save_path)
                
            except Exception as e:
                print(f"分析 {disease_file} 時發生錯誤: {str(e)}")
                continue
        
        # 建立結果總表
        self.create_summary_table()
        
        # 建立所有門檻點的詳細表格
        self.create_all_thresholds_table()
    
    def create_summary_table(self):
        """建立結果總表"""
        if not self.results:
            print("沒有分析結果可顯示")
            return
        
        # 轉換為DataFrame
        df_results = pd.DataFrame(self.results)
        
        print("\n" + "=" * 100)
        print("PM2.5 門檻值分析結果總表")
        print("=" * 100)
        print(df_results.to_string(index=False))
        
        # 儲存結果
        output_path = "./6-2_Threshold_Analysis"
        os.makedirs(output_path, exist_ok=True)
        df_results.to_csv(f"{output_path}/PM25_threshold_analysis_results.csv", 
                         index=False, encoding='utf-8-sig')
        
        # 統計分析
        print("\n" + "=" * 80)
        print("統計摘要:")
        print("=" * 80)
        
        # 門檻值統計
        thresholds = [float(t) for t in df_results['最佳門檻PM2.5']]
        print(f"PM2.5門檻值統計:")
        print(f"  平均門檻: {np.mean(thresholds):.2f} μg/m³")
        print(f"  中位數門檻: {np.median(thresholds):.2f} μg/m³")
        print(f"  門檻範圍: {np.min(thresholds):.2f} - {np.max(thresholds):.2f} μg/m³")
        
        # 效應大小統計
        cohens_d = [float(cd) for cd in df_results['Cohen_d']]
        print(f"\n效應大小統計 (Cohen's d):")
        print(f"  平均效應大小: {np.mean(cohens_d):.4f}")
        print(f"  中位數效應大小: {np.median(cohens_d):.4f}")
        
        # 效應大小分類
        small_effect = sum(1 for d in cohens_d if abs(d) >= 0.2 and abs(d) < 0.5)
        medium_effect = sum(1 for d in cohens_d if abs(d) >= 0.5 and abs(d) < 0.8)
        large_effect = sum(1 for d in cohens_d if abs(d) >= 0.8)
        
        print(f"\n效應大小分類:")
        print(f"  小效應 (0.2 ≤ |d| < 0.5): {small_effect} 個疾病")
        print(f"  中等效應 (0.5 ≤ |d| < 0.8): {medium_effect} 個疾病")
        print(f"  大效應 (|d| ≥ 0.8): {large_effect} 個疾病")
        
        # 相對變化統計
        relative_changes = [float(rc.replace('%', '')) for rc in df_results['相對變化']]
        print(f"\n相對變化統計:")
        print(f"  平均相對變化: {np.mean(relative_changes):.2f}%")
        print(f"  中位數相對變化: {np.median(relative_changes):.2f}%")
        print(f"  最大相對變化: {np.max(relative_changes):.2f}%")
        print(f"  最小相對變化: {np.min(relative_changes):.2f}%")
        
        print(f"\n結果已儲存至: {output_path}/PM25_threshold_analysis_results.csv")
        print(f"圖表已儲存至: {output_path}/")
    
    def create_all_thresholds_table(self):
        """建立所有門檻點的詳細表格"""
        if not self.all_threshold_effects:
            print("沒有門檻效應資料可顯示")
            return
        
        # 轉換為DataFrame
        df_all_thresholds = pd.DataFrame(self.all_threshold_effects)
        
        print("\n" + "=" * 120)
        print("所有門檻點詳細分析結果")
        print("=" * 120)
        print(df_all_thresholds.to_string(index=False))
        
        # 儲存結果
        output_path = "./6-2_Threshold_Analysis"
        os.makedirs(output_path, exist_ok=True)
        df_all_thresholds.to_csv(f"{output_path}/PM25_all_thresholds_detailed_analysis.csv", 
                                index=False, encoding='utf-8-sig')
        
        # 按疾病分組統計
        print("\n" + "=" * 80)
        print("按疾病分組的門檻統計:")
        print("=" * 80)
        
        for disease in df_all_thresholds['疾病名稱'].unique():
            disease_data = df_all_thresholds[df_all_thresholds['疾病名稱'] == disease]
            print(f"\n{disease}:")
            print(f"  R²: {disease_data['Cubic_R2'].iloc[0]}")
            print(f"  門檻點數量: {len(disease_data)}")
            
            for _, row in disease_data.iterrows():
                print(f"  門檻{row['門檻編號']}: {row['門檻PM2.5']} μg/m³ "
                      f"(相對變化: {row['相對變化']}, Cohen's d: {row['Cohen_d']})")
        
        print(f"\n詳細結果已儲存至: {output_path}/PM25_all_thresholds_detailed_analysis.csv")

def main():
    # 設定資料路徑
    data_path = "./1_disease_with_PM25_lag0-10"
    
    # 建立分析物件
    analyzer = PM25ThresholdAnalysis(data_path)
    
    # 執行分析
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
