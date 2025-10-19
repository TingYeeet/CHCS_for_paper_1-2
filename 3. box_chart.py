import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 顏色對應（可自行擴展）
disease_colors = {
    "URI": "#1f77b4",
    "急性Rhinosinusitis": "#ff7f0e",
    "Allergic rhinitis": "#2ca02c",
    "Influenza": "#d62728",
    "急性Bronchitis": "#9467bd",
    "慢性Bronchitis": "#8c564b",
    "Pneumonia": "#e377c2",
    "氣喘": "#7f7f7f"
}

# 區域清單
regions = ["高屏", "苗中彰投", "雲嘉南", "北北基桃竹", "宜花東"]

# Lag 列
lag_cols = [f"PM25_lag{i}" for i in range(11)]

# 輸入、輸出資料夾
input_folder = "1_disease_with_PM25_lag0-10"
output_folder_upper = "3-1_pearson_by_region_upper"
output_folder_lower = "3-2_pearson_by_region_lower"
os.makedirs(output_folder_upper, exist_ok=True)
os.makedirs(output_folder_lower, exist_ok=True)

# 疾病分類
upper_respiratory = ["URI", "急性Rhinosinusitis", "Allergic rhinitis", "Influenza"]
lower_respiratory = ["急性Bronchitis", "慢性Bronchitis", "Pneumonia", "氣喘"]

# 讀取所有疾病資料
disease_files = [f for f in os.listdir(input_folder) if f.endswith("_PM25_lag.csv")]
disease_dfs = {}
for file in disease_files:
    disease_name = file.replace("_PM25_lag.csv", "")
    df = pd.read_csv(os.path.join(input_folder, file))
    df = df.dropna(subset=lag_cols + ["case_per_capita(‰)"])
    disease_dfs[disease_name] = df

# 依區域繪圖
for region in regions:
    # 上呼吸道
    plt.figure(figsize=(12, 6))
    for disease in upper_respiratory:
        if disease not in disease_dfs:
            continue
        df = disease_dfs[disease]
        df_region = df[df["region"] == region]
        if df_region.empty:
            continue
        pearson_vals = []
        for lag in lag_cols:
            val = pearsonr(df_region[lag], df_region["case_per_capita(‰)"])[0]
            pearson_vals.append(val)
        plt.plot(lag_cols, pearson_vals, marker='o', label=disease,
                 color=disease_colors.get(disease, "#000000"))
        for x, y in zip(lag_cols, pearson_vals):
            plt.text(x, y, f"{y:.2f}", fontsize=8, ha='center', va='bottom')
    plt.axhline(0, color="black", linewidth=1)
    plt.ylim(0, 1)
    plt.title(f"{region} 上呼吸道疾病 PM2.5 lag0~lag10 Pearson")
    plt.xlabel("Lag (週)")
    plt.ylabel("Pearson 相關係數")
    plt.legend(loc="lower right")
    plt.tight_layout()
    output_path = os.path.join(output_folder_upper, f"{region}_upper_pearson.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 輸出 {output_path}")

    # 下呼吸道
    plt.figure(figsize=(12, 6))
    for disease in lower_respiratory:
        if disease not in disease_dfs:
            continue
        df = disease_dfs[disease]
        df_region = df[df["region"] == region]
        if df_region.empty:
            continue
        pearson_vals = []
        for lag in lag_cols:
            val = pearsonr(df_region[lag], df_region["case_per_capita(‰)"])[0]
            pearson_vals.append(val)
        plt.plot(lag_cols, pearson_vals, marker='o', label=disease,
                 color=disease_colors.get(disease, "#000000"))
        for x, y in zip(lag_cols, pearson_vals):
            plt.text(x, y, f"{y:.2f}", fontsize=8, ha='center', va='bottom')
    plt.axhline(0, color="black", linewidth=1)
    plt.ylim(-1, 1)
    plt.title(f"{region} 下呼吸道疾病 PM2.5 lag0~lag10 Pearson")
    plt.xlabel("Lag (週)")
    plt.ylabel("Pearson 相關係數")
    plt.legend(loc="lower right")
    plt.tight_layout()
    output_path = os.path.join(output_folder_lower, f"{region}_lower_pearson.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 輸出 {output_path}")

print("\n🎯 所有地區的上/下呼吸道疾病 Pearson 圖完成！")
