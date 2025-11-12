import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

input_folder = "1-4_little_area_lag"
output_folder = "3-3_pearson_all_taiwan_fixed"
os.makedirs(output_folder, exist_ok=True)

upper_respiratory = ["URI", "急性Rhinosinusitis", "Allergic rhinitis", "Influenza"]
lower_respiratory = ["急性Bronchitis", "慢性Bronchitis", "Pneumonia", "氣喘"]

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

lag_cols = [f"PM25_lag{i}" for i in range(11)]
x = list(range(11))

# === ✅ 修正版：正確移除 "_PM25_lag.csv" 後的疾病名稱 ===
disease_files = [f for f in os.listdir(input_folder) if f.endswith("_PM25_lag.csv")]
disease_dfs = {}
for file in disease_files:
    name = file.replace("_PM25_lag.csv", "")  # <-- 只去掉這個部分，保留正確疾病名稱
    df = pd.read_csv(os.path.join(input_folder, file), dtype={"ID": str})
    disease_dfs[name] = df
    print(f"讀入 {name}: {len(df)} 筆")

def compute_pearson_for_df(df):
    vals = []
    for lag in lag_cols:
        if lag not in df.columns or "case_per_capita(‰)" not in df.columns:
            vals.append(np.nan)
            continue
        sub = df[[lag, "case_per_capita(‰)"]].dropna()
        if len(sub) < 4:
            vals.append(np.nan)
            continue
        try:
            r, p = pearsonr(sub[lag], sub["case_per_capita(‰)"])
            vals.append(r)
        except Exception:
            vals.append(np.nan)
    return vals

def plot_group(disease_list, title, fname):
    plt.figure(figsize=(10,6))
    has_any = False
    for disease in disease_list:
        if disease not in disease_dfs:
            print(f"跳過：找不到 {disease} 的檔案")
            continue
        df = disease_dfs[disease]
        pearson_vals = compute_pearson_for_df(df)
        if all(np.isnan(pearson_vals)):
            print(f"跳過 {disease}（所有 lag 的 Pearson 值皆為 NaN）")
            continue
        has_any = True
        plt.plot(x, pearson_vals, marker='o', label=disease, color=disease_colors.get(disease, None))
        for xi, yi in zip(x, pearson_vals):
            if not np.isnan(yi):
                plt.text(xi, yi, f"{yi:.2f}", fontsize=8, ha='center', va='bottom')

    plt.axhline(0, color='k', linewidth=1)
    plt.xticks(x, [f"lag{i}" for i in x])
    plt.ylim(-0.1, 0.5)
    plt.xlabel("Lag (週)")
    plt.ylabel("Pearson r")
    plt.title(title)
    if has_any:
        plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, fname), dpi=300)
    plt.close()
    print("輸出：", fname)

# === 畫圖 ===
plot_group(upper_respiratory, "全台上呼吸道疾病 PM2.5 lag0~lag10 Pearson", "upper_respiratory_pearson.png")
plot_group(lower_respiratory, "全台下呼吸道疾病 PM2.5 lag0~lag10 Pearson", "lower_respiratory_pearson.png")
