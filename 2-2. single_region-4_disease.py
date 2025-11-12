import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, spearmanr

# === 字體設定 ===
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# === 路徑設定 ===
input_folder = "./1_disease_with_PM25_lag0-10"
output_folder = "./2-3_disease_PM25_lag0_scatter_by_region"
upper_folder = os.path.join(output_folder, "upper")
lower_folder = os.path.join(output_folder, "lower")
os.makedirs(upper_folder, exist_ok=True)
os.makedirs(lower_folder, exist_ok=True)

# === 顏色設定 ===（這次是疾病顏色）
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

# === 疾病分類 ===
upper_respiratory = ["急性Rhinosinusitis", "Allergic rhinitis", "Influenza"]
lower_respiratory = ["慢性Bronchitis", "Pneumonia", "氣喘"]

# === 區域列表 ===
regions = ["高屏", "雲嘉南", "苗中彰投", "北北基桃竹", "宜花東"]

# === 讀入所有疾病資料 ===
all_disease_data = {}
for file in os.listdir(input_folder):
    if not file.endswith("_PM25_lag.csv"):
        continue
    disease_name = file.replace("_PM25_lag.csv", "")
    df = pd.read_csv(os.path.join(input_folder, file))
    df = df.dropna(subset=["PM25_lag0", "case_per_capita(‰)"])
    all_disease_data[disease_name] = df

print(f"✅ 已讀入 {len(all_disease_data)} 種疾病資料")

# === 上呼吸道與下呼吸道分開畫 ===
for group_name, disease_list, save_folder in [
    ("上呼吸道", upper_respiratory, upper_folder),
    ("下呼吸道", lower_respiratory, lower_folder)
]:

    for region in regions:
        plt.figure(figsize=(8, 6))
        plt.title(f"{region} — {group_name}疾病 PM2.5(lag0) 與就診率", fontsize=14)

        legend_handles = []
        legend_labels = []

        # === 疾病回圈 ===
        for disease_name in disease_list:
            if disease_name not in all_disease_data:
                continue

            df = all_disease_data[disease_name]
            df_region = df[df["region"] == region]
            if df_region.empty:
                continue

            color = disease_colors.get(disease_name, "gray")

            # 畫散點
            scatter = plt.scatter(
                df_region["PM25_lag0"], df_region["case_per_capita(‰)"],
                label=disease_name, color=color, alpha=0.7
            )

            # 線性回歸線
            slope, intercept, *_ = linregress(
                df_region["PM25_lag0"], df_region["case_per_capita(‰)"]
            )
            x_vals = sorted(df_region["PM25_lag0"])
            y_vals = [slope * x + intercept for x in x_vals]
            plt.plot(x_vals, y_vals, color=color, linestyle="-", linewidth=1.5)

            legend_handles.append(scatter)
            legend_labels.append(f"{disease_name} (slope={slope:.3f})")

        plt.xlabel("PM2.5 lag0 平均暴露量 (μg/m³)", fontsize=12)
        plt.ylabel("就診率 (‰)", fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(legend_handles, legend_labels,
                   title="疾病與回歸斜率", fontsize=9,
                   loc="lower right", frameon=True)

        plt.tight_layout()
        save_path = os.path.join(save_folder, f"{region}_{group_name}_scatter.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"✅ 已輸出：{save_path}")


print("\n🎯 各地區多疾病散布圖繪製完成！")
