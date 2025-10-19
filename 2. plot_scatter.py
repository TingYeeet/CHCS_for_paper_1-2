import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, spearmanr

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# === 路徑設定 ===
input_folder = "./1_disease_with_PM25_lag0-10"
output_folder = "./2_disease_PM25_lag0_scatter"
upper_folder = os.path.join(output_folder, "upper")
lower_folder = os.path.join(output_folder, "lower")
os.makedirs(upper_folder, exist_ok=True)
os.makedirs(lower_folder, exist_ok=True)

# === 顏色設定 ===
region_colors = {
    "高屏": "#AA04AA",
    "雲嘉南": "#FF0000",
    "苗中彰投": "#FFA500",
    "北北基桃竹": "#FFFF00",
    "宜花東": "#23B623"
}

# === 疾病分類 ===
upper_respiratory = ["URI", "急性Rhinosinusitis", "Allergic rhinitis", "Influenza"]
lower_respiratory = ["急性Bronchitis", "慢性Bronchitis", "Pneumonia", "氣喘"]

# === 讀取每個疾病檔案 ===
for file in os.listdir(input_folder):
    if not file.endswith("_PM25_lag.csv"):
        continue

    disease_name = file.replace("_PM25_lag.csv", "")
    print(f"=== 繪製疾病：{disease_name} ===")

    # 判斷分類
    if disease_name in upper_respiratory:
        save_folder = upper_folder
    elif disease_name in lower_respiratory:
        save_folder = lower_folder
    else:
        save_folder = output_folder  # 若未分類，放總資料夾

    df = pd.read_csv(os.path.join(input_folder, file))
    df = df.dropna(subset=["PM25_lag0", "case_per_capita(‰)"])

    # 計算整體 Pearson / Spearman
    pearson_r, _ = pearsonr(df["PM25_lag0"], df["case_per_capita(‰)"])
    spearman_r, _ = spearmanr(df["PM25_lag0"], df["case_per_capita(‰)"])

    # 畫圖
    plt.figure(figsize=(8, 6))
    plt.title(f"{disease_name} — PM2.5 (lag0) 與 就診率 散布圖", fontsize=14)

    legend_handles = []
    legend_labels = []

    for region, color in region_colors.items():
        df_region = df[df["region"] == region]
        if df_region.empty:
            continue

        scatter = plt.scatter(
            df_region["PM25_lag0"], df_region["case_per_capita(‰)"],
            label=region, color=color, alpha=0.7
        )

        slope, intercept, *_ = linregress(
            df_region["PM25_lag0"], df_region["case_per_capita(‰)"]
        )
        x_vals = sorted(df_region["PM25_lag0"])
        y_vals = [slope * x + intercept for x in x_vals]
        plt.plot(x_vals, y_vals, color=color, linestyle="-", linewidth=1.5)

        legend_handles.append(scatter)
        legend_labels.append(f"{region} (slope={slope:.3f})")

    plt.xlabel("PM2.5 lag 0週平均暴露量 (μg/m³)", fontsize=12)
    plt.ylabel("就診率 (‰)", fontsize=12)
    plt.grid(alpha=0.3)

    # 圖例（僅圓點）
    plt.legend(legend_handles, legend_labels,
               title="地區與回歸斜率", fontsize=9,
               loc="lower right", frameon=True)

    # 左上角文字（整體）
    text = (
        "              整體\n"
        f"Pearson r = {pearson_r:.3f}\n"
        f"Spearman ρ = {spearman_r:.3f}"
    )
    plt.text(
        0.02, 0.98, text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='black',
                  boxstyle='round,pad=0.4', alpha=0.8)
    )

    # 儲存圖片到對應資料夾
    output_path = os.path.join(save_folder, f"{disease_name}_PM25_lag0_scatter.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ 已輸出：{output_path}")

print("\n🎯 所有疾病散布圖繪製完成！")
