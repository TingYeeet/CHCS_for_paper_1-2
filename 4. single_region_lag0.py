import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, spearmanr

# === 字型設定 ===
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# === 路徑設定 ===
input_folder = "./1_disease_with_PM25_lag0-10"
output_folder = "./4_disease_PM25_lag0_scatter_single_region"
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
    print(f"=== 處理疾病：{disease_name} ===")

    # 判斷分類
    if disease_name in upper_respiratory:
        save_folder = upper_folder
    elif disease_name in lower_respiratory:
        save_folder = lower_folder
    else:
        save_folder = output_folder

    # 讀取資料
    df = pd.read_csv(os.path.join(input_folder, file))
    df = df.dropna(subset=["PM25_lag0", "case_per_capita(‰)"])

    # === 對每個地區個別畫圖 ===
    for region, color in region_colors.items():
        df_region = df[df["region"] == region]
        if df_region.empty:
            continue

        # 計算 Pearson / Spearman
        pearson_r, _ = pearsonr(df_region["PM25_lag0"], df_region["case_per_capita(‰)"])
        spearman_r, _ = spearmanr(df_region["PM25_lag0"], df_region["case_per_capita(‰)"])

        # 線性回歸
        slope, intercept, *_ = linregress(
            df_region["PM25_lag0"], df_region["case_per_capita(‰)"]
        )
        x_vals = sorted(df_region["PM25_lag0"])
        y_vals = [slope * x + intercept for x in x_vals]

        # 畫圖
        plt.figure(figsize=(7, 5))
        plt.scatter(
            df_region["PM25_lag0"], df_region["case_per_capita(‰)"],
            label=f"{region}", color=color, alpha=0.7
        )
        plt.plot(x_vals, y_vals, color=color, linestyle="-", linewidth=1.5)

        plt.title(f"{disease_name} — {region}\nPM2.5 (lag0) 與 就診率 散布圖", fontsize=13)
        plt.xlabel("PM2.5 lag 0週平均暴露量 (μg/m³)", fontsize=11)
        plt.ylabel("就診率 (‰)", fontsize=11)
        plt.grid(alpha=0.3)

        # 左上角文字（整體）
        text = (
            f"Pearson r = {pearson_r:.3f}\n"
            f"Spearman ρ = {spearman_r:.3f}\n"
            f"Slope = {slope:.3f}"
        )
        plt.text(
            0.02, 0.98, text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='black',
                      boxstyle='round,pad=0.4', alpha=0.8)
        )

        # 儲存
        output_path = os.path.join(save_folder, f"{disease_name}_{region}_PM25_lag0_scatter.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"✅ 已輸出：{output_path}")

print("\n🎯 所有疾病的分區散布圖繪製完成！")
