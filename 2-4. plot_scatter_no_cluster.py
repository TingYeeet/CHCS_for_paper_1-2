import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 字體設定（支援中文）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# === 路徑設定 ===
input_folder = "./1-3_little_area"
output_folder = "./2-4_disease_PM25_scatter_no_cluster"
os.makedirs(output_folder, exist_ok=True)

# === 疾病分類 ===
upper_respiratory = ["URI", "急性Rhinosinusitis", "Allergic rhinitis", "Influenza"]
lower_respiratory = ["急性Bronchitis", "慢性Bronchitis", "Pneumonia", "氣喘"]

# === 顏色設定 ===
disease_colors = {
    "URI": "#1f77b4",                # 藍
    "急性Rhinosinusitis": "#ff7f0e", # 橘
    "Allergic rhinitis": "#2ca02c",  # 綠
    "Influenza": "#d62728",          # 紅
    "急性Bronchitis": "#9467bd",     # 紫
    "慢性Bronchitis": "#8c564b",     # 棕
    "Pneumonia": "#e377c2",          # 粉
    "氣喘": "#17becf"                # 青
}

# === 畫圖函數 ===
def plot_group(disease_list, title, save_path):
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=14)
    plt.xlabel("PM2.5 暴露濃度 (μg/m³)", fontsize=12)
    plt.ylabel("就診率 (‰)", fontsize=12)
    plt.grid(alpha=0.3)

    for disease in disease_list:
        file_path = os.path.join(input_folder, f"{disease}_with_pollution.csv")
        if not os.path.exists(file_path):
            print(f"⚠️ 找不到檔案：{file_path}")
            continue

        df = pd.read_csv(file_path)
        df = df.dropna(subset=["PM25", "case_per_capita(‰)"])

        color = disease_colors.get(disease, "gray")
        plt.scatter(df["PM25"], df["case_per_capita(‰)"],
                    label=disease, alpha=0.6, color=color)

        # 加上回歸線
        slope, intercept, r, p, _ = linregress(df["PM25"], df["case_per_capita(‰)"])
        x_vals = sorted(df["PM25"])
        y_vals = [slope * x + intercept for x in x_vals]
        plt.plot(x_vals, y_vals, color=color, linestyle="-", linewidth=1.5)
    
    plt.legend(title="疾病名稱", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 已輸出：{save_path}")

# === 畫圖 ===
plot_group(
    upper_respiratory,
    "上呼吸道疾病 PM2.5 與就診率散布圖（全台）",
    os.path.join(output_folder, "upper_respiratory_PM25_scatter.png")
)

plot_group(
    lower_respiratory,
    "下呼吸道疾病 PM2.5 與就診率散布圖（全台）",
    os.path.join(output_folder, "lower_respiratory_PM25_scatter.png")
)

print("\n🎯 所有散布圖繪製完成！")
