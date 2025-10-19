import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from math import atanh, tanh, sqrt

# === 路徑設定 ===
input_folder = "./1_disease_with_PM25_lag0-10"
output_path = "./PM25_lag0_correlation_summary.csv"

# === 儲存結果 ===
results = []

# === 逐一處理每個疾病檔 ===
for file in os.listdir(input_folder):
    if not file.endswith("_PM25_lag.csv"):
        continue

    disease_name = file.replace("_PM25_lag.csv", "")
    df = pd.read_csv(os.path.join(input_folder, file))

    # 檢查必要欄位
    if "PM25_lag0" not in df.columns or "case_per_capita(‰)" not in df.columns:
        print(f"⚠️ {disease_name} 缺少必要欄位，略過")
        continue

    # 移除缺值
    df = df.dropna(subset=["PM25_lag0", "case_per_capita(‰)"])
    if len(df) < 5:
        print(f"⚠️ {disease_name} 資料不足（<5筆），略過")
        continue

    # 計算 Pearson r 與 p-value
    r, p_value = pearsonr(df["PM25_lag0"], df["case_per_capita(‰)"])

    # 計算 95% CI (Fisher z 轉換)
    n = len(df)
    if abs(r) < 1:
        z = atanh(r)
        se = 1 / sqrt(n - 3)
        z_crit = 1.96  # 95% CI
        lo_z, hi_z = z - z_crit * se, z + z_crit * se
        ci_low, ci_high = tanh(lo_z), tanh(hi_z)
    else:
        ci_low, ci_high = r, r

    # R²
    r2 = r ** 2

    # 加入結果
    results.append({
        "疾病名稱": disease_name,
        "相關係數(r)": round(r, 3),
        "95% CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
        "p值": f"{p_value:.3e}",
        "R²": round(r2, 3),
        "樣本數": n
    })

# === 匯出結果 ===
df_result = pd.DataFrame(results)
df_result = df_result.sort_values(by="相關係數(r)", ascending=False)
df_result.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\n✅ 已完成 PM25_lag0 統計相關性分析")
print(f"📄 結果已輸出：{output_path}")
print(df_result)
