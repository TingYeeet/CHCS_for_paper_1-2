import os
import pandas as pd

# === 1️⃣ 路徑設定 ===
input_folder = "./1-3_little_area"
exposure_2015_path = "./0-2_exposure_by_town/PM25_weekly_exposure_with_ID.csv"
output_folder = "./1-4_little_area_lag"
os.makedirs(output_folder, exist_ok=True)

# === 2️⃣ 讀取 2015 年 PM2.5 ===
df_2015 = pd.read_csv(exposure_2015_path)
df_2015["ID"] = df_2015["ID"].astype(str)
df_2015 = df_2015.rename(columns={"PM25": "PM25_2015"})

# === 3️⃣ 輔助函數：取得往前 n 週的 (year, week)，自動跨年 ===
def get_prev_week(year, week, n):
    y, w = year, week - n
    while w <= 0:
        y -= 1
        w += 52
    return y, w

# === 4️⃣ 設定最大 lag 數 ===
N_LAG = 10

# === 5️⃣ 處理每個疾病檔案 ===
for file in os.listdir(input_folder):
    if not file.endswith("_with_pollution.csv"):
        continue

    disease_name = file.replace("_with_pollution.csv", "")
    print(f"\n=== 處理疾病：{disease_name} ===")

    df = pd.read_csv(os.path.join(input_folder, file))
    df["ID"] = df["ID"].astype(str)
    df = df.sort_values(by=["ID", "year", "week"]).reset_index(drop=True)

    # === 🔹 加入 2015 年資料（僅保留需要的 ID 與 PM25）===
    ids_in_disease = df["ID"].unique().tolist()
    df_2015_sub = df_2015[df_2015["ID"].isin(ids_in_disease)].copy()
    df_2015_sub = df_2015_sub.rename(columns={"PM25_2015": "PM25"})
    df_2015_sub["year"] = 2015
    df_2015_sub = df_2015_sub[["ID", "year", "week", "PM25"]]

    # 將 2015 + 疾病資料合併
    df_all = pd.concat([df_2015_sub, df], ignore_index=True)
    df_all = df_all.sort_values(by=["ID", "year", "week"]).reset_index(drop=True)

    # === 計算 lag ===
    df_all["PM25_lag0"] = df_all["PM25"]
    df_lookup = df_all.set_index(["ID", "year", "week"])["PM25"].to_dict()

    for i in range(1, N_LAG + 1):
        lag_vals = []
        for _, row in df_all.iterrows():
            id_ = row["ID"]
            year = int(row["year"])
            week = int(row["week"])
            vals = []
            for n in range(0, i + 1):
                prev_y, prev_w = get_prev_week(year, week, n)
                val = df_lookup.get((id_, prev_y, prev_w), None)
                if val is not None:
                    vals.append(val)
            lag_mean = round(sum(vals) / len(vals), 2) if vals else None
            lag_vals.append(lag_mean)
        df_all[f"PM25_lag{i}"] = lag_vals

    # 只保留疾病年份（從 2016 起）
    df_out = df_all[df_all["year"] >= 2016].copy()
    df_out = df_out.drop(columns=["NO2", "NOx", "NO", "O3", "PM10", "SO2"], errors="ignore")

    # === 輸出 ===
    output_path = os.path.join(output_folder, f"{disease_name}_PM25_lag.csv")
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已輸出：{output_path}（含跨年 lag）")

print("\n🎯 所有疾病的 PM25 lag0~lag10 已計算完成（含 2015 年補足）！")
