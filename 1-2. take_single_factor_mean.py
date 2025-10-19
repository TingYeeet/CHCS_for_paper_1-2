import os
import pandas as pd
from functools import reduce

# === 1️⃣ 檔案路徑設定 ===
disease_folder = "./0-1_周就醫轉比例_fill"
exposure_folder = "./0-2_exposure_by_town"
cluster_path = "./PM25_manual_cluster_2019.csv"
output_folder = "./1-2_disease_with_PM25_lag0-10_mean"

os.makedirs(output_folder, exist_ok=True)

# === 2️⃣ 讀取手動分群結果 ===
df_cluster = pd.read_csv(cluster_path)
df_cluster["ID"] = df_cluster["ID"].astype(str)

cluster_name_map = {
    1: "高屏",
    2: "雲嘉南",
    3: "苗中彰投",
    4: "北北基桃竹",
    5: "宜花東"
}

# === 3️⃣ 讀取 PM2.5 資料 ===
pm25_file = os.path.join(exposure_folder, "PM25_weekly_exposure_with_ID.csv")
df_pm25 = pd.read_csv(pm25_file)
df_pm25["ID"] = df_pm25["ID"].astype(str)

# 加入群集資訊
df_pm25 = df_pm25.merge(df_cluster, on="ID", how="left")

# === 4️⃣ 計算群平均（每群、年、週） ===
df_pm25_grouped = (
    df_pm25.groupby(["year", "week", "cluster"], as_index=False)["PM25"]
    .mean()
    .round(2)
)
df_pm25_grouped["region"] = df_pm25_grouped["cluster"].map(cluster_name_map)

# === 5️⃣ 幫助計算 lag 的輔助函數 ===
def get_prev_week(year, week, n):
    """
    取得往前 n 週的 (year, week)，會自動跨年
    """
    y, w = year, week - n
    while w <= 0:
        y -= 1
        w += 52
    return y, w

# === 6️⃣ 計算 lag0~lag10 的 PM2.5 平均 ===
N_LAG = 10
df_pm25_grouped = df_pm25_grouped.rename(columns={"PM25": "PM25_lag0"})

for i in range(1, N_LAG + 1):
    lag_map = {}
    for _, row in df_pm25_grouped.iterrows():
        prev_y, prev_w = get_prev_week(int(row["year"]), int(row["week"]), i)
        key = (row["cluster"], int(row["year"]), int(row["week"]))
        lag_value = df_pm25_grouped[
            (df_pm25_grouped["cluster"] == row["cluster"]) &
            (df_pm25_grouped["year"] == prev_y) &
            (df_pm25_grouped["week"] == prev_w)
        ]["PM25_lag0"]
        lag_map[key] = lag_value.values[0] if not lag_value.empty else None

    df_pm25_grouped[f"PM25_lag{i}"] = df_pm25_grouped.apply(
        lambda r: lag_map.get((r["cluster"], int(r["year"]), int(r["week"])), None),
        axis=1
    )

# 保留 lag 欄位
lag_cols = [f"PM25_lag{i}" for i in range(N_LAG + 1)]
df_pm25_grouped = df_pm25_grouped[["cluster", "year", "week", "region", *lag_cols]]

print(f"✅ PM2.5 lag0~10 計算完成，共 {len(df_pm25_grouped)} 筆")

# === 7️⃣ 整合疾病資料 ===
for file in os.listdir(disease_folder):
    if not file.endswith("_filtered.csv"):
        continue

    disease_name = file.replace("_filtered.csv", "")
    print(f"\n=== 處理疾病：{disease_name} ===")

    # 讀取疾病資料
    df_disease = pd.read_csv(os.path.join(disease_folder, file))
    df_disease["ID"] = df_disease["ID1_CITY"].astype(str)

    # 合併 cluster
    df_disease = df_disease.merge(df_cluster, on="ID", how="left")

    # 依群集、年、週加總病例與人口數
    df_disease_grouped = (
        df_disease.groupby(["year", "week", "cluster"], as_index=False)
        .agg({"case_c": "sum", "pop_total": "sum"})
    )

    # 計算每千人病例率
    df_disease_grouped["case_per_capita(‰)"] = (
        df_disease_grouped["case_c"] / df_disease_grouped["pop_total"] * 1000
    ).round(2)

    # 合併 PM2.5 lag 表
    merged = pd.merge(
        df_disease_grouped,
        df_pm25_grouped,
        on=["year", "week", "cluster"],
        how="left"
    )

    # 保留所需欄位
    lag_cols = [f"PM25_lag{i}" for i in range(N_LAG + 1)]
    merged = merged[
        ["region", "year", "week", "case_c", "pop_total", "case_per_capita(‰)", *lag_cols]
    ]

    # 排序與輸出
    merged = merged.sort_values(by=["region", "year", "week"])
    output_path = os.path.join(output_folder, f"{disease_name}_PM25_lag.csv")
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ 已輸出：{output_path}，共 {len(merged)} 筆")

print("\n🎯 所有疾病與 PM2.5 lag0~10 整合完成！")
