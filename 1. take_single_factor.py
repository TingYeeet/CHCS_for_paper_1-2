# 把就診比例資料(0-1_周就診轉比例_fill) -> 和"周就診轉比例"不同處在於有沒有對就診資料中有缺失的部分做補值
# 和空汙資料(0-2_exposure_by_town)合併
# 然後使用PM25_manual_cluster_2019.csv把就診比例資料換算成以大區域為單位的
# 接著捨棄除PM2.5以外的空汙因子，搭配2015年的資料做出延遲0到10周的暴露量累計平均
import os
import pandas as pd
from functools import reduce

# === 1️⃣ 檔案路徑設定 ===
disease_folder = "./0-1_周就醫轉比例_fill"
exposure_folder = "./0-2_exposure_by_town"
cluster_path = "./PM25_manual_cluster_2019.csv"
output_folder = "./1_disease_with_PM25_lag0-10"
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
    """取得往前 n 週的 (year, week)，會自動跨年"""
    y, w = year, week - n
    while w <= 0:
        y -= 1
        w += 52
    return y, w

# === 6️⃣ 計算 lag0~lag10 的 PM2.5 累計加總 ===
N_LAG = 10
df_pm25_grouped = df_pm25_grouped.rename(columns={"PM25": "PM25_lag0"})

for i in range(1, N_LAG + 1):
    lag_sum_map = {}
    for _, row in df_pm25_grouped.iterrows():
        cluster = row["cluster"]
        year = int(row["year"])
        week = int(row["week"])
        # 計算過去 i 週的累計加總
        lag_sum = 0
        for n in range(0, i + 1):
            prev_y, prev_w = get_prev_week(year, week, n)
            val = df_pm25_grouped[
                (df_pm25_grouped["cluster"] == cluster) &
                (df_pm25_grouped["year"] == prev_y) &
                (df_pm25_grouped["week"] == prev_w)
            ]["PM25_lag0"]
            lag_sum += val.values[0] if not val.empty else 0
        lag_sum_map[(cluster, year, week)] = round(lag_sum, 2)

    df_pm25_grouped[f"PM25_lag{i}"] = df_pm25_grouped.apply(
        lambda r: lag_sum_map.get((r["cluster"], int(r["year"]), int(r["week"])), None),
        axis=1
    )

# 保留 lag 欄位
lag_cols = [f"PM25_lag{i}" for i in range(N_LAG + 1)]
df_pm25_grouped = df_pm25_grouped[["cluster", "year", "week", "region", *lag_cols]]

print("✅ PM2.5 lag0~10 計算完成，共", len(df_pm25_grouped), "筆")

# === 7️⃣ 整合疾病資料 ===
for file in os.listdir(disease_folder):
    if not file.endswith("_filtered.csv"):
        continue

    disease_name = file.replace("_filtered.csv", "")
    print(f"\n=== 處理疾病：{disease_name} ===")

    df_disease = pd.read_csv(os.path.join(disease_folder, file))
    df_disease["ID"] = df_disease["ID1_CITY"].astype(str)

    # 合併 cluster
    df_disease = df_disease.merge(df_cluster, on="ID", how="left")

    # 依群集、年、週加總病例與人口數
    df_disease_grouped = (
        df_disease.groupby(["year", "week", "cluster"], as_index=False)
        .agg({"case_c": "sum", "pop_total": "sum"})
    )
    df_disease_grouped["case_per_capita(‰)"] = (
        df_disease_grouped["case_c"] / df_disease_grouped["pop_total"] * 1000
    ).round(2)

    # 合併 lag 表
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

    # 移除 week == 53 的資料
    merged = merged[merged["week"] != 53]

    # 排序與輸出
    merged = merged.sort_values(by=["region", "year", "week"])
    output_path = os.path.join(output_folder, f"{disease_name}_PM25_lag.csv")
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ 已輸出：{output_path}，共 {len(merged)} 筆")

print("\n🎯 所有疾病與 PM2.5 lag0~10 整合完成！")
