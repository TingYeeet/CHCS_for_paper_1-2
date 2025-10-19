import os
import pandas as pd

# === 路徑設定 ===
disease_folder = "./0-1_周就醫轉比例_fill"
pollution_folder = "./0-2_exposure_by_town"
output_folder = "./1-3_little_area"
os.makedirs(output_folder, exist_ok=True)

# === 讀取所有空汙檔案 ===
pollution_dfs = []
for file in os.listdir(pollution_folder):
    if not file.endswith("_weekly_exposure_with_ID.csv"):
        continue

    pollutant_name = file.split("_")[0]  # 例如 "NO" or "NO2"
    df_pollution = pd.read_csv(os.path.join(pollution_folder, file))

    # 只保留必要欄位並重新命名
    df_pollution = df_pollution[["ID", "year", "week", pollutant_name]]
    pollution_dfs.append(df_pollution)

# 將所有空汙資料依 ID,year,week 合併
df_pollution_all = pollution_dfs[0]
for df_p in pollution_dfs[1:]:
    df_pollution_all = pd.merge(df_pollution_all, df_p, on=["ID", "year", "week"], how="outer")

print(f"✅ 已整合空汙資料，共 {len(pollution_dfs)} 種測項")

# === 讀取每個疾病檔案並合併 ===
for file in os.listdir(disease_folder):
    if not file.endswith("_filtered.csv"):
        continue

    disease_name = file.replace("_filtered.csv", "")
    df_disease = pd.read_csv(os.path.join(disease_folder, file))
    df_disease = df_disease.rename(columns={"ID1_CITY": "ID"})

    # 合併疾病資料與空汙資料
    df_merged = pd.merge(
        df_disease,
        df_pollution_all,
        on=["ID", "year", "week"],
        how="inner"  # 只保留兩者皆有資料的週
    )

    # 儲存
    output_path = os.path.join(output_folder, f"{disease_name}_with_pollution.csv")
    df_merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ 已輸出：{output_path} (共 {len(df_merged)} 筆)")

print("\n🎯 所有疾病與空汙資料整合完成！")
