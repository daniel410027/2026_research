"""
record_portfolio.py
===================
從 final_day_prediction_sort.csv 撈出 stocklist.csv 內所有股票的預測資訊，
寫入 portfolio_record.csv（只保留最新一天的紀錄）。

輸出欄位：證券代碼, 年月日, pred_score, rank, is_selected
"""

import pandas as pd
from pathlib import Path

# ── 路徑設定 ─────────────────────────────────────────────────
STOCKLIST_PATH  = Path("stocklist.csv")
PREDICTION_PATH = Path("database/daily_predict/final_day_prediction_sort.csv")
RECORD_PATH     = Path("portfolio_record.csv")

# ── 讀取 stocklist ───────────────────────────────────────────
stocklist = pd.read_csv(STOCKLIST_PATH, dtype={"證券代碼": str})
if "證券代碼" not in stocklist.columns:
    raise ValueError("stocklist.csv 缺少 '證券代碼' 欄位")

target_stocks = stocklist["證券代碼"].str.strip().tolist()
print(f"  stocklist 股票數: {len(target_stocks)}")

# ── 讀取預測結果 ─────────────────────────────────────────────
if not PREDICTION_PATH.exists():
    raise FileNotFoundError(f"找不到預測檔: {PREDICTION_PATH}")

pred_df = pd.read_csv(PREDICTION_PATH, dtype={"證券代碼": str})
pred_df["證券代碼"] = pred_df["證券代碼"].str.strip()

# ── 篩選目標股票 ─────────────────────────────────────────────
cols = ["證券代碼", "年月日", "pred_score", "rank", "is_selected"]
missing_cols = [c for c in cols if c not in pred_df.columns]
if missing_cols:
    raise ValueError(f"預測檔缺少欄位: {missing_cols}")

result = pred_df[pred_df["證券代碼"].isin(target_stocks)][cols].copy()

not_found = set(target_stocks) - set(result["證券代碼"])
if not_found:
    print(f"  ⚠ 以下股票在預測檔中找不到: {sorted(not_found)}")

today = result["年月日"].iloc[0] if len(result) else None
print(f"  命中 {len(result)} 筆（{today}）")

# ── 覆寫（只保留最新日期，捨棄舊日紀錄）────────────────────
if RECORD_PATH.exists():
    existing = pd.read_csv(RECORD_PATH, dtype={"證券代碼": str})
    # 移除與本次同股票的舊紀錄，再合併（保留非本次日期的其他股票資料若有需要可移除此行）
    existing = existing[existing["年月日"] == today]  # ← 只留最新日期
    combined = pd.concat([existing, result], ignore_index=True)
    combined = combined.drop_duplicates(subset=["證券代碼", "年月日"], keep="last")
else:
    combined = result

combined = combined.sort_values(["年月日", "rank"]).reset_index(drop=True)
combined.to_csv(RECORD_PATH, index=False, encoding="utf-8-sig")

print(f"  ✓ 已寫入 {RECORD_PATH}（累計 {len(combined)} 筆）")
print(result[["證券代碼", "年月日", "pred_score", "rank", "is_selected"]].to_string(index=False))