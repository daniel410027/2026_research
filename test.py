"""
backtest_test.py
================
診斷原版 backtest.py 的 D1 報酬是否合理。
"""
from pathlib import Path
import pandas as pd
import numpy as np

EXPERIMENT_DIR = Path("database/experiment")
DATE_COL = "年月日"
ID_COL   = "證券代碼"

def load_predictions(exp_dir):
    frames = []
    for path in sorted(exp_dir.glob("*/predictions.csv")):
        df = pd.read_csv(path, low_memory=False)
        df["_window"] = path.parent.name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined[DATE_COL] = pd.to_datetime(combined[DATE_COL])
    combined[ID_COL]   = combined[ID_COL].astype(str)
    combined = combined.sort_values([ID_COL, DATE_COL, "_window"])
    combined = combined.drop_duplicates(subset=[ID_COL, DATE_COL], keep="first")
    return combined.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

def assign_deciles(df, n=10):
    df = df.copy().reset_index(drop=True)
    df["_rank"] = df.groupby(DATE_COL)["y_prob"].rank(method="first", ascending=False)
    df["_size"] = df.groupby(DATE_COL)["y_prob"].transform("count")
    mask = df["_size"] >= n
    df["decile"] = np.nan
    df.loc[mask, "decile"] = np.ceil(df.loc[mask, "_rank"] / df.loc[mask, "_size"] * n).clip(1, n)
    return df.drop(columns=["_rank", "_size"])

print("► 載入 predictions...")
df = load_predictions(EXPERIMENT_DIR)
print(f"  {len(df):,} 筆")

print("\n► 【return 欄位全體分布】")
r = df["return"]
print(r.describe(percentiles=[.001,.01,.05,.25,.5,.75,.95,.99,.999]))
print(f"  超過 10% 的筆數：{(r > 0.10).sum():,} ({(r > 0.10).mean():.2%})")
print(f"  超過 50% 的筆數：{(r > 0.50).sum():,} ({(r > 0.50).mean():.2%})")
print(f"  低於 -10% 的筆數：{(r < -0.10).sum():,} ({(r < -0.10).mean():.2%})")

print("\n► 【分 Decile 後 D1 的 return 分布】")
df = assign_deciles(df)
d1 = df[df["decile"] == 1]["return"]
print(d1.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))
print(f"  D1 平均日報酬：{d1.mean():.6f}  ({d1.mean()*100:.4f}%)")
print(f"  D1 年化驗算：{(1+d1.mean())**252 - 1:.4f}  ({((1+d1.mean())**252 - 1)*100:.1f}%)")

print("\n► 【各 Decile 平均日報酬】")
dec_ret = df.groupby("decile")["return"].agg(["mean","median","std","count"])
dec_ret["ann_approx"] = (1 + dec_ret["mean"])**252 - 1
print(dec_ret.round(6).to_string())

print("\n► 【return 極端值樣本（前20大）】")
top = df.nlargest(20, "return")[[ID_COL, DATE_COL, "return", "y_prob", "_window"]]
print(top.to_string(index=False))

print("\n► 【同一股票連續日 return 是否重疊（抽查10支）】")
sample_ids = df[ID_COL].drop_duplicates().sample(10, random_state=42).tolist()
for sid in sample_ids:
    sub = df[df[ID_COL] == sid].sort_values(DATE_COL)[["年月日","return"]].head(5)
    dates = sub["年月日"].tolist()
    rets  = sub["return"].tolist()
    # 連續兩天的報酬如果幾乎相同，可能有重複計算
    if len(rets) >= 2:
        diffs = [abs(rets[i] - rets[i-1]) for i in range(1, len(rets))]
        flag = "⚠ 可能重複" if max(diffs) < 1e-8 else "OK"
        print(f"  {sid}: {[f'{v:.4f}' for v in rets]}  {flag}")