"""
check_d10.py
============

【功能】
針對 D10 的 PnL 異常進行深入診斷，涵蓋：

  1. D10 daily PnL 分布（是否有極端值拉高均值）
  2. D10 vs D01 報酬對稱性（L/S 有效性）
  3. D10 個股層級報酬分布（異常股是誰在拉？）
  4. D10 每日持股數穩定性（分組是否正常）
  5. D10 報酬來源拆解：極端報酬貢獻 vs 正常報酬貢獻
  6. D10 超額報酬（vs 市場均值）是否持續
  7. D10 按年分層分析（是否某年特別異常）
  8. D10 前 N% 貢獻者識別（哪幾筆交易撐起整體）
  9. y_prob 在 D10 的分布（是否幾乎都擠在邊界？）
 10. D10 PnL 的自相關性（alpha 是否有持續性）

【使用方式】
  python check_d10.py

  預設讀取：
    ./database/experiment_backtest/experiment_merge_prediction.csv
    ./database/experiment_backtest/daily_pnl.csv

  若路徑不同，請修改下方 CONFIG。
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
BASE_DIR   = Path(__file__).resolve().parent
PRED_PATH  = BASE_DIR / 'database' / 'experiment_backtest' / 'experiment_merge_prediction.csv'
PNL_PATH   = BASE_DIR / 'database' / 'experiment_backtest' / 'daily_pnl.csv'

LIMIT_THR  = 0.095   # 台股漲跌停門檻
TOP_N_PCT  = 0.05    # 「前 N% 貢獻者」門檻
LAG_MAX    = 10      # 自相關檢查最大 lag（天）

SEP = "─" * 60


def sep(title=''):
    if title:
        print(f"\n{'═'*60}")
        print(f"  {title}")
        print(f"{'═'*60}")
    else:
        print(SEP)


# ══════════════════════════════════════════════════════════════════
# 讀資料 & 重新分 Decile
# ══════════════════════════════════════════════════════════════════

def load_and_assign_decile(pred_path: Path, pnl_path: Path):
    print(f"📂 讀取 {pred_path.name} ...")
    df = pd.read_csv(pred_path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['年月日'].astype(str).str.zfill(8),
                                 infer_datetime_format=True)

    # 重新逐日分 decile（與 backtest 邏輯一致）
    result = []
    for d, g in df.groupby('date'):
        if len(g) < 10:
            continue
        g = g.copy()
        g['decile'] = pd.qcut(g['y_prob'], 10,
                               labels=range(1, 11), duplicates='drop')
        result.append(g)
    df = pd.concat(result).reset_index(drop=True)
    df['decile'] = df['decile'].astype(int)

    print(f"📂 讀取 {pnl_path.name} ...")
    pnl = pd.read_csv(pnl_path, encoding='utf-8-sig', parse_dates=['date'])

    return df, pnl


# ══════════════════════════════════════════════════════════════════
# Check 1  D10 daily PnL 分布
# ══════════════════════════════════════════════════════════════════

def check_daily_pnl_distribution(pnl: pd.DataFrame):
    sep("CHECK 1  D10 Daily PnL 分布")

    d10 = pnl[pnl['decile'] == 10]['pnl'].dropna()
    d01 = pnl[pnl['decile'] ==  1]['pnl'].dropna()

    for label, s in [('D10', d10), ('D01', d01)]:
        print(f"\n  {label}:")
        print(f"    count  : {len(s)}")
        print(f"    mean   : {s.mean():.6f}  ({s.mean()*252:.2%}/yr annualized)")
        print(f"    median : {s.median():.6f}")
        print(f"    std    : {s.std():.6f}")
        print(f"    skew   : {stats.skew(s):.4f}")
        print(f"    kurt   : {stats.kurtosis(s):.4f}")
        print(f"    min    : {s.min():.6f}  max: {s.max():.6f}")

        # 極端值比例
        for thr in [0.05, 0.10, 0.20]:
            pct_above = (s > thr).mean()
            pct_below = (s < -thr).mean()
            print(f"    > {thr:.0%}: {pct_above:.2%}   < -{thr:.0%}: {pct_below:.2%}")

    # mean vs median 差距警告
    gap = abs(d10.mean() - d10.median())
    print(f"\n  ⚠ D10 mean-median gap = {gap:.6f}")
    if gap > 0.003:
        print("    → mean 遠高於 median，報酬可能由少數極端交易日撐起")
    else:
        print("    → ✅ mean ≈ median，分布較對稱")


# ══════════════════════════════════════════════════════════════════
# Check 2  D10 vs D01 對稱性
# ══════════════════════════════════════════════════════════════════

def check_d10_d01_symmetry(pnl: pd.DataFrame):
    sep("CHECK 2  D10 vs D01 報酬對稱性")

    d10 = pnl[pnl['decile'] == 10].set_index('date')['pnl']
    d01 = pnl[pnl['decile'] ==  1].set_index('date')['pnl']
    common = d10.index.intersection(d01.index)
    d10, d01 = d10.loc[common], d01.loc[common]

    ls = (d10 - d01) / 2
    print(f"\n  L/S (D10-D01)/2:")
    print(f"    mean  : {ls.mean():.6f}  ({ls.mean()*252:.2%}/yr)")
    print(f"    Sharpe: {ls.mean() / ls.std() * np.sqrt(252):.4f}")

    corr = d10.corr(d01)
    print(f"\n  D10 vs D01 相關係數 = {corr:.4f}")
    if corr > 0.5:
        print("  ⚠ 兩組報酬高度正相關 → L/S 效果有限，因子分層能力弱")
    elif corr < -0.1:
        print("  ✅ 兩組報酬負相關 → L/S 對沖效果良好")
    else:
        print("  ℹ 相關性低 → 兩組報酬幾乎獨立")

    # 同方向日數
    same_dir = ((d10 > 0) & (d01 > 0)) | ((d10 < 0) & (d01 < 0))
    print(f"\n  D10 & D01 同向上漲/下跌日比例 = {same_dir.mean():.2%}")
    if same_dir.mean() > 0.6:
        print("  ⚠ 兩組大多同向，市場 beta 效應主導，alpha 不純")


# ══════════════════════════════════════════════════════════════════
# Check 3  D10 個股層級報酬分布
# ══════════════════════════════════════════════════════════════════

def check_stock_level(df: pd.DataFrame):
    sep("CHECK 3  D10 個股層級報酬分布")

    d10_stocks = df[df['decile'] == 10].copy()
    print(f"\n  D10 個股觀察數 : {len(d10_stocks):,}")
    print(f"  D10 唯一股票數 : {d10_stocks['證券代碼'].nunique():,}")

    ret = d10_stocks['return']
    print(f"\n  return 分布：")
    print(f"    mean   : {ret.mean():.6f}")
    print(f"    median : {ret.median():.6f}")
    print(f"    std    : {ret.std():.6f}")
    print(f"    p5     : {ret.quantile(0.05):.6f}")
    print(f"    p25    : {ret.quantile(0.25):.6f}")
    print(f"    p75    : {ret.quantile(0.75):.6f}")
    print(f"    p95    : {ret.quantile(0.95):.6f}")
    print(f"    max    : {ret.max():.6f}")
    print(f"    min    : {ret.min():.6f}")

    # 極端漲幅貢獻
    n_total  = len(ret)
    for thr in [LIMIT_THR, 0.15, 0.20]:
        n_above = (ret > thr).sum()
        sum_above = ret[ret > thr].sum()
        sum_total = ret.sum()
        pct_obs  = n_above / n_total
        pct_contr = sum_above / sum_total if sum_total != 0 else np.nan
        print(f"\n  return > {thr:.0%}：{n_above} 筆 ({pct_obs:.2%} of obs)")
        print(f"    這些筆的 return 合計 = {sum_above:.4f}")
        print(f"    佔全部 D10 return 總和的 {pct_contr:.2%}")
        if pct_contr > 0.5:
            print(f"    ⚠ 超過一半的 D10 總報酬來自 return > {thr:.0%} 的極端股")

    # 個股平均報酬排名
    stock_avg = d10_stocks.groupby('證券代碼')['return'].agg(['mean', 'count'])
    stock_avg = stock_avg[stock_avg['count'] >= 5].sort_values('mean', ascending=False)
    print(f"\n  個股平均報酬 Top 15（至少出現 5 次）：")
    print(f"    {'代碼':>8}  {'mean':>9}  {'出現次數':>6}")
    for code, row in stock_avg.head(15).iterrows():
        print(f"    {code:>8}  {row['mean']:>9.4%}  {int(row['count']):>6}")


# ══════════════════════════════════════════════════════════════════
# Check 4  D10 每日持股數穩定性
# ══════════════════════════════════════════════════════════════════

def check_daily_n_stocks(pnl: pd.DataFrame, df: pd.DataFrame):
    sep("CHECK 4  D10 每日持股數穩定性")

    n_by_date = pnl[pnl['decile'] == 10][['date', 'n_stocks']].set_index('date')['n_stocks']
    print(f"\n  持股數統計：")
    print(f"    mean   : {n_by_date.mean():.1f}")
    print(f"    median : {n_by_date.median():.1f}")
    print(f"    std    : {n_by_date.std():.1f}")
    print(f"    min    : {n_by_date.min()}")
    print(f"    max    : {n_by_date.max()}")

    tiny_days = (n_by_date < 5).sum()
    print(f"\n  持股數 < 5 的交易日 : {tiny_days} 天")
    if tiny_days > 0:
        print("  ⚠ 部分交易日 D10 持股過少，均值報酬不穩定")

    # 持股數異常低的日期
    low_days = n_by_date[n_by_date < n_by_date.quantile(0.05)]
    if len(low_days) > 0:
        print(f"\n  持股數最低的 5% 交易日 ({len(low_days)} 天)：")
        for d, n in low_days.sort_values().head(10).items():
            print(f"    {d.date()}  n={n}")


# ══════════════════════════════════════════════════════════════════
# Check 5  D10 報酬來源拆解：去掉極端後的 Sharpe
# ══════════════════════════════════════════════════════════════════

def check_return_source(pnl: pd.DataFrame):
    sep("CHECK 5  D10 報酬來源拆解（去極端值）")

    d10 = pnl[pnl['decile'] == 10]['pnl'].dropna()

    base_sharpe = d10.mean() / d10.std() * np.sqrt(252)
    print(f"\n  原始 Sharpe = {base_sharpe:.4f}")

    for thr in [0.05, 0.08, 0.10, 0.15]:
        clipped = d10.clip(upper=thr)
        s = clipped.mean() / clipped.std() * np.sqrt(252) if clipped.std() > 0 else np.nan
        drop = base_sharpe - s
        print(f"  PnL clip(upper={thr:.0%})  →  Sharpe = {s:.4f}  (下降 {drop:.4f})")
        if s <= 0 and base_sharpe > 0:
            print(f"    ⚠ clip 後 Sharpe 轉負 → alpha 幾乎完全依賴 return > {thr:.0%} 的日子")

    for thr in [0.05, 0.10]:
        trimmed = d10[(d10 > -thr) & (d10 < thr)]
        pct_kept = len(trimmed) / len(d10)
        s = trimmed.mean() / trimmed.std() * np.sqrt(252) if trimmed.std() > 0 else np.nan
        print(f"\n  去除 |pnl| > {thr:.0%}（保留 {pct_kept:.1%} 交易日）  →  Sharpe = {s:.4f}")


# ══════════════════════════════════════════════════════════════════
# Check 6  D10 超額報酬 vs 市場
# ══════════════════════════════════════════════════════════════════

def check_excess_return(pnl: pd.DataFrame, df: pd.DataFrame):
    sep("CHECK 6  D10 超額報酬（vs 市場均值）")

    d10 = pnl[pnl['decile'] == 10].set_index('date')['pnl']

    mkt = df.groupby('date')['return'].mean()
    common = d10.index.intersection(mkt.index)
    d10, mkt = d10.loc[common], mkt.loc[common]
    alpha = d10 - mkt

    print(f"\n  D10 mean         : {d10.mean():.6f}  ({d10.mean()*252:.2%}/yr)")
    print(f"  Market mean      : {mkt.mean():.6f}  ({mkt.mean()*252:.2%}/yr)")
    print(f"  Alpha mean       : {alpha.mean():.6f}  ({alpha.mean()*252:.2%}/yr)")
    print(f"  Alpha Sharpe     : {alpha.mean()/alpha.std()*np.sqrt(252):.4f}")

    # t 檢定
    t, p = stats.ttest_1samp(alpha.dropna(), 0)
    print(f"  Alpha t-stat     : {t:.4f}  p={p:.4f}")
    if p < 0.05:
        print("  ✅ D10 alpha 顯著異於零（p < 0.05）")
    else:
        print("  ⚠ D10 alpha 不顯著，可能只是 beta 暴露")

    # correlation with market
    corr = d10.corr(mkt)
    print(f"  D10 vs Mkt corr  : {corr:.4f}")
    if corr > 0.8:
        print("  ⚠ D10 與市場高度同向，策略以 beta 暴露為主，alpha 不明顯")


# ══════════════════════════════════════════════════════════════════
# Check 7  D10 按年分層
# ══════════════════════════════════════════════════════════════════

def check_yearly(pnl: pd.DataFrame):
    sep("CHECK 7  D10 按年分層分析")

    d10 = pnl[pnl['decile'] == 10].copy()
    d10['year'] = d10['date'].dt.year

    yearly = d10.groupby('year')['pnl'].agg(
        mean='mean', std='std', n='count',
        pos_rate=lambda x: (x > 0).mean()
    )
    yearly['sharpe'] = yearly['mean'] / yearly['std'] * np.sqrt(252)
    yearly['ann_ret'] = yearly['mean'] * 252

    print(f"\n  {'Year':>6}  {'AnnRet':>8}  {'Sharpe':>8}  {'WinRate':>8}  {'Days':>5}")
    print(f"  {SEP}")
    for yr, row in yearly.iterrows():
        flag = '  ⚠' if row['sharpe'] < 0 else ''
        print(f"  {yr:>6}  {row['ann_ret']:>8.2%}  {row['sharpe']:>8.3f}  "
              f"{row['pos_rate']:>8.2%}  {int(row['n']):>5}{flag}")

    # 找出貢獻最大的年份
    cum_by_year = d10.groupby('year')['pnl'].sum()
    top_yr = cum_by_year.idxmax()
    print(f"\n  PnL 累積貢獻最大年份 : {top_yr}  ({cum_by_year[top_yr]:.4f})")
    pct = cum_by_year[top_yr] / cum_by_year.sum() * 100
    print(f"  佔全期 D10 總 PnL 的 {pct:.1f}%")
    if pct > 50:
        print("  ⚠ 超過一半的 D10 PnL 來自單一年份，結果穩健性存疑")


# ══════════════════════════════════════════════════════════════════
# Check 8  D10 前 N% 貢獻者識別
# ══════════════════════════════════════════════════════════════════

def check_top_contributors(pnl: pd.DataFrame):
    sep(f"CHECK 8  D10 前 {TOP_N_PCT:.0%} 貢獻日識別")

    d10 = pnl[pnl['decile'] == 10].set_index('date')['pnl'].dropna().sort_values(ascending=False)
    n_top = max(1, int(len(d10) * TOP_N_PCT))

    top = d10.head(n_top)
    rest = d10.iloc[n_top:]

    print(f"\n  全期交易日數  : {len(d10)}")
    print(f"  前 {TOP_N_PCT:.0%} 交易日 : {n_top} 天")
    print(f"\n  前 {TOP_N_PCT:.0%} 日 PnL 合計  : {top.sum():.6f}")
    print(f"  其餘日  PnL 合計  : {rest.sum():.6f}")
    pct = top.sum() / d10.sum() * 100 if d10.sum() != 0 else np.nan
    print(f"  前 {TOP_N_PCT:.0%} 日貢獻佔比    : {pct:.1f}%")
    if pct > 80:
        print(f"  ⚠ {pct:.0f}% 的總報酬集中在 {TOP_N_PCT:.0%} 的交易日，策略極不穩定")

    print(f"\n  前 10 高報酬交易日：")
    print(f"    {'Date':>12}  {'PnL':>9}")
    for d, v in top.head(10).items():
        print(f"    {str(d.date()):>12}  {v:>9.4%}")


# ══════════════════════════════════════════════════════════════════
# Check 9  y_prob 在 D10 的分布
# ══════════════════════════════════════════════════════════════════

def check_yprob_distribution(df: pd.DataFrame):
    sep("CHECK 9  y_prob 在 D10 的分布")

    d10 = df[df['decile'] == 10]['y_prob']
    all_stocks = df['y_prob']

    print(f"\n  全體 y_prob：")
    print(f"    mean={all_stocks.mean():.4f}  std={all_stocks.std():.4f}")
    print(f"    min={all_stocks.min():.4f}  max={all_stocks.max():.4f}")

    print(f"\n  D10 y_prob：")
    print(f"    mean={d10.mean():.4f}  std={d10.std():.4f}")
    print(f"    min={d10.min():.4f}  max={d10.max():.4f}")
    print(f"    p90={d10.quantile(0.90):.4f}  p99={d10.quantile(0.99):.4f}")

    # 確認 D10 是高 y_prob 還是低 y_prob
    d01_mean = df[df['decile'] == 1]['y_prob'].mean()
    print(f"\n  D01 y_prob mean = {d01_mean:.4f}")
    print(f"  D10 y_prob mean = {d10.mean():.4f}")
    if d10.mean() < d01_mean:
        print("  ⚠ D10 y_prob < D01 y_prob → 因子方向反轉！")
        print("    模型預測低分 = 高報酬，請確認因子方向是否符合預期")
    else:
        print("  ✅ D10 y_prob > D01 → 因子方向正常（高預測分數 = 高報酬）")

    # 每日 D10 y_prob 切點統計
    daily_d10_min = df[df['decile'] == 10].groupby('date')['y_prob'].min()
    print(f"\n  每日 D10 的最低 y_prob（進入門檻）：")
    print(f"    mean={daily_d10_min.mean():.4f}  std={daily_d10_min.std():.4f}")
    print(f"    min={daily_d10_min.min():.4f}  max={daily_d10_min.max():.4f}")


# ══════════════════════════════════════════════════════════════════
# Check 10  D10 PnL 自相關性（持續性）
# ══════════════════════════════════════════════════════════════════

def check_autocorrelation(pnl: pd.DataFrame):
    sep("CHECK 10  D10 PnL 自相關性（alpha 持續性）")

    d10 = pnl[pnl['decile'] == 10].set_index('date')['pnl'].dropna().sort_index()

    print(f"\n  Lag  AutoCorr  Interpretation")
    print(f"  {SEP}")
    for lag in range(1, LAG_MAX + 1):
        ac = d10.autocorr(lag=lag)
        bar = '█' * int(abs(ac) * 30) if not np.isnan(ac) else ''
        sign = '+' if ac > 0 else '-'
        print(f"  {lag:>3}   {ac:>7.4f}   {sign}{bar}")

    # Ljung-Box 概略檢定（用 lag=5 的 AC 作簡單判斷）
    ac1 = d10.autocorr(1)
    if abs(ac1) > 0.1:
        print(f"\n  ⚠ lag-1 自相關 = {ac1:.4f}，PnL 可能有短期慣性/均值回歸效應")
    else:
        print(f"\n  ✅ lag-1 自相關 = {ac1:.4f}，無明顯序列相關")


# ══════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  D10 PnL 診斷工具")
    print("=" * 60)

    if not PRED_PATH.exists():
        print(f"❌ 找不到 {PRED_PATH}")
        print("   請確認路徑，或先執行 backtest_with_sanity.py 生成檔案")
        return

    df, pnl = load_and_assign_decile(PRED_PATH, PNL_PATH)

    check_daily_pnl_distribution(pnl)
    check_d10_d01_symmetry(pnl)
    check_stock_level(df)
    check_daily_n_stocks(pnl, df)
    check_return_source(pnl)
    check_excess_return(pnl, df)
    check_yearly(pnl)
    check_top_contributors(pnl)
    check_yprob_distribution(df)
    check_autocorrelation(pnl)

    print(f"\n{'═'*60}")
    print("  診斷完成")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()