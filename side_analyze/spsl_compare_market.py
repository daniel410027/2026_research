#!/usr/bin/env python3
"""
compare_with_market.py
──────────────────────
1. 讀取 ga_best_result.csv 取得最佳參數，重跑 train/test backtest
2. 讀取 market.csv（加權指數）對齊日期
3. 畫一張合併長圖：train（藍）/ test（綠）策略線 + 市場線（灰）
4. OLS 回歸：策略日報酬 ~ 市場日報酬
   → 印出 alpha / beta / R² / t-stat / p-value（train / test / full 各一組）
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy import stats

warnings.filterwarnings("ignore")

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
PRED_PATH   = "database/experiment_backtest/experiment_merge_prediction.csv"
MARKET_PATH = "database/market/market.csv"
GA_RESULT   = "database/stoplossprofit/ga_best_result.csv"
OUTPUT_DIR  = "database/stoplossprofit"

INITIAL_CAP = 10_000_000
N_STOCKS    = 10
PROC_THRESH = 2_000_000
COMMISSION_RATE = 0.001425
TAX_RATE        = 0.003
MIN_COMMISSION  = 20

SPLITS = {
    "train": ("2016-01-01", "2022-12-31"),
    "test":  ("2023-01-01", "2025-12-31"),
}

COLORS = {
    "train":  "#3B8BD4",   # 藍
    "test":   "#1D9E75",   # 綠
    "market": "#888780",   # 灰
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Backtest（與 stoplossprofit.py 相同核心）────────────────────────────────
def run_backtest(df: pd.DataFrame, sl_pct: float,
                 sp_pct: float, rank_thresh: int) -> pd.Series:
    dates = sorted(df["年月日"].unique())
    if not dates:
        return pd.Series(dtype=float)

    cash, portfolio, pending_buy, pv_list = float(INITIAL_CAP), {}, None, []

    def _buy(alloc, op):
        comm = max(alloc * COMMISSION_RATE, MIN_COMMISSION)
        return {"shares": (alloc - comm) / op, "entry_price": op, "cost": alloc}

    def _sell(pos, sell_at):
        gross = pos["shares"] * sell_at
        comm  = max(gross * COMMISSION_RATE, MIN_COMMISSION)
        return gross - comm - gross * TAX_RATE

    d0 = df[df["年月日"] == dates[0]].set_index("證券代碼").query("`開盤價元` > 0")
    for code in d0.nlargest(N_STOCKS, "y_prob").index:
        alloc = cash / N_STOCKS
        portfolio[code] = _buy(alloc, d0.loc[code, "開盤價元"])
        cash -= alloc

    for date in dates:
        day = df[df["年月日"] == date].copy().set_index("證券代碼")
        day["prob_rank"] = day["y_prob"].rank(ascending=False, method="min")

        if pending_buy is not None:
            buy_cash, cands = pending_buy
            n = min(2 if buy_cash > PROC_THRESH else 1,
                    len([c for c in cands if c in day.index
                         and c not in portfolio
                         and day.loc[c, "開盤價元"] > 0]))
            avail = [c for c in cands if c in day.index
                     and c not in portfolio
                     and day.loc[c, "開盤價元"] > 0][:n]
            if avail:
                per = buy_cash / len(avail)
                for code in avail:
                    portfolio[code] = _buy(per, day.loc[code, "開盤價元"])
                    cash -= per
            else:
                cash += buy_cash
            pending_buy = None

        sell_pool, to_sell = 0.0, []
        for code, pos in portfolio.items():
            if code not in day.index:
                continue
            row   = day.loc[code]
            entry = pos["entry_price"]
            hi, lo, op = row["最高價元"], row["最低價元"], row["開盤價元"]
            sp_p, sl_p = entry * (1 + sp_pct), entry * (1 - sl_pct)
            sell_at = None
            if row["prob_rank"] > rank_thresh:
                sell_at = max(op, 0.01)
            if sell_at is None:
                hit_sp, hit_sl = hi >= sp_p, lo <= sl_p
                if hit_sp and hit_sl:
                    sell_at = sp_p if abs(op - sp_p) <= abs(op - sl_p) else sl_p
                elif hit_sp:
                    sell_at = sp_p
                elif hit_sl:
                    sell_at = sl_p
            if sell_at and sell_at > 0:
                sell_pool += _sell(pos, sell_at)
                to_sell.append(code)

        for code in to_sell:
            del portfolio[code]
        cash += sell_pool

        if sell_pool > 0:
            held = set(portfolio.keys())
            pending_buy = (sell_pool,
                           day[~day.index.isin(held)]
                           .sort_values("y_prob", ascending=False)
                           .index.tolist())

        pv = cash + sum(
            pos["shares"] * (day.loc[c, "收盤價元"]
                             if c in day.index and day.loc[c, "收盤價元"] > 0
                             else pos["entry_price"])
            for c, pos in portfolio.items()
        )
        pv_list.append({"date": date, "value": pv})

    return pd.DataFrame(pv_list).set_index("date")["value"]


# ── OLS 回歸 ─────────────────────────────────────────────────────────────────
def run_ols(strat_ret: pd.Series, mkt_ret: pd.Series,
            label: str) -> dict:
    """日報酬 OLS：strategy ~ alpha + beta * market"""
    merged = pd.concat([strat_ret, mkt_ret], axis=1).dropna()
    merged.columns = ["strat", "mkt"]
    x = merged["mkt"].values
    y = merged["strat"].values
    slope, intercept, r, p_val, se = stats.linregress(x, y)

    # annualise alpha（日 alpha * 252）
    ann_alpha = intercept * 252
    t_stat    = slope / se

    print(f"\n{'─'*55}")
    print(f"  OLS Regression [{label}]  N={len(merged)} days")
    print(f"{'─'*55}")
    print(f"  Alpha (ann.) = {ann_alpha*100:+.2f}%")
    print(f"  Beta         = {slope:.4f}")
    print(f"  R²           = {r**2:.4f}")
    print(f"  t-stat(beta) = {t_stat:.3f}")
    print(f"  p-value      = {p_val:.4f}")
    print(f"{'─'*55}")

    return {
        "label":     label,
        "n":         len(merged),
        "alpha_ann": ann_alpha,
        "beta":      slope,
        "r2":        r**2,
        "t_stat":    t_stat,
        "p_value":   p_val,
    }


# ── 主圖：合併長圖 ────────────────────────────────────────────────────────────
def plot_combined(pv_train: pd.Series, pv_test: pd.Series,
                  mkt_train: pd.Series, mkt_test: pd.Series,
                  best_sl: float, best_sp: float, best_rank: int):

    # 各段以起始點 = 1 做標準化，再銜接成連續線
    def norm(s): return s / s.iloc[0]

    # 策略：train 結尾值 * test 歸一化
    strat_train = norm(pv_train)
    strat_test  = norm(pv_test) * strat_train.iloc[-1]

    # 市場：同樣銜接
    mkt_tr = norm(mkt_train)
    mkt_te = norm(mkt_test) * mkt_tr.iloc[-1]

    all_strat = pd.concat([strat_train, strat_test])
    all_mkt   = pd.concat([mkt_tr, mkt_te])
    split_date = pv_test.index[0]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    # ── 上圖：累積報酬 ────────────────────────────────────────────────────────
    ax1.axvspan(strat_train.index[0], strat_train.index[-1],
                alpha=0.04, color=COLORS["train"], label="_train_bg")
    ax1.axvspan(strat_test.index[0],  strat_test.index[-1],
                alpha=0.04, color=COLORS["test"],  label="_test_bg")
    ax1.axvline(split_date, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax1.axhline(1.0, color="gray", lw=0.6, ls="--", alpha=0.5)

    # 市場線
    ax1.plot(all_mkt.index, all_mkt.values,
             color=COLORS["market"], lw=1.2, alpha=0.7, label="TAIEX")
    # 策略 train
    ax1.plot(strat_train.index, strat_train.values,
             color=COLORS["train"], lw=1.5, label="Strategy (train)")
    # 策略 test
    ax1.plot(strat_test.index, strat_test.values,
             color=COLORS["test"], lw=1.5, label="Strategy (test)")

    ax1.fill_between(all_strat.index, all_strat.values, 1.0,
                     where=all_strat.values < 1.0,
                     alpha=0.12, color="red")

    # 標示 train/test 區間
    ax1.text(strat_train.index[len(strat_train)//2], ax1.get_ylim()[1] if ax1.get_ylim()[1] > 1 else 1.05,
             "TRAIN", ha="center", va="top", fontsize=9,
             color=COLORS["train"], alpha=0.6)
    ax1.text(strat_test.index[len(strat_test)//2], ax1.get_ylim()[1] if ax1.get_ylim()[1] > 1 else 1.05,
             "TEST", ha="center", va="top", fontsize=9,
             color=COLORS["test"], alpha=0.6)

    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax1.set_ylabel("Cumulative Return (base=1)")
    ax1.set_title(
        f"Strategy vs TAIEX  |  SL={best_sl*100:.2f}%  "
        f"SP={best_sp*100:.2f}%  Rank={best_rank}",
        fontsize=11,
    )
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.5)

    # ── 下圖：超額報酬（策略 - 市場）─────────────────────────────────────────
    excess = all_strat - all_mkt
    ax2.axhline(0, color="gray", lw=0.6, ls="--")
    ax2.axvline(split_date, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax2.fill_between(strat_train.index,
                     (strat_train - mkt_tr).values, 0,
                     where=(strat_train - mkt_tr).values >= 0,
                     alpha=0.35, color=COLORS["train"])
    ax2.fill_between(strat_train.index,
                     (strat_train - mkt_tr).values, 0,
                     where=(strat_train - mkt_tr).values < 0,
                     alpha=0.25, color="red")
    ax2.fill_between(strat_test.index,
                     (strat_test - mkt_te).values, 0,
                     where=(strat_test - mkt_te).values >= 0,
                     alpha=0.35, color=COLORS["test"])
    ax2.fill_between(strat_test.index,
                     (strat_test - mkt_te).values, 0,
                     where=(strat_test - mkt_te).values < 0,
                     alpha=0.25, color="red")
    ax2.set_ylabel("Excess Return\n(Strategy − Market)")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "compare_with_market.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"\n  Saved: {out}")


# ── 回歸散點圖 ────────────────────────────────────────────────────────────────
def plot_regression(strat_ret: pd.Series, mkt_ret: pd.Series,
                    ols_res: dict, label: str, color: str):
    merged = pd.concat([strat_ret, mkt_ret], axis=1).dropna()
    merged.columns = ["strat", "mkt"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(merged["mkt"] * 100, merged["strat"] * 100,
               s=6, alpha=0.25, color=color)

    # 回歸線
    x_range = np.linspace(merged["mkt"].min(), merged["mkt"].max(), 200)
    y_range  = ols_res["beta"] * x_range + ols_res["alpha_ann"] / 252
    ax.plot(x_range * 100, y_range * 100, color="black", lw=1.2)

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")

    ax.set_xlabel("Market daily return (%)")
    ax.set_ylabel("Strategy daily return (%)")
    ax.set_title(
        f"OLS Regression [{label}]\n"
        f"α(ann)={ols_res['alpha_ann']*100:+.2f}%  "
        f"β={ols_res['beta']:.3f}  "
        f"R²={ols_res['r2']:.3f}  "
        f"p={ols_res['p_value']:.4f}",
        fontsize=10,
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"regression_{label.lower()}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)

    # ── 讀取最佳參數 ─────────────────────────────────────────────────────────
    if not os.path.exists(GA_RESULT):
        sys.exit(f"[ERROR] {GA_RESULT} 不存在，請先執行 stoplossprofit.py")
    res = pd.read_csv(GA_RESULT).iloc[0]
    best_sl   = float(res["best_sl"])
    best_sp   = float(res["best_sp"])
    best_rank = int(res["best_rank"])
    print(f"Best params: SL={best_sl*100:.2f}%  "
          f"SP={best_sp*100:.2f}%  Rank={best_rank}")

    # ── 讀取預測資料 ─────────────────────────────────────────────────────────
    print("Loading prediction data …")
    df = pd.read_csv(PRED_PATH, parse_dates=["年月日"])
    df = df.dropna(subset=["y_prob", "開盤價元", "最高價元", "最低價元", "收盤價元"])
    df = df[(df["開盤價元"] > 0) & (df["收盤價元"] > 0)].copy()

    # ── 讀取市場資料 ─────────────────────────────────────────────────────────
    print("Loading market data …")
    mkt = pd.read_csv(MARKET_PATH)
    mkt.columns = mkt.columns.str.strip()
    # 年月日可能是整數 20140101 或字串
    mkt["年月日"] = pd.to_datetime(mkt["年月日"].astype(str), format="%Y%m%d")
    mkt = mkt.set_index("年月日").sort_index()
    # 取收盤價欄（第二欄）
    price_col = [c for c in mkt.columns if "收盤" in c or "close" in c.lower()][0]
    mkt_price = mkt[price_col].astype(float)

    # ── 各 split 跑 backtest ─────────────────────────────────────────────────
    pv = {}
    for split, (s, e) in SPLITS.items():
        mask = (df["年月日"] >= s) & (df["年月日"] <= e)
        df_sp = df[mask].reset_index(drop=True)
        print(f"  Running backtest [{split}] {s} ~ {e}  ({mask.sum():,} rows) …")
        pv[split] = run_backtest(df_sp, best_sl, best_sp, best_rank)

    pv_train, pv_test = pv["train"], pv["test"]

    # ── 對齊市場日期 ─────────────────────────────────────────────────────────
    def align_market(pv_s: pd.Series, mkt_p: pd.Series) -> pd.Series:
        idx = pv_s.index
        m   = mkt_p.reindex(idx).ffill()
        return m / m.iloc[0]   # 歸一化

    mkt_train = align_market(pv_train, mkt_price)
    mkt_test  = align_market(pv_test,  mkt_price)

    # ── 日報酬 ───────────────────────────────────────────────────────────────
    strat_ret_train = pv_train.pct_change().dropna()
    strat_ret_test  = pv_test.pct_change().dropna()
    strat_ret_full  = pd.concat([strat_ret_train, strat_ret_test])

    mkt_ret_train = mkt_price.reindex(strat_ret_train.index).pct_change().dropna()
    mkt_ret_test  = mkt_price.reindex(strat_ret_test.index).pct_change().dropna()
    mkt_ret_full  = pd.concat([mkt_ret_train, mkt_ret_test])

    # ── OLS 回歸 ─────────────────────────────────────────────────────────────
    ols_train = run_ols(strat_ret_train, mkt_ret_train, "Train")
    ols_test  = run_ols(strat_ret_test,  mkt_ret_test,  "Test")
    ols_full  = run_ols(strat_ret_full,  mkt_ret_full,  "Full")

    # ── 儲存回歸結果 ─────────────────────────────────────────────────────────
    pd.DataFrame([ols_train, ols_test, ols_full]).to_csv(
        os.path.join(OUTPUT_DIR, "regression_results.csv"), index=False)

    # ── 畫圖 ─────────────────────────────────────────────────────────────────
    print("\nPlotting …")
    plot_combined(pv_train, pv_test, mkt_train, mkt_test,
                  best_sl, best_sp, best_rank)
    plot_regression(strat_ret_train, mkt_ret_train, ols_train,
                    "Train", COLORS["train"])
    plot_regression(strat_ret_test,  mkt_ret_test,  ols_test,
                    "Test",  COLORS["test"])
    plot_regression(strat_ret_full,  mkt_ret_full,  ols_full,
                    "Full",  "#7F77DD")

    print(f"\nAll outputs → {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()