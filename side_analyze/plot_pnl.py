"""
plot_pnl.py
===========
條件交易策略 PnL 視覺化。

讀取 strategy.py 輸出的 predictions.csv，
搭配 market.csv 大盤指數，繪製：
    1. 多方組合（y_pred=1）累積報酬
    2. 空方組合（y_pred=0）累積報酬
    3. 大盤加權指數累積報酬
    4. 多空價差（Long - Short spread）累積報酬

【使用方式】
    python plot_pnl.py
    # 或指定路徑
    python plot_pnl.py --pred_dir database/condition_trade --market database/market/market.csv

作者：Daniel Huang
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================
#  載入函式
# ============================================================

def load_predictions(pred_dir: Path) -> pd.DataFrame:
    """
    遞迴搜尋 pred_dir 下所有 predictions.csv，合併後回傳。
    欄位：證券代碼, 年月日(datetime), return, y_true, y_prob, y_pred
    """
    files = sorted(pred_dir.rglob("predictions.csv"))
    if not files:
        raise FileNotFoundError(f"找不到 predictions.csv：{pred_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, dtype={"年月日": str})
        dfs.append(df)
        print(f"  ✓ 載入 {f}  ({len(df):,} 筆)")

    df = pd.concat(dfs, ignore_index=True)
    df["年月日"] = pd.to_datetime(df["年月日"], format="%Y%m%d")
    df = df.sort_values("年月日").reset_index(drop=True)
    print(f"  合計 {len(df):,} 筆 | 日期：{df['年月日'].min().date()} ~ {df['年月日'].max().date()}")
    return df


def load_market(market_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    載入大盤指數，回傳日報酬 Series（index = 日期）。
    格式：證券代碼, 年月日, 收盤價(元)
    """
    mkt = pd.read_csv(market_path, dtype={"年月日": str})
    mkt = mkt[mkt["證券代碼"].str.contains("Y9999|加權", na=False)].copy()
    mkt["年月日"] = pd.to_datetime(mkt["年月日"], format="%Y%m%d")
    mkt = mkt.sort_values("年月日")
    price_col = [c for c in mkt.columns if "收盤" in c or "close" in c.lower()][0]
    mkt = mkt[(mkt["年月日"] >= start) & (mkt["年月日"] <= end)]
    mkt["mkt_return"] = mkt[price_col].pct_change()
    mkt = mkt.dropna(subset=["mkt_return"])
    mkt = mkt.set_index("年月日")["mkt_return"]
    return mkt


# ============================================================
#  計算每日組合報酬
# ============================================================

def daily_portfolio_return(
    df: pd.DataFrame,
    mask: pd.Series,
    label: str = "",
) -> pd.Series:
    """
    mask 篩選後，每日截面等權平均報酬。
    回傳 pd.Series，index = 年月日（datetime），name = label。
    """
    subset = df[mask].copy()
    daily = subset.groupby("年月日")["return"].mean()
    daily.name = label
    return daily


# ============================================================
#  繪圖
# ============================================================

def plot_pnl(
    pred_dir: str = "database/condition_trade",
    market_path: str = "database/market/market.csv",
    output_path: str | None = None,
    top_pct_label: str = "Top 20%",
):
    pred_dir    = Path(pred_dir)
    market_path = Path(market_path)

    print("\n📂 載入 predictions...")
    df = load_predictions(pred_dir)

    start, end = df["年月日"].min(), df["年月日"].max()

    print("\n📈 載入大盤指數...")
    mkt_daily = load_market(market_path, start, end)

    # ── 每日組合報酬 ─────────────────────────────────────────
    long_ret  = daily_portfolio_return(df, df["y_pred"] == 1, label="Long (訊號買入)")
    short_ret = daily_portfolio_return(df, df["y_pred"] == 0, label="Short (非訊號)")
    all_ret   = daily_portfolio_return(df, pd.Series(True, index=df.index), label="All Stocks")

    # ── 對齊日期 ─────────────────────────────────────────────
    common_dates = long_ret.index.union(mkt_daily.index)
    long_ret  = long_ret.reindex(common_dates).fillna(0)
    short_ret = short_ret.reindex(common_dates).fillna(0)
    all_ret   = all_ret.reindex(common_dates).fillna(0)
    mkt_ret   = mkt_daily.reindex(common_dates).fillna(0)

    # ── 累積報酬 ─────────────────────────────────────────────
    cum_long  = (1 + long_ret).cumprod() - 1
    cum_short = (1 + short_ret).cumprod() - 1
    cum_all   = (1 + all_ret).cumprod() - 1
    cum_mkt   = (1 + mkt_ret).cumprod() - 1
    cum_ls    = cum_long - cum_short   # Long-Short spread

    # ── 統計摘要 ─────────────────────────────────────────────
    def sharpe(ret_series):
        r = ret_series[ret_series != 0]
        if len(r) < 2 or r.std() == 0:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(252))

    def max_drawdown(cum_series):
        roll_max = cum_series.cummax()
        dd = (cum_series - roll_max) / (1 + roll_max)
        return float(dd.min())

    stats = {
        "Long Portfolio":  (cum_long.iloc[-1],  sharpe(long_ret),  max_drawdown(cum_long)),
        "All Stocks":      (cum_all.iloc[-1],   sharpe(all_ret),   max_drawdown(cum_all)),
        "Market (TWII)":   (cum_mkt.iloc[-1],   sharpe(mkt_ret),   max_drawdown(cum_mkt)),
        "L-S Spread":      (cum_ls.iloc[-1],    sharpe(long_ret - short_ret), max_drawdown(cum_ls)),
    }

    print("\n📊 績效摘要：")
    print(f"  {'策略':<18} {'累積報酬':>10} {'Sharpe':>8} {'Max DD':>9}")
    print(f"  {'─'*48}")
    for name, (ret, sh, dd) in stats.items():
        print(f"  {name:<18} {ret:>9.2%} {sh:>8.3f} {dd:>9.2%}")

    # ============================================================
    #  繪圖
    # ============================================================
    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 11),
        gridspec_kw={"height_ratios": [3, 1.5, 1]},
        sharex=True,
    )
    fig.suptitle(
        f"條件交易策略 PnL 分析（{top_pct_label} 買入訊號）\n"
        f"{start.date()} ~ {end.date()}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    colors = {
        "long":  "#E84040",
        "short": "#5B8FD6",
        "all":   "#888888",
        "mkt":   "#27AE60",
        "ls":    "#F5A623",
    }

    # ── 子圖 1：累積報酬 ─────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(cum_long.index,  cum_long.values  * 100, color=colors["long"],  lw=1.8, label=f"Long Portfolio  ({cum_long.iloc[-1]:.1%})")
    ax1.plot(cum_mkt.index,   cum_mkt.values   * 100, color=colors["mkt"],   lw=1.8, label=f"Market (TWII)   ({cum_mkt.iloc[-1]:.1%})", linestyle="--")
    ax1.plot(cum_all.index,   cum_all.values   * 100, color=colors["all"],   lw=1.2, label=f"All Stocks       ({cum_all.iloc[-1]:.1%})", linestyle=":", alpha=0.8)
    ax1.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.4)
    ax1.set_ylabel("累積報酬 (%)", fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("累積報酬（Cumulative Return）", fontsize=11, loc="left", pad=4)

    # ── 子圖 2：Long-Short Spread ────────────────────────────
    ax2 = axes[1]
    ax2.plot(cum_ls.index, cum_ls.values * 100, color=colors["ls"], lw=1.8, label=f"L-S Spread  ({cum_ls.iloc[-1]:.1%})")
    ax2.fill_between(
        cum_ls.index,
        cum_ls.values * 100,
        0,
        where=cum_ls.values >= 0,
        alpha=0.15, color=colors["ls"],
    )
    ax2.fill_between(
        cum_ls.index,
        cum_ls.values * 100,
        0,
        where=cum_ls.values < 0,
        alpha=0.15, color="gray",
    )
    ax2.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.4)
    ax2.set_ylabel("L-S Spread (%)", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("多空價差（Long - Short Spread）", fontsize=11, loc="left", pad=4)

    # ── 子圖 3：Long Portfolio Drawdown ──────────────────────
    ax3 = axes[2]
    roll_max = cum_long.cummax()
    drawdown = (cum_long - roll_max) / (1 + roll_max) * 100
    ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color=colors["long"])
    ax3.plot(drawdown.index, drawdown.values, color=colors["long"], lw=1.0)
    ax3.set_ylabel("Drawdown (%)", fontsize=11)
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f"Long Portfolio Drawdown（Max: {drawdown.min():.1f}%）", fontsize=11, loc="left", pad=4)

    # ── 共用 x 軸格式 ────────────────────────────────────────
    axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)

    # ── 績效摘要文字框 ───────────────────────────────────────
    summary_lines = [
        f"{'策略':<16} {'累積報酬':>8}  {'Sharpe':>7}  {'MaxDD':>8}",
        "─" * 46,
    ]
    for name, (ret, sh, dd) in stats.items():
        sh_str = f"{sh:.3f}" if not np.isnan(sh) else " N/A "
        summary_lines.append(f"{name:<16} {ret:>8.2%}  {sh_str:>7}  {dd:>8.2%}")
    summary_text = "\n".join(summary_lines)

    fig.text(
        0.99, 0.01, summary_text,
        ha="right", va="bottom", fontsize=8,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  ✓ 圖表儲存：{out}")
    else:
        plt.show()

    return fig


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="條件交易策略 PnL 視覺化")
    parser.add_argument(
        "--pred_dir",
        default="database/condition_trade",
        help="predictions.csv 所在目錄（會遞迴搜尋）",
    )
    parser.add_argument(
        "--market",
        default="database/market/market.csv",
        help="市場大盤 CSV 路徑",
    )
    parser.add_argument(
        "--output",
        default="database/experiment_backtest/pnl.png",
        help="輸出圖片路徑，預設 database/experiment_backtest/pnl.png",
    )
    parser.add_argument(
        "--top_pct",
        default="Top 20%",
        help="標題顯示的訊號門檻說明",
    )
    args = parser.parse_args()

    plot_pnl(
        pred_dir    = args.pred_dir,
        market_path = args.market,
        output_path = args.output,
        top_pct_label = args.top_pct,
    )