"""
backtest.py
===========
Walk-Forward 回測模組。

─────────────────────────────────────────────
  資料流向（Data Flow）
─────────────────────────────────────────────
  [輸入 Input]
    database/experiment/YYYY_YYYY/predictions.csv
        欄位：證券代碼, 年月日, return, y_true, y_prob, y_pred
        return = 明日開盤 → 後日開盤（已排除 look-ahead bias）

  [處理 Process]
    1. 讀取所有 window 的 predictions.csv 並合併
    2. 去除重複（同股票同日期保留最早 window）
    3. 每個交易日依 y_prob 排序，分成 10 個 Decile 組
       Decile 1 = y_prob 最高（最看漲）
    4. 計算各組每日等權報酬，累積成時間序列
    5. 輸出指標與圖表

  [輸出 Output]
    output/backtest/
        decile_metrics.csv          10 組的 return/std/sharpe/mdd 等指標
        cumulative_returns.png      10 組累積報酬曲線
        annual_decile1.png          Decile 1 各年度表現長條圖
        decile_summary_bar.png      各組年化報酬 & Sharpe 橫向比較
        daily_returns.csv           每日各組報酬（可供進一步分析）

─────────────────────────────────────────────
  設定（Config）
─────────────────────────────────────────────
    EXPERIMENT_DIR  : predictions.csv 所在的上層目錄
    OUTPUT_DIR      : 圖表與指標輸出目錄
    N_DECILES       : 分組數（預設 10）
    COST_BPS        : 單邊交易成本 bps（預設 0）
    ANNUAL_DAYS     : 年化交易日數（預設 252）

作者：Daniel Huang
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ============================================================
#  設定
# ============================================================

EXPERIMENT_DIR = Path("database/experiment")
OUTPUT_DIR     = Path("output/backtest")
N_DECILES      = 10
COST_BPS       = 0        # 單邊交易成本（bps），每日換倉 × 2
ANNUAL_DAYS    = 252


# ============================================================
#  工具函式
# ============================================================

def load_all_predictions(exp_dir: Path) -> pd.DataFrame:
    """讀取所有 window 的 predictions.csv，合併並去重。"""
    frames = []
    for path in sorted(exp_dir.glob("*/predictions.csv")):
        df = pd.read_csv(path, low_memory=False)
        df["_window"] = path.parent.name
        frames.append(df)
        print(f"  讀取 {path}  {df.shape}")

    if not frames:
        raise FileNotFoundError(f"{exp_dir} 下找不到任何 predictions.csv")

    combined = pd.concat(frames, ignore_index=True)
    combined["年月日"] = pd.to_datetime(combined["年月日"])
    combined["證券代碼"] = combined["證券代碼"].astype(str)
    combined = combined.sort_values(["證券代碼", "年月日", "_window"])

    before = len(combined)
    combined = combined.drop_duplicates(subset=["證券代碼", "年月日"], keep="first")
    after = len(combined)
    if before != after:
        print(f"  去重：移除 {before - after:,} 筆重複（保留最早 window）")

    combined = combined.sort_values(["年月日", "證券代碼"]).reset_index(drop=True)
    print(f"\n  合併後：{len(combined):,} 筆  日期範圍 {combined['年月日'].min().date()} ~ {combined['年月日'].max().date()}")
    return combined


def assign_deciles(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """每個交易日依 y_prob 降序分 n 組，Decile 1 = 最高 prob。"""
    def _decile(group):
        group = group.copy()
        group["decile"] = pd.qcut(
            group["y_prob"].rank(method="first", ascending=False),
            q=n,
            labels=range(1, n + 1),
        )
        return group

    df = df.groupby("年月日", group_keys=False).apply(_decile)
    df["decile"] = df["decile"].astype(int)
    return df


def daily_returns(df: pd.DataFrame, cost_bps: float = 0) -> pd.DataFrame:
    """
    計算每日各 Decile 的等權報酬。
    cost_bps：單邊成本（每日換倉視為全部換，× 2）
    """
    cost = cost_bps / 10_000 * 2
    ret = (
        df.groupby(["年月日", "decile"])["return"]
        .mean()
        .unstack("decile")
        .sort_index()
    )
    ret.columns = [f"D{c}" for c in ret.columns]
    ret = ret - cost
    return ret


def calc_metrics(ret_series: pd.Series, annual_days: int = 252) -> dict:
    """計算單一報酬序列的績效指標。"""
    r = ret_series.dropna()
    ann_ret  = (1 + r).prod() ** (annual_days / len(r)) - 1
    ann_std  = r.std()  * np.sqrt(annual_days)
    sharpe   = ann_ret / ann_std if ann_std > 0 else np.nan
    cum      = (1 + r).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    mdd      = drawdown.min()
    calmar   = ann_ret / abs(mdd) if mdd != 0 else np.nan
    hit_rate = (r > 0).mean()
    return {
        "ann_return":  round(ann_ret,  4),
        "ann_std":     round(ann_std,  4),
        "sharpe":      round(sharpe,   4),
        "mdd":         round(mdd,      4),
        "calmar":      round(calmar,   4),
        "hit_rate":    round(hit_rate, 4),
        "n_days":      len(r),
    }


# ============================================================
#  圖表
# ============================================================

COLORS = plt.cm.RdYlGn(np.linspace(0.15, 0.85, N_DECILES))


def plot_cumulative(ret_df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, col in enumerate(ret_df.columns):
        cum     = (1 + ret_df[col].fillna(0)).cumprod()
        log_cum = np.log10(cum)
        lw      = 2.5 if col == "D1" else 1.0
        ax.plot(log_cum.index, log_cum.values, label=col, color=COLORS[i], linewidth=lw)

    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.set_title("Decile Cumulative Returns (log10 NAV)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("log10(NAV)")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.legend(ncol=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "cumulative_returns.png", dpi=150)
    plt.close(fig)
    print("  ✓ cumulative_returns.png")


def plot_decile_summary(metrics_df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    deciles = metrics_df.index.tolist()
    x = np.arange(len(deciles))

    # 年化報酬
    ax = axes[0]
    bars = ax.bar(x, metrics_df["ann_return"], color=COLORS, edgecolor="none", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(deciles, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_title("Annualised Return by Decile", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, metrics_df["ann_return"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002 * np.sign(val + 1e-9),
                f"{val:.1%}", ha="center", va="bottom", fontsize=8)

    # Sharpe
    ax = axes[1]
    bars = ax.bar(x, metrics_df["sharpe"], color=COLORS, edgecolor="none", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(deciles, fontsize=9)
    ax.set_title("Sharpe Ratio by Decile", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, metrics_df["sharpe"]):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02 * np.sign(val + 1e-9),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "decile_summary_bar.png", dpi=150)
    plt.close(fig)
    print("  ✓ decile_summary_bar.png")


def plot_annual_decile1(ret_df: pd.DataFrame, output_dir: Path):
    """Decile 1 各年度年化報酬長條圖。"""
    d1 = ret_df["D1"].dropna()
    annual = d1.groupby(d1.index.year).apply(
        lambda r: (1 + r).prod() - 1
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d7191c" if v < 0 else "#2c7bb6" for v in annual.values]
    bars = ax.bar(annual.index.astype(str), annual.values,
                  color=colors, edgecolor="none", alpha=0.85)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_title("Decile 1 – Annual Return", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, annual.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * np.sign(val + 1e-9),
                f"{val:.1%}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "annual_decile1.png", dpi=150)
    plt.close(fig)
    print("  ✓ annual_decile1.png")


def plot_drawdown(ret_df: pd.DataFrame, output_dir: Path):
    """Decile 1 水下曲線。"""
    d1  = ret_df["D1"].fillna(0)
    cum = (1 + d1).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, color="#d7191c", alpha=0.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_title("Decile 1 – Drawdown", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "drawdown_decile1.png", dpi=150)
    plt.close(fig)
    print("  ✓ drawdown_decile1.png")


# ============================================================
#  主流程
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Backtest")
    print(f"  Experiment dir : {EXPERIMENT_DIR}")
    print(f"  Deciles        : {N_DECILES}")
    print(f"  Cost (bps)     : {COST_BPS}")
    print(f"{'='*60}\n")

    # ── 1. 載入資料 ──────────────────────────────────────────
    print("► 載入 predictions...")
    df = load_all_predictions(EXPERIMENT_DIR)

    # ── 2. 分組 ──────────────────────────────────────────────
    print("\n► 分 Decile 組...")
    df = assign_deciles(df, n=N_DECILES)
    print(f"  Decile 分布：\n{df['decile'].value_counts().sort_index().to_string()}")

    # ── 3. 每日報酬 ──────────────────────────────────────────
    print("\n► 計算每日組別報酬...")
    ret_df = daily_returns(df, cost_bps=COST_BPS)
    ret_df.to_csv(OUTPUT_DIR / "daily_returns.csv", encoding="utf-8-sig")
    print(f"  ✓ daily_returns.csv  shape={ret_df.shape}")

    # ── 4. 績效指標 ──────────────────────────────────────────
    print("\n► 計算績效指標...")
    metrics_list = []
    for col in ret_df.columns:
        m = calc_metrics(ret_df[col], ANNUAL_DAYS)
        m["decile"] = col
        metrics_list.append(m)

    metrics_df = pd.DataFrame(metrics_list).set_index("decile")
    metrics_df.to_csv(OUTPUT_DIR / "decile_metrics.csv", encoding="utf-8-sig")
    print(metrics_df[["ann_return", "ann_std", "sharpe", "mdd", "calmar", "hit_rate"]].to_string())

    # ── 5. 圖表 ──────────────────────────────────────────────
    print("\n► 產出圖表...")
    plot_cumulative(ret_df, OUTPUT_DIR)
    plot_decile_summary(metrics_df, OUTPUT_DIR)
    plot_annual_decile1(ret_df, OUTPUT_DIR)
    plot_drawdown(ret_df, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f"  完成  →  {OUTPUT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()