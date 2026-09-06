"""
analyze_labels.py
=================
快速分析 labels.csv 的內容。

功能：
    1. beta 敘述統計（含 beta==1.0 佔比診斷）
    2. return vs excess_return 散佈圖（含 y=x 參考線與相關係數）

使用方式：
    python analyze_labels.py
    python analyze_labels.py --csv database/processed/20240101_2014_2023/labels.csv

作者：Daniel Huang
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")   # 無視窗環境請改 "Agg"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
#  CONFIG
# ============================================================

DEFAULT_CSV = Path("database/processed/2014_2016/labels.csv")  # 預設路徑，可在此修改


# ============================================================
#  分析函式
# ============================================================

def print_beta_stats(df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("  Beta 敘述統計")
    print("=" * 50)

    stats = df["beta"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(stats.to_string())

    print(f"\n  skewness : {df['beta'].skew():.4f}")
    print(f"  kurtosis : {df['beta'].kurt():.4f}")
    print(f"  beta < 0 : {(df['beta'] < 0).sum():,}  ({(df['beta'] < 0).mean():.2%})")
    print(f"  beta > 2 : {(df['beta'] > 2).sum():,}  ({(df['beta'] > 2).mean():.2%})")

    # ── Vasicek shrinkage 診斷 ───────────────────────────────
    exact_one = (df["beta"] == 1.0).sum()
    print(f"\n  [Vasicek 診斷]")
    print(f"  beta 恰好 == 1.0 : {exact_one:,}  ({exact_one / len(df):.2%})")
    print(f"  （此比例過高代表大量股票資料稀疏，shrinkage 完全縮向預設值）")


ANALYZE_DIR = Path("database/analyzebeta")


def plot_return_vs_excess(df: pd.DataFrame):
    r  = df["return"].dropna()
    er = df["excess_return"].dropna()
    common = r.index.intersection(er.index)
    r, er  = r.loc[common], er.loc[common]

    corr = np.corrcoef(r, er)[0, 1]

    # 抽樣（資料量大時散佈圖太密）
    n_sample = min(30_000, len(r))
    idx  = np.random.choice(len(r), n_sample, replace=False)
    r_s  = r.iloc[idx]
    er_s = er.iloc[idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(r_s, er_s, s=2, alpha=0.2, color="#2c7bb6", rasterized=True)

    lim = max(abs(r_s).max(), abs(er_s).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="#d7191c", linewidth=1,
            linestyle="--", label="y = x")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("return")
    ax.set_ylabel("excess_return")
    ax.set_title(
        f"return vs excess_return  (n={n_sample:,} sampled, corr={corr:.4f})",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    ANALYZE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYZE_DIR / "return_vs_excess_return.png"
    fig.savefig(out_path, dpi=150)

    print(f"\n  Pearson correlation (return, excess_return) : {corr:.6f}")
    print(f"  ✓ 圖表已儲存: {out_path}")
    print("  （圖表已顯示，關閉視窗繼續）")
    plt.show()


# ============================================================
#  MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="分析 labels.csv")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="labels.csv 路徑")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.csv.exists():
        print(f"✗ 找不到 CSV：{args.csv}")
        return

    print(f"  載入: {args.csv}")
    df = pd.read_csv(args.csv, parse_dates=["年月日"])
    print(f"  筆數: {len(df):,}  |  期間: {df['年月日'].min().date()} ~ {df['年月日'].max().date()}")

    print_beta_stats(df)
    plot_return_vs_excess(df)


if __name__ == "__main__":
    main()