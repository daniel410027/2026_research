"""
main.py
=======
Pipeline 入口，所有設定集中於各 Config 區塊修改。

使用方式：
    python main.py

作者：Daniel Huang
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from preprocess import PreprocessConfig, DataPreprocessor
from model import WalkForwardConfig, WalkForwardTrainer


# ============================================================
#  RUN CONFIG：控制執行哪些步驟
# ============================================================

class RunConfig:
    # ── 執行步驟：True = 執行，False = 跳過 ──────────────────
    RUN_PREPROCESS = False    # 前處理 EDA（存分布圖 + labels.csv）
    RUN_ML         = True    # Walk-forward 機器學習訓練

    # ──────────────────────────────────────────────────────────
    #  前處理 EDA 設定（RUN_PREPROCESS 用）
    # ──────────────────────────────────────────────────────────
    START_YEAR = 2014
    END_YEAR   = 2016        # EDA 用較小範圍即可；ML 有自己的 windows

    RAW_DATA_DIR     = Path("database/raw_data")
    MARKET_DATA_PATH = Path("database/market/market.csv")
    OUTPUT_BASE_DIR  = Path("database/processed")

    RETURN_CLIP     = 0.15
    TICK_THRESHOLD  = 0.01
    TICK0_THRESHOLD = 0.00
    BETA_WINDOW     = 60

    LABEL = "excess_return_tick"

    # ──────────────────────────────────────────────────────────
    #  ML 設定（RUN_ML 用）── 細部超參數在 WalkForwardConfig 調整
    # ──────────────────────────────────────────────────────────
    ML_WINDOW_START = 2014
    ML_WINDOW_END   = 2025


# ============================================================
#  輸出路徑
# ============================================================

def get_output_dir(cfg: RunConfig) -> Path:
    folder = f"{cfg.START_YEAR}_{cfg.END_YEAR}"
    return cfg.OUTPUT_BASE_DIR / folder


# ============================================================
#  分布圖 + labels.csv（EDA 用）
# ============================================================

def save_distribution_plots(df: pd.DataFrame, output_dir: Path):
    PLOT_COLS = {
        "return":             "continuous",
        "excess_return":      "continuous",
        "return_tick":        "discrete",
        "excess_return_tick": "discrete",
    }

    for col, kind in PLOT_COLS.items():
        if col not in df.columns:
            print(f"  ⚠ 欄位 '{col}' 不存在，跳過")
            continue

        series = df[col].dropna()
        fig, ax = plt.subplots(figsize=(8, 4))

        if kind == "continuous":
            ax.hist(series, bins=100, color="#2c7bb6", edgecolor="none", alpha=0.85)
            ax.axvline(series.mean(),   color="#d7191c", linewidth=1.2,
                       label=f"mean={series.mean():.4f}")
            ax.axvline(series.median(), color="#fdae61", linewidth=1.2, linestyle="--",
                       label=f"median={series.median():.4f}")
            ax.legend(fontsize=9)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        else:
            counts = series.value_counts().sort_index()
            labels = [str(int(v)) for v in counts.index]
            colors = ["#d7191c" if v == 1 else "#2c7bb6" for v in counts.index]
            bars   = ax.bar(labels, counts.values, color=colors, edgecolor="none", alpha=0.85)
            total  = counts.sum()
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + total * 0.005,
                        f"{val:,}\n({val/total:.1%})",
                        ha="center", va="bottom", fontsize=9)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")

        ax.set_title(f"Distribution of {col}  (n={len(series):,})", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        out_path = output_dir / f"{col}_dist.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  ✓ 圖表: {out_path}")


def save_label_csv(df: pd.DataFrame, output_dir: Path):
    cols    = ["證券代碼", "年月日", "return", "excess_return", "beta"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  ⚠ 缺少欄位 {missing}，CSV 跳過")
        return

    out_path = output_dir / "labels.csv"
    df[cols].to_csv(out_path, index=False)
    print(f"  ✓ CSV  : {out_path}  ({len(df):,} 筆)")


# ============================================================
#  步驟函式
# ============================================================

def run_preprocess(cfg: RunConfig) -> pd.DataFrame:
    """
    執行 EDA 前處理，回傳完整 DataFrame。

    回傳值供 run_ml() 的 inject_cache() 使用：
    當 EDA 年份範圍（START_YEAR, END_YEAR）與 ML 第一個 window 相同時，
    可直接注入 cache，避免重複跑一次 DataPreprocessor。
    """
    preprocess_config = PreprocessConfig(
        start_year       = cfg.START_YEAR,
        end_year         = cfg.END_YEAR,
        raw_data_dir     = cfg.RAW_DATA_DIR,
        market_data_path = cfg.MARKET_DATA_PATH,
        beta_window      = cfg.BETA_WINDOW,
        return_clip      = cfg.RETURN_CLIP,
        tick_threshold   = cfg.TICK_THRESHOLD,
        tick0_threshold  = cfg.TICK0_THRESHOLD,
    )

    df = DataPreprocessor(preprocess_config).run()

    if cfg.LABEL not in df.columns:
        print(f"\n  ✗ Label '{cfg.LABEL}' 不存在，請確認前處理步驟。")
        sys.exit(1)

    # ── 印開頭結尾 ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  DataFrame Head (5)")
    print(f"{'─'*60}")
    print(df.head(5).to_string())

    print(f"\n{'─'*60}")
    print("  DataFrame Tail (5)")
    print(f"{'─'*60}")
    print(df.tail(5).to_string())

    # ── 輸出 ─────────────────────────────────────────────────
    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n► 輸出至 {output_dir}")

    save_distribution_plots(df, output_dir)
    save_label_csv(df, output_dir)

    return df


def run_ml(cfg: RunConfig, precomputed_df: pd.DataFrame | None = None):
    """
    執行 walk-forward ML 訓練。

    Parameters
    ----------
    cfg : RunConfig
    precomputed_df : pd.DataFrame | None
        若 EDA 步驟已產生與 ML 第一個 window（ML_WINDOW_START, ML_WINDOW_START+2）
        相同年份的 df，傳入後自動注入 cache，跳過重複前處理。
        年份不符時會印出提示並跳過注入。
    """
    ml_config = WalkForwardConfig(
        window_start     = cfg.ML_WINDOW_START,
        window_end       = cfg.ML_WINDOW_END,
        raw_data_dir     = cfg.RAW_DATA_DIR,
        market_data_path = cfg.MARKET_DATA_PATH,
        return_clip      = cfg.RETURN_CLIP,
        tick_threshold   = cfg.TICK_THRESHOLD,
        tick0_threshold  = cfg.TICK0_THRESHOLD,
        beta_window      = cfg.BETA_WINDOW,
        use_vol_weight = False,   # 暫時關閉 vol weighting
    )
    trainer = WalkForwardTrainer(ml_config)

    # ── 注入 EDA df cache（避免第一個 window 重複前處理）────
    if precomputed_df is not None:
        first_window_end = cfg.ML_WINDOW_START + 2
        if cfg.START_YEAR == cfg.ML_WINDOW_START and cfg.END_YEAR == first_window_end:
            trainer.inject_cache(cfg.ML_WINDOW_START, first_window_end, precomputed_df)
        else:
            print(
                f"  ⚠ EDA 年份（{cfg.START_YEAR}–{cfg.END_YEAR}）"
                f"與 ML 第一個 window（{cfg.ML_WINDOW_START}–{first_window_end}）不符，"
                f"跳過 cache 注入"
            )

    trainer.run()


# ============================================================
#  MAIN
# ============================================================

def main():
    cfg = RunConfig()

    steps = [
        ("前處理 EDA", cfg.RUN_PREPROCESS, run_preprocess),
        ("ML 訓練",    cfg.RUN_ML,         run_ml),
    ]

    active = [name for name, flag, _ in steps if flag]

    print(f"\n{'='*60}")
    print(f"  Pipeline 設定")
    print(f"  EDA 年份  : {cfg.START_YEAR} ~ {cfg.END_YEAR}")
    print(f"  ML Windows: {cfg.ML_WINDOW_START} ~ {cfg.ML_WINDOW_END}")
    print(f"  執行步驟  : {' → '.join(active) if active else '（無）'}")
    print(f"{'='*60}")

    if not active:
        print("  所有步驟皆已關閉，結束。")
        return

    t0 = time.time()

    eda_df = None
    for name, flag, fn in steps:
        if not flag:
            continue
        print(f"\n{'─'*60}")
        print(f"  ▶ {name}")
        print(f"{'─'*60}")

        if name == "前處理 EDA":
            eda_df = fn(cfg)
        elif name == "ML 訓練":
            fn(cfg, precomputed_df=eda_df)
        else:
            fn(cfg)

    print(f"\n  總耗時: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()