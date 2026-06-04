"""
main.py
=======
Walk-Forward Pipeline 入口（靜態研究版）。

─────────────────────────────────────────────
  資料流向（Data Flow）
─────────────────────────────────────────────
  [輸入 Input]
    database_make/YYYY.csv          預計算特徵檔（make_features.py 產出）
                                    每個 YYYY 對應一個年份（2014–2025）
                                    欄位包含：證券代碼、年月日、特徵欄、label 欄

  [處理 Process]
    1. load_precomputed()           讀取指定年份範圍的 CSV 並合併
    2. save_distribution_plots()    診斷：繪製 label 分布圖（可選）
    3. save_label_csv()             診斷：輸出 labels.csv（可選）
    4. WalkForwardTrainer.run()     執行 walk-forward 訓練（見 objective/model.py）
                                    - 超參數策略：第一個 fold Optuna tune，後續凍結沿用
                                    - 每個 window：train 前兩年，test 第三年

  [輸出 Output]
    output/{START}_{END}/
        {col}_dist.png              label 分布圖（RUN_DIAGNOSTICS=True 時產出）
        labels.csv                  return / excess_return / beta 診斷表

    database/experiment/{START}_{END}/
        predictions.csv             測試年份逐筆預測結果（y_true, y_prob, y_pred）
        metrics.json                AUC / F1 / log-loss 等評估指標
        feature_importance.csv      特徵 gain 重要性排序
        best_params.json            本 window 使用的 LightGBM 超參數

    database/experiment/
        walk_forward_summary.csv    所有 window 的彙總指標表

─────────────────────────────────────────────
  執行方式（Usage）
─────────────────────────────────────────────
    python main.py

  調整設定：修改 RunConfig 類別內的常數即可，無需改動其他模組。

─────────────────────────────────────────────
  依賴模組（Dependencies）
─────────────────────────────────────────────
    objective/model.py              WalkForwardConfig, WalkForwardTrainer
    objective/lgb_utils.py          tune_lgb, collect_oof_prob, train_final_lgb

作者：Daniel Huang
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from objective.model import WalkForwardConfig, WalkForwardTrainer


# ============================================================
#  設定
# ============================================================

class RunConfig:
    # ── 資料來源 ─────────────────────────────────────────────
    PRECOMPUTED_DIR = Path("database_make")

    # ── Label ────────────────────────────────────────────────
    LABEL = "excess_return_tick"

    # ── Walk-Forward 範圍（2026 不納入）─────────────────────
    ML_WINDOW_START = 2014
    ML_WINDOW_END   = 2025

    # ── 執行步驟 ─────────────────────────────────────────────
    RUN_DIAGNOSTICS = False   # 分布圖 + labels.csv（需要檢查資料時再開）
    RUN_ML          = True


# ============================================================
#  診斷工具
# ============================================================

ID_COLS    = {"證券代碼", "年月日"}
LABEL_COLS = {"return", "return_tick", "return_tick_0",
              "excess_return", "excess_return_tick"}


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
        else:
            counts = series.value_counts().sort_index()
            labels = [str(int(v)) for v in counts.index]
            colors = ["#d7191c" if v == 1 else "#2c7bb6" for v in counts.index]
            bars   = ax.bar(labels, counts.values, color=colors, edgecolor="none", alpha=0.85)
            total  = counts.sum()
            for bar, val in zip(bars, counts.values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.005,
                    f"{val:,}\n({val/total:.1%})",
                    ha="center", va="bottom", fontsize=9,
                )

        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {col}  (n={len(series):,})", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        out_path = output_dir / f"{col}_dist.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  ✓ {out_path}")


def save_label_csv(df: pd.DataFrame, output_dir: Path):
    cols    = ["證券代碼", "年月日", "return", "excess_return", "beta"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  ⚠ 缺少欄位 {missing}，labels.csv 跳過")
        return
    out_path = output_dir / "labels.csv"
    df[cols].to_csv(out_path, index=False)
    print(f"  ✓ {out_path}  ({len(df):,} 筆)")


# ============================================================
#  資料載入
# ============================================================

def load_precomputed(cfg: RunConfig) -> pd.DataFrame:
    """讀取 database_make/YYYY.csv（window_start ~ window_end）並合併。"""
    frames = []
    for year in range(cfg.ML_WINDOW_START, cfg.ML_WINDOW_END + 1):
        path = cfg.PRECOMPUTED_DIR / f"{year}.csv"
        if not path.exists():
            print(f"  ⚠ {path} 不存在，跳過")
            continue
        tmp = pd.read_csv(path, low_memory=False)
        frames.append(tmp)
        print(f"  讀取 {path.name}  {tmp.shape}")

    if not frames:
        raise FileNotFoundError(
            f"{cfg.PRECOMPUTED_DIR}/ 下找不到任何 CSV，"
            "請確認 database_make/ 已正確產出"
        )

    df = pd.concat(frames, ignore_index=True)
    df["年月日"]   = pd.to_datetime(df["年月日"])
    df["證券代碼"] = df["證券代碼"].astype(str)
    df = df.sort_values(["證券代碼", "年月日"]).reset_index(drop=True)

    print(f"\n  合併後 shape: {df.shape}")

    if cfg.LABEL not in df.columns:
        raise ValueError(
            f"Label '{cfg.LABEL}' 不存在於資料中，"
            "請確認 database_make/ 已正確產出 label"
        )
    return df


# ============================================================
#  主流程
# ============================================================

def run_ml(cfg: RunConfig):
    output_dir = Path("output") / f"{cfg.ML_WINDOW_START}_{cfg.ML_WINDOW_END}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.RUN_DIAGNOSTICS:
        print("\n► 讀取預計算特徵（診斷用）…")
        df = load_precomputed(cfg)
        print("\n► 診斷圖表")
        save_distribution_plots(df, output_dir)
        save_label_csv(df, output_dir)
        del df   # 診斷完立即釋放記憶體

    print("\n► Walk-Forward Training")
    ml_config = WalkForwardConfig(
        window_start       = cfg.ML_WINDOW_START,
        window_end         = cfg.ML_WINDOW_END,
        precomputed_dir    = cfg.PRECOMPUTED_DIR,
        target_col         = cfg.LABEL,
        use_vol_weight     = False,
        freeze_hyperparams = True,   # 只在第一個 fold 執行 Optuna，後續沿用凍結超參數
        retune_every_n     = 999,    # 實質上只 tune 一次
    )
    trainer = WalkForwardTrainer(ml_config)
    trainer.run()


def main():
    cfg = RunConfig()

    print(f"\n{'='*60}")
    print(f"  Walk-Forward Pipeline")
    print(f"  資料來源  : {cfg.PRECOMPUTED_DIR}/")
    print(f"  年份範圍  : {cfg.ML_WINDOW_START} ~ {cfg.ML_WINDOW_END}")
    print(f"  Label     : {cfg.LABEL}")
    print(f"  Diagnostics: {cfg.RUN_DIAGNOSTICS}")
    print(f"{'='*60}")

    if not cfg.RUN_ML:
        print("  RUN_ML=False，結束。")
        return

    t0 = time.time()
    run_ml(cfg)
    print(f"\n  總耗時: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()