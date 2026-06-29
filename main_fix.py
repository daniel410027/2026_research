"""
main_fix.py
===========
Walk-Forward Pipeline 入口（靜態研究版・固定超參數版）。

與 main.py 的唯一差異：
    ★ 不使用 Optuna。直接套用 best_params.json（或內嵌 dict）的固定超參數，
      所有 window 共用同一組超參數。

  運作原理（無需改 model.py / lgb_utils.py）：
    WalkForwardTrainer 的凍結機制（_should_tune）在
        freeze_hyperparams=True、retune_every_n=999、且 _frozen_params 已有值
    時，對所有 fold 回傳 False → 跳過 tune_lgb、改走 collect_oof_prob
    （固定超參數下只收集 OOF prob 供 find_threshold）。
    因此在 trainer.run() 前「預先植入」trainer._frozen_params = best_params，
    Optuna 即一次都不會執行。

    其餘行為與 main.py 完全相同：
      - final model 仍用 train_final_lgb 預設 500 輪
      - threshold 仍由 collect_oof_prob 的 OOF F1 決定
      - 特徵集對齊（排除 amount 等）同 main.py

  超參數來源（擇一）：
    1. RunConfig.BEST_PARAMS_INLINE 不為 None → 用內嵌 dict
    2. 否則讀 RunConfig.BEST_PARAMS_PATH（預設 best_params.json）

─────────────────────────────────────────────
  ★ 2026-06-15（1）：繞過 objective/__init__.py
─────────────────────────────────────────────
  本實驗空間（2026_research）的 objective/__init__.py 內有
      from daily_model import DailyConfig, DailyTrainer
  而 daily_model.py 屬於另一個（live）專案、不在此 repo →
  任何正常 import objective 套件都會先執行 __init__.py 而觸發 ModuleNotFoundError。
  改以 importlib 直接載入 objective.model，繞過 __init__.py。

─────────────────────────────────────────────
  ★ 2026-06-15（2）：對齊 live(daily) pipeline 訓練特徵集
─────────────────────────────────────────────
  daily_model.py 的 _EXCLUDE_COLS 將 amount（成交金額，僅流動性過濾用）排除於
  訓練之外；static 端的 model._resolve_features 會把所有數值欄當特徵 →
  於 run_ml() 啟動 trainer 前擴充 objective.model._EXCLUDE_COLS（不改 model.py）。

─────────────────────────────────────────────
  ★ 2026-06-23：新增 size/price 外部條件變數排除
─────────────────────────────────────────────
  為研究「市值(size) / 股價(price) × 模型訊號」對報酬的交互效果，
  make_new.py 已把 market_value、close 保留進 database_make/YYYY.csv。
  這兩欄屬【模型外部的 conditioning variable】，**不可進訓練**——
  否則模型直接學到 size/price，交互分析淪為循環論證（模型本就看了 size，
  再問它的 edge 是否隨 size 變化，訊號被自己吃掉）。
  故於 EXTRA_EXCLUDE_COLS 一併排除（與 amount 同性質：保留於 CSV、排除於訓練）。
  下游 interaction.py 以這兩欄作為 conditioning 軸。

執行方式：
    python main_fix.py

作者：Daniel Huang
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
#  載入 objective 套件（繞過 __init__.py）
# ============================================================

_ROOT = os.path.dirname(os.path.abspath(__file__))
_OBJ  = os.path.join(_ROOT, "objective")

if "objective" not in sys.modules:
    _pkg             = types.ModuleType("objective")
    _pkg.__path__    = [_OBJ]
    _pkg.__package__ = "objective"
    sys.modules["objective"] = _pkg


def _load_obj_module(name: str) -> types.ModuleType:
    """以 importlib 載入 objective/{name}.py，繞過 objective/__init__.py。"""
    full = f"objective.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec            = importlib.util.spec_from_file_location(full, os.path.join(_OBJ, f"{name}.py"))
    mod             = importlib.util.module_from_spec(spec)
    mod.__package__ = "objective"
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_wf_model          = _load_obj_module("model")   # ★ 模組本體，供擴充 _EXCLUDE_COLS
WalkForwardConfig  = _wf_model.WalkForwardConfig
WalkForwardTrainer = _wf_model.WalkForwardTrainer


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
    RUN_DIAGNOSTICS = False
    RUN_ML          = True

    # ── ★ 固定超參數（不使用 Optuna）─────────────────────────
    # BEST_PARAMS_INLINE 不為 None → 優先使用內嵌 dict；
    # 否則讀 BEST_PARAMS_PATH（best_params.json）。
    BEST_PARAMS_PATH   = Path("best_params.json")
    BEST_PARAMS_INLINE = None
    # 範例（直接內嵌時可填）：
    # BEST_PARAMS_INLINE = {
    #     "learning_rate":    0.01648668912028271,
    #     "num_leaves":       425,
    #     "feature_fraction": 0.913747901784067,
    #     "bagging_fraction": 0.9310440872731132,
    #     "bagging_freq":     20,
    #     "min_data_in_leaf": 77,
    #     "max_depth":        18,
    # }

    # ── ★ 對齊 live(daily) 訓練特徵集 ───────────────────────
    ALIGN_FEATURES_WITH_LIVE = True
    EXTRA_EXCLUDE_COLS = {
        "amount",              # 成交金額：daily 僅作流動性過濾，不進訓練
        "market_value",        # ★ size proxy：外部條件變數（size×model 交互分析），不進訓練
        "close",               # ★ price proxy：外部條件變數（price×model 交互分析），不進訓練
        "market_return_fwd",   # ★ 前視 T+1→T+2：2026-06-25 起保留於 database_make/ 輸出
                                #   （供 backtest.py 對齊大盤用),訓練端仍須排除,防 look-ahead
        "market_index",        # 與 model._EXCLUDE_COLS 既有項目對齊（idempotent）
    }


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
#  ★ 固定超參數載入
# ============================================================

def load_best_params(cfg: RunConfig) -> dict:
    """
    取得固定超參數（不經 Optuna）。
    優先用 BEST_PARAMS_INLINE，否則讀 BEST_PARAMS_PATH（best_params.json）。
    回傳的 dict 不含 base params（objective/metric 等），
    由 lgb_utils 在 collect_oof_prob / train_final_lgb 內以 {**_LGB_BASE_PARAMS, **best_params} 合併。
    """
    if cfg.BEST_PARAMS_INLINE is not None:
        params = dict(cfg.BEST_PARAMS_INLINE)
        print("  超參數來源：RunConfig.BEST_PARAMS_INLINE（內嵌）")
    else:
        path = Path(cfg.BEST_PARAMS_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"找不到固定超參數檔：{path}\n"
                f"請放置 best_params.json，或改用 RunConfig.BEST_PARAMS_INLINE 內嵌。"
            )
        with open(path, encoding="utf-8") as f:
            params = json.load(f)
        print(f"  超參數來源：{path}")

    if not isinstance(params, dict) or not params:
        raise ValueError(f"固定超參數內容無效（需非空 dict）：{params}")

    for k, v in params.items():
        print(f"    {k:18s}: {v}")
    return params


# ============================================================
#  ★ 特徵集對齊（與 live/daily 一致）
# ============================================================

def align_features_with_live(cfg: RunConfig):
    """擴充 objective.model._EXCLUDE_COLS，使訓練特徵集與 daily_model.py 一致。"""
    before = set(_wf_model._EXCLUDE_COLS)
    _wf_model._EXCLUDE_COLS = before | set(cfg.EXTRA_EXCLUDE_COLS)
    added = sorted(set(cfg.EXTRA_EXCLUDE_COLS) - before)
    if added:
        print(f"  ▸ 對齊 live 特徵集，額外自訓練排除：{added}")
    else:
        print("  ▸ live 特徵集已對齊（無新增排除欄）")


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
        del df

    # ── ★ 載入固定超參數 ──────────────────────────────────────
    print("\n► 固定超參數（不使用 Optuna）")
    best_params = load_best_params(cfg)

    # ── ★ 訓練特徵集對齊（必須在 trainer.run() 之前）──────────
    if cfg.ALIGN_FEATURES_WITH_LIVE:
        print("\n► 特徵集對齊（train/serve consistency）")
        align_features_with_live(cfg)

    print("\n► Walk-Forward Training（固定超參數）")
    ml_config = WalkForwardConfig(
        window_start       = cfg.ML_WINDOW_START,
        window_end         = cfg.ML_WINDOW_END,
        precomputed_dir    = cfg.PRECOMPUTED_DIR,
        target_col         = cfg.LABEL,
        use_vol_weight     = False,
        freeze_hyperparams = True,   # ★ 配合預植 _frozen_params，使所有 fold 跳過 Optuna
        retune_every_n     = 999,    # ★ 完全凍結，永不 re-tune
    )
    trainer = WalkForwardTrainer(ml_config)

    # ── ★ 預先植入凍結超參數 → _should_tune 對所有 fold 回傳 False ──
    #    → tune_lgb 一次都不執行，改走 collect_oof_prob（固定參數收集 OOF）。
    trainer._frozen_params = best_params
    print("  ✓ 已植入固定超參數，Optuna 全程停用")

    trainer.run()


def main():
    cfg = RunConfig()

    print(f"\n{'='*60}")
    print(f"  Walk-Forward Pipeline（固定超參數版）")
    print(f"  資料來源  : {cfg.PRECOMPUTED_DIR}/")
    print(f"  年份範圍  : {cfg.ML_WINDOW_START} ~ {cfg.ML_WINDOW_END}")
    print(f"  Label     : {cfg.LABEL}")
    print(f"  Optuna    : 停用（固定超參數）")
    print(f"  Diagnostics: {cfg.RUN_DIAGNOSTICS}")
    print(f"  對齊 live  : {cfg.ALIGN_FEATURES_WITH_LIVE}")
    print(f"{'='*60}")

    if not cfg.RUN_ML:
        print("  RUN_ML=False，結束。")
        return

    t0 = time.time()
    run_ml(cfg)
    print(f"\n  總耗時: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()