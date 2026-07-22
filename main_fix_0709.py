"""
main_fix.py
===========
Walk-Forward Pipeline 入口（靜態研究版）。

超參數模式（RunConfig.TUNE_MODE 切換，2026-07-22 改版）：
    ★ "every_fold"（預設）：每個 fold 都重跑 Optuna（N_TRIALS 次），
      各折超參數不同。逐折參數見 database/experiment/{window}/best_params.json。

    ★ "first_only"：只有第一個 fold 執行 Optuna，之後所有 fold 凍結沿用。
      tune 出的參數會另存 best_params_regression.json。

    ★ "periodic"：每 RETUNE_EVERY_N 個 fold 重新 tune 一次。

    ★ "frozen"：完全不使用 Optuna，直接套用 best_params.json（或內嵌 dict）
      的固定超參數，所有 window 共用同一組。

  ⚠ 舊版用 TUNE_FIRST_FOLD(bool) 表達，但字面意思與實際行為不符：
    它同時對 WalkForwardConfig 傳 freeze_hyperparams=False，而
    model._should_tune 的第一個分支會在檢查 _frozen_params 之前就短路 →
      - TUNE_FIRST_FOLD=True  實際上是「每個 fold 都 tune」（非「第一折後凍結」），
        存出的 best_params_regression.json 其實是最後一折的參數；
      - TUNE_FIRST_FOLD=False 的預植參數被完全忽略，Optuna 照跑，
        best_params.json 被靜默丟棄，畫面卻印「Optuna 全程停用」。
    現行 TUNE_MODE="every_fold" 即舊版 TUNE_FIRST_FOLD=True 的實際行為，
    換寫法不改結果。詳見 summary_research.md 5.1。

    其餘行為各模式相同：
      - final model 仍用 train_final_lgb 預設 500 輪
      - OOF Spearman IC 仍由 collect_oof_prob 的 OOF 預測值計算（監控用，無 threshold）
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

─────────────────────────────────────────────
  ★ 2026-07-09：整合 805_group2 流動性篩選（formula3）
─────────────────────────────────────────────
  在 WalkForwardTrainer 讀取資料之前，先對全範圍
  [ML_WINDOW_START, ML_WINDOW_END] 套用 formula3_amount_mv_turnover
  篩選（w1=0.1004, w2=0.1896, keep_ratio=0.2000），train + predict
  一起篩（本支線純研究/回測，無 daily_model 的既有持倉問題，
  不需要區分 train-only vs. predict-all）。
  篩選後資料寫到 database_make_liq_filtered/{tag}/YYYY.csv（依參數
  tag 命名快取，非時間戳，重跑會自動重用），再把這個目錄餵給
  WalkForwardConfig.precomputed_dir，不改 model.py。
  同時寫一份 liquidity_filter_meta.json 到 database/experiment/，
  供 backtest.py 自動讀取並在報表上標示本次用了哪組篩選參數。

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

import numpy as np
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
    LABEL = "excess_return"

    # ── Walk-Forward 範圍（2026 不納入）─────────────────────
    ML_WINDOW_START = 2014
    ML_WINDOW_END   = 2025

    # ── 執行步驟 ─────────────────────────────────────────────
    RUN_ML          = True

    # ── ★ 超參數模式 ─────────────────────────────────────────
    # ★ 2026-07-22：原本的 TUNE_FIRST_FOLD(bool) 換成 TUNE_MODE 字串。
    #   舊寫法的字面意思（「第一個 fold tune 後凍結」）與實際行為不符——
    #   它同時傳 freeze_hyperparams=False，導致每個 fold 都重跑 Optuna。
    #   下面的預設值 "every_fold" 就是**舊版實際跑的行為**，換寫法不改結果。
    #
    #   "every_fold" ── 每個 fold 都重跑 Optuna（每年重訓超參，現行預設）
    #   "first_only" ── 只有第一個 fold tune，之後凍結沿用
    #   "periodic"   ── 每 RETUNE_EVERY_N 個 fold 重新 tune 一次
    #   "frozen"     ── 完全不 tune，用下方固定超參數
    #                   （來源：BEST_PARAMS_INLINE 或 BEST_PARAMS_PATH）
    TUNE_MODE       = "every_fold"
    RETUNE_EVERY_N  = 3      # 僅 TUNE_MODE="periodic" 時有意義
    N_TRIALS        = 15

    # ── ★ 固定超參數（TUNE_MODE="frozen" 時使用）─────────────
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
        "market_return",       # ★ backward 大盤日報酬：per-day constant（日期指紋），
                                #   對 cross-sectional 排序無資訊且易致 date-level 過擬合；
                                #   保留於 CSV 供 make 階段與檢視用（與 model._EXCLUDE_COLS 雙重防護）
        "market_index",        # 與 model._EXCLUDE_COLS 既有項目對齊（idempotent）
        "clu_id_local",        # ★ 606_corr 群組編號：跨期 label switching 無對應意義，不當訓練
                                #   特徵；若 database_make/ 之後從 2026_daily 移植含此欄的資料，
                                #   保留於 CSV 供下游集中風險控管（同一輪選股避免同一群組壓過多
                                #   倉位）用，訓練端仍排除（與 904_group/main_fix_0709.py 一致）
        "clu_id_daily",        # ★ 2026-07-22：同上，2024 年起的新欄名（2017–2023 為
                                #   clu_id_local）。此處保留顯式列舉僅為可讀性，實際防護是
                                #   model._EXCLUDE_PREFIXES 的 "clu_id" 前綴比對（雙重防護，
                                #   之後再改名不必回來改這裡）
    }

    # ── ★ 流動性篩選（805_group2 formula3_amount_mv_turnover）────
    # score = w1·Z(ln(amount)) + w2·Z(ln(market_value)) + (1-w1-w2)·Z(ln(amount/mv))
    # 每日依 score 排名，保留前 LIQ_KEEP_RATIO 比例的股票 → 整批（train+predict）
    # 一起篩選後才餵給 WalkForwardTrainer（本支線為純研究/回測，無「既有持倉」
    # 概念，不像 daily_model 需區分 train-only vs. predict-all）。
    # 最佳組合來源：Optuna 搜尋 tag f3_w101004_w201896_kr02000
    #   （D1 ann_return=151.9% / sharpe=4.38 / obj=6.66）
    LIQ_FILTER_ENABLED: bool  = True
    LIQ_W1:             float = 0.1004
    LIQ_W2:             float = 0.1896
    LIQ_KEEP_RATIO:     float = 0.20
    # 篩選後資料快取目錄（依參數 tag 命名，非時間戳；同 tag 已存在則重用，
    # 避免每次重跑都重算，也避免殘留舊目錄造成混淆——805_group2 bug #5 教訓）
    LIQ_FILTERED_ROOT = Path("database_make_liq_filtered")

    # ── ★ Return-Magnitude Sample Weighting（rank-based，預設關閉）────
    # 開啟後訓練時報酬（LABEL）越高權重越大；若同時開 vol weight 則相乘正規化。
    USE_RETURN_WEIGHT: bool = False


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
#  ★ 流動性篩選（805_group2 formula3_amount_mv_turnover）
# ============================================================

def compute_liquidity_mask(
    df:         pd.DataFrame,
    w1:         float,
    w2:         float,
    keep_ratio: float,
    date_col:   str = "年月日",
) -> pd.Series:
    """
    formula3_amount_mv_turnover 流動性篩選（與 daily_fix_hyperparameter_model.py
    同一公式，維持兩端一致；此處為研究支線獨立副本，非共用 import）。

    score = w1·Z(ln(amount)) + w2·Z(ln(market_value))
            + (1-w1-w2)·Z(ln(amount/market_value))

    - Z(·) 為每日 cross-sectional z-score
    - 每日依 score 由大到小排名，保留前 keep_ratio 比例
    - amount / market_value 缺失、<=0、或當日 std=0（如僅 1 檔）
      → score = NaN → 明確不合格（不得意外通過篩選）

    Returns
    -------
    pd.Series[bool]，index 與 df 對齊；True = 通過篩選。
    """
    for col in ("amount", "market_value"):
        if col not in df.columns:
            raise KeyError(
                f"流動性篩選需要 '{col}' 欄，但 database_make/ 資料中找不到。"
                f"請確認 make_new.py 有保留該欄。"
            )

    amt = pd.to_numeric(df["amount"],       errors="coerce").where(lambda s: s > 0)
    mv  = pd.to_numeric(df["market_value"], errors="coerce").where(lambda s: s > 0)

    ln_amt = np.log(amt)
    ln_mv  = np.log(mv)
    ln_to  = np.log(amt / mv)          # turnover = amount / market_value

    grp = df[date_col]

    def _z(s: pd.Series) -> pd.Series:
        g   = s.groupby(grp)
        std = g.transform("std")
        return (s - g.transform("mean")) / std.where(std > 0)

    w3    = 1.0 - w1 - w2
    score = w1 * _z(ln_amt) + w2 * _z(ln_mv) + w3 * _z(ln_to)

    rank_pct = score.groupby(grp).rank(ascending=False, pct=True, method="first")
    mask     = score.notna() & (rank_pct <= keep_ratio)
    return mask


def liq_tag(w1: float, w2: float, keep_ratio: float) -> str:
    """篩選參數 → tag 字串（與 805_group2 summary 命名慣例一致）。
    e.g. w1=0.1004, w2=0.1896, kr=0.2000 → 'f3_w101004_w201896_kr02000'
    """
    def _fmt(x: float) -> str:
        return f"{x:.4f}".replace("0.", "").zfill(5)
    return f"f3_w1{_fmt(w1)}_w2{_fmt(w2)}_kr{_fmt(keep_ratio)}"


def _liq_cache_stale_reason(cfg: "RunConfig", out_dir: Path, years: list[int]) -> str | None:
    """
    比對 database_make/ 來源檔與快取檔的修改時間。
    若任一年份的來源檔比快取檔新（例如重跑過 make_new.py），
    代表快取內容已經過期、不能再重用 → 回傳過期原因字串；否則回傳 None。
    """
    for year in years:
        src = cfg.PRECOMPUTED_DIR / f"{year}.csv"
        cached = out_dir / f"{year}.csv"
        if not src.exists() or not cached.exists():
            continue
        if src.stat().st_mtime > cached.stat().st_mtime:
            return f"{src} 比快取檔 {cached} 新"
    return None


def build_liquidity_filtered_dir(cfg: "RunConfig") -> Path:
    """
    對 [ML_WINDOW_START, ML_WINDOW_END] 全範圍資料套用流動性篩選
    （train + predict 一起篩，本支線純研究/回測、無既有持倉概念），
    輸出成同樣的 per-year CSV 結構到 LIQ_FILTERED_ROOT/{tag}/，
    供 WalkForwardTrainer 直接當作 precomputed_dir 讀取。

    快取：若目錄已存在、涵蓋所需年份、且來源檔（database_make/）未被
    更新過 → 直接重用，不重算（與 test.py/optimize_turnover_filter.py
    的 SQLite 續跑精神一致）。若來源檔的修改時間比快取檔新（例如重跑過
    make_new.py 產生新特徵），則視為過期並自動重算，避免訓練用到舊資料。
    """
    tag     = liq_tag(cfg.LIQ_W1, cfg.LIQ_W2, cfg.LIQ_KEEP_RATIO)
    out_dir = cfg.LIQ_FILTERED_ROOT / tag
    years   = list(range(cfg.ML_WINDOW_START, cfg.ML_WINDOW_END + 1))

    existing = [out_dir / f"{y}.csv" for y in years if (out_dir / f"{y}.csv").exists()]
    if out_dir.exists() and len(existing) == len(years):
        stale_reason = _liq_cache_stale_reason(cfg, out_dir, years)
        if stale_reason is None:
            print(f"  ✓ 快取命中，重用已篩選資料: {out_dir}/（{len(years)} 年）")
            return out_dir
        print(f"  ⚠ 快取已過期（{stale_reason}），重新計算流動性篩選 → {out_dir}/")
    else:
        print(f"  快取未命中，重新計算流動性篩選 → {out_dir}/")
    print(f"  w1(amount)={cfg.LIQ_W1}  w2(mv)={cfg.LIQ_W2}  "
          f"w3(turnover)={1 - cfg.LIQ_W1 - cfg.LIQ_W2:.4f}  "
          f"keep_ratio={cfg.LIQ_KEEP_RATIO}")

    df = load_precomputed(cfg)   # 讀原始 database_make/（含 amount, market_value）
    mask = compute_liquidity_mask(df, cfg.LIQ_W1, cfg.LIQ_W2, cfg.LIQ_KEEP_RATIO)
    n_before = len(df)
    df = df[mask].copy()
    n_after = len(df)

    kept_per_day = df.groupby("年月日").size()
    print(f"  篩選結果: {n_before:,} → {n_after:,} 列 ({n_after / n_before:.1%})  "
          f"每日平均保留 {kept_per_day.mean():.0f} 檔")

    out_dir.mkdir(parents=True, exist_ok=True)
    for year, part in df.groupby(df["年月日"].dt.year):
        part = part.sort_values(["證券代碼", "年月日"])
        part.to_csv(out_dir / f"{year}.csv", index=False, encoding="utf-8-sig")
        print(f"    ✓ {out_dir}/{year}.csv  {part.shape}")

    meta = {
        "formula":       "formula3_amount_mv_turnover",
        "w1_amount":     cfg.LIQ_W1,
        "w2_market_value": cfg.LIQ_W2,
        "w3_turnover":   round(1 - cfg.LIQ_W1 - cfg.LIQ_W2, 4),
        "keep_ratio":    cfg.LIQ_KEEP_RATIO,
        "tag":           tag,
        "rows_before":   n_before,
        "rows_after":    n_after,
        "window_start":  cfg.ML_WINDOW_START,
        "window_end":    cfg.ML_WINDOW_END,
        "source":        "805_group2 summary.md（f3_w101004_w201896_kr01520）",
    }
    with open(out_dir / "liquidity_filter_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ★ 同步複製一份到 database/experiment/（backtest.py 預設 EXPERIMENT_DIR 的
    #   上層），供 backtest.py 自動偵測本次訓練用了哪組篩選參數，避免回測時
    #   誤以為是全市場訓練的結果。
    exp_meta_dir = Path("database/experiment")
    exp_meta_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_meta_dir / "liquidity_filter_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return out_dir


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
    # ── ★ 超參數決策 ─────────────────────────────────────────
    if cfg.TUNE_MODE == "frozen":
        print("\n► 超參數模式：frozen（不使用 Optuna，套用固定超參數）")
        best_params = load_best_params(cfg)
    else:
        _desc = {
            "every_fold": "每個 fold 都重跑 Optuna",
            "first_only": "第一個 fold tune，之後凍結沿用",
            "periodic":   f"每 {cfg.RETUNE_EVERY_N} 個 fold 重新 tune 一次",
        }[cfg.TUNE_MODE]
        print(f"\n► 超參數模式：{cfg.TUNE_MODE}（{cfg.N_TRIALS} trials）— {_desc}")
        best_params = None

    # ── ★ 訓練特徵集對齊（必須在 trainer.run() 之前）──────────
    if cfg.ALIGN_FEATURES_WITH_LIVE:
        print("\n► 特徵集對齊（train/serve consistency）")
        align_features_with_live(cfg)

    # ── ★ 流動性篩選（805_group2 formula3）──────────────────
    if cfg.LIQ_FILTER_ENABLED:
        print("\n► 流動性篩選（formula3_amount_mv_turnover）")
        train_precomputed_dir = build_liquidity_filtered_dir(cfg)
    else:
        train_precomputed_dir = cfg.PRECOMPUTED_DIR

    print("\n► Walk-Forward Training")
    ml_config = WalkForwardConfig(
        window_start      = cfg.ML_WINDOW_START,
        window_end        = cfg.ML_WINDOW_END,
        precomputed_dir   = train_precomputed_dir,
        target_col        = cfg.LABEL,
        n_trials          = cfg.N_TRIALS,
        use_vol_weight    = False,
        use_return_weight = cfg.USE_RETURN_WEIGHT,
        tune_mode         = cfg.TUNE_MODE,
        retune_every_n    = cfg.RETUNE_EVERY_N,
    )
    trainer = WalkForwardTrainer(ml_config)

    if cfg.TUNE_MODE == "frozen":
        # 預植固定超參數 → _should_tune 對所有 fold 回傳 False。
        # 若忘了給參數，run() 會直接拋 RuntimeError（不會靜默改跑 Optuna）。
        trainer.preload_params(best_params)

    trainer.run()

    # ── ★ 另存 tune 出的參數，供之後 TUNE_MODE="frozen" 重跑 ──
    #    注意：every_fold / periodic 模式下每折的參數都不同，這裡存的是
    #    **最後一折**的參數，不代表全程使用的超參數。逐折參數請看
    #    database/experiment/{window}/best_params.json。
    if cfg.TUNE_MODE == "first_only" and trainer._frozen_params:
        out_path = Path("best_params_regression.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trainer._frozen_params, f, ensure_ascii=False, indent=2)
        print(f"\n  ✓ 本次 tune 出的超參數已另存：{out_path}")
        print(f"    （之後可設 TUNE_MODE=\"frozen\" 並將 BEST_PARAMS_PATH 指向此檔，"
              f"跳過 tune 快速重跑）")
    elif cfg.TUNE_MODE in ("every_fold", "periodic"):
        print(f"\n  ▸ TUNE_MODE={cfg.TUNE_MODE}：各折超參數不同，未另存單一檔案。"
              f"逐折參數見 {ml_config.output_dir}/{{window}}/best_params.json")


def main():
    cfg = RunConfig()

    print(f"\n{'='*60}")
    print(f"  Walk-Forward Pipeline")
    print(f"  資料來源  : {cfg.PRECOMPUTED_DIR}/")
    print(f"  年份範圍  : {cfg.ML_WINDOW_START} ~ {cfg.ML_WINDOW_END}")
    print(f"  Label     : {cfg.LABEL}")
    if cfg.TUNE_MODE == "frozen":
        print(f"  Optuna    : 停用（TUNE_MODE=frozen，固定超參數）")
    else:
        print(f"  Optuna    : TUNE_MODE={cfg.TUNE_MODE}（{cfg.N_TRIALS} trials）")
    print(f"  對齊 live  : {cfg.ALIGN_FEATURES_WITH_LIVE}")
    if cfg.LIQ_FILTER_ENABLED:
        print(f"  流動性篩選: 開啟（formula3  w1={cfg.LIQ_W1}  w2={cfg.LIQ_W2}  "
              f"keep_ratio={cfg.LIQ_KEEP_RATIO}  tag={liq_tag(cfg.LIQ_W1, cfg.LIQ_W2, cfg.LIQ_KEEP_RATIO)}）")
    else:
        print(f"  流動性篩選: 關閉（全市場訓練）")
    print(f"{'='*60}")

    if not cfg.RUN_ML:
        print("  RUN_ML=False，結束。")
        return

    t0 = time.time()
    run_ml(cfg)
    print(f"\n  總耗時: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()