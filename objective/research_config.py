"""
research_config.py
==================
walk-forward 研究管線的設定（`RunConfig`）。對應 `2026_daily` 的
`daily_model/model_config.py`——設定住在 package 裡，不住在入口腳本裡。

★ 2026-07-27 從 `main_fix_0709.py` 搬過來（參考 2026_daily 的分層）。

搬的理由跟 §3.3 拆掉 monkeypatch 是同一個：本 repo 的實驗是**整包複製資料夾**
（summary_research.md §0.1），而且複製後常另寫一支 `main_fix_*.py`。設定長在
入口腳本裡時，每個副本、每支新入口都各自 fork 一份預設值——之後在本資料夾修
好的東西（例如快取驗證、排除清單）不會被新入口繼承，得靠人記得抄。設定集中在
這裡之後，新入口只要 import 再覆寫要動的那幾項。

用法——實驗要改設定時**繼承後覆寫**，不要改這個檔（改這裡等於改所有實驗的基準）：

    from objective.research_config import RunConfig

    class MyRunConfig(RunConfig):
        LIQ_KEEP_RATIO = 0.15
        N_TRIALS       = 30

    cfg = MyRunConfig()

⚠ 這個 class 是 `2026_research` 專屬的（daily 沒有對應檔案），不列入兩邊
`objective/` 的對拍範圍——見 `tools/check_objective_sync.py` 的 `RESEARCH_ONLY`。
"""

from __future__ import annotations

from pathlib import Path


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
    # ★ 2026-07-27：快取驗證嚴格度。False（預設）＝只比對來源檔的
    #   (size, mtime, 表頭雜湊)，讀一行、幾乎不花時間，足以抓到「換過資料」
    #   與「欄位集變了」這兩種真正咬人的情況。
    #   True ＝雜湊整份檔案內容（database_make/ 約 3.2 GB，每次檢查多花數秒到
    #   數十秒磁碟讀取）。只在「懷疑欄位沒變但值變了」時才需要開。
    LIQ_CACHE_STRICT_HASH: bool = False

    # ── ★ Return-Magnitude Sample Weighting（rank-based，預設關閉）────
    # 開啟後訓練時報酬（LABEL）越高權重越大；若同時開 vol weight 則相乘正規化。
    USE_RETURN_WEIGHT: bool = False


