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
    # ★ 2026-08-04：2025 → 2026，納入今年到目前為止的資料當最後一折。
    # ⚠ 2026 是**不完整的年份**：database_make/2026.csv 到 2026-07-30，
    #   有標籤的交易日 135 天（其他年 242–247 天）。後果：
    #     · `ic_daily_t` 隨 √n 縮，跟其他年並排會被誤讀成「訊號變弱」
    #     · 年化報酬是把半年用 252 天外推，波動遠大於其他年
    #   → **2026 那折當「今年到目前為止的進度追蹤」看，不要放進 baseline 的
    #     逐年比較表。** 要回到只跑完整年，把這裡改回 2025。
    ML_WINDOW_END   = 2026

    # ── ★ 訓練視窗長度（2026-08-04，移植自 1107_window_length）──
    # 測試年 Y 的訓練年 = [Y-OFFSET-N, ..., Y-OFFSET-1]
    #   TRAIN_YEARS_N     = 訓練年數；None = expanding（從 ML_WINDOW_START 起）
    #   TRAIN_YEAR_OFFSET = 訓練視窗整體往前推幾年（0 = 緊貼測試年）
    #   DECAY_HALF_LIFE_MONTHS = 指數時間衰減樣本權重的半衰期；None = 關閉
    #
    # ★★ 2026-08-04：2 → 4，**對齊 2026_daily 線上的 `train_years=4`** ★★
    #
    # 2026_daily 已於 08-03（commit 823be5a）把每日排程改成 4 年，依據是
    # 1107_window_length（L2→L4 逐年 ic_daily +0.0066，噪音帶 ±0.0018）與
    # 1108_icdecay（「模型跟著死掉的因子走」的八個候選機制裡，只剩「訓練視窗
    # 太短」這個假說活著）。母體這邊還留在 2 年的話，之後每個複製出去的副本
    # 都在跟線上不同的視窗上做實驗——這正是 §0.1 說的「這裡的每個瑕疵都會被
    # 複製到未來所有實驗」。
    #
    # ⚠ 連帶後果，兩件事都還沒做：
    #   1. §5.8 的 baseline（database/experiment/）是 2 年視窗跑的，**與現行
    #      預設不可直接比較**。要沿用舊數字就顯式設 TRAIN_YEARS_N = 2。
    #   2. daily 的 FIXED_PARAMS 與本 repo §5.8 的逐折參數都是在 2 年視窗下
    #      調的。訓練列數約翻倍後，num_leaves / min_data_in_leaf / final_rounds
    #      這三項容量相關參數可能偏保守（daily model_config.py 自己標註
    #      「複核完成前屬暫代」）。→ backlog §8-#2。
    #
    # ⚠ DECAY_HALF_LIFE_MONTHS 留著是為了可複現 1107 的**負面**結論
    #   （更短視窗與時間衰減權重全部更差），不是推薦設定，見 §9-#18。
    TRAIN_YEARS_N:          int | None   = 4
    TRAIN_YEAR_OFFSET:      int          = 0
    DECAY_HALF_LIFE_MONTHS: float | None = None

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
    #   （D1 ann_return=151.9% / sharpe=4.38 / obj=6.66，此數字是純上市宇宙下的舊校準）
    # ★ 2026-08-18：0.20 → 0.25（1209_otcclu 切片實驗，上市+上櫃合併宇宙）。
    #   母體擴張後在 database_make_otc/ 上重跑 7 組 kr×k 全 12 年 walk-forward+
    #   回測，kr=0.25 在 net Sharpe/年化報酬/ic_daily 全面優於 0.20（net Sharpe
    #   1.782 vs 1.667），收緊到 0.10/0.15 則全面變差——與「候選池變大該收緊」
    #   的直覺相反。clu_ 分群數 k=18 同批驗證仍是最穩健的選擇，未變動。
    #   只有單一 seed，正式定案前建議多 seed 重跑確認；詳見 1209_otcclu/summary_otcclu.md §7。
    LIQ_FILTER_ENABLED: bool  = True
    LIQ_W1:             float = 0.1004
    LIQ_W2:             float = 0.1896
    LIQ_KEEP_RATIO:     float = 0.25
    # 篩選後資料快取目錄（依參數 tag 命名，非時間戳；同 tag 已存在則重用，
    # 避免每次重跑都重算，也避免殘留舊目錄造成混淆——805_group2 bug #5 教訓）
    LIQ_FILTERED_ROOT = Path("database_make_liq_filtered")
    # ★ 2026-07-27：快取驗證嚴格度。False（預設）＝只比對來源檔的
    #   (size, mtime, 表頭雜湊)，讀一行、幾乎不花時間，足以抓到「換過資料」
    #   與「欄位集變了」這兩種真正咬人的情況。
    #   True ＝雜湊整份檔案內容（database_make/ 約 3.2 GB，每次檢查多花數秒到
    #   數十秒磁碟讀取）。只在「懷疑欄位沒變但值變了」時才需要開。
    LIQ_CACHE_STRICT_HASH: bool = False

    # ── ★ 行情凍結（下市殭屍股）剔除（2026-08-04）──────────────
    # 同一檔股票連續 STALE_WINDOW 個交易日 close 與 amount 逐字相同 → 視為
    # 已下市／長期停牌被 ffill，該段列從宇宙中剔除（訓練與測試都剔）。
    # 詳見 objective/liquidity.py::find_stale_rows 的說明與 6806 案例。
    #
    # 實測影響（篩選後宇宙）：2025 年 254 列 / 9 檔（0.50%）、
    # 2026 年 73 列 / 2 檔（0.25%，含 6806）。列數佔比小，但它們是**恆定報酬
    # 且必然通過流動性濾網**的樣本，且在測試端會被排進 decile。
    #
    # ⚠ 開關會改變訓練宇宙 → 快取目錄名與 window 目錄名都會帶標記
    #   （`..._nz` 與 `..._L4z`），開關前後的產出不會互相覆蓋，也不會被
    #   backtest 默默混在一起（見 backtest_0709.py 的混用偵測）。
    EXCLUDE_STALE_QUOTES: bool = True
    STALE_WINDOW:        int  = 5

    # ── ★ Return-Magnitude Sample Weighting（rank-based，預設關閉）────
    # 開啟後訓練時報酬（LABEL）越高權重越大；若同時開 vol weight 則相乘正規化。
    USE_RETURN_WEIGHT: bool = False

    # ── ★ Inverse-Vol Sample Weighting（2026-08-04 新增旋鈕）────
    # ⚠ **這一項兩邊現在不一致，而且是真落差，不是刻意分歧：**
    #   daily 線上 `DailyConfig.use_vol_weight = True`（cv_metrics.json 的
    #   `vol_weight_used` 逐日為 true 可證），research 這邊 main_fix_0709.py
    #   從以前就寫死 False，連旋鈕都沒有。
    #   → §5.8 baseline 與線上模型在**樣本權重**上就不同，IC/RMSE 本來就不是
    #     完全可比。2026-08-04 先把它變成可設定的旋鈕並**維持現行值 False**
    #     （不偷改 baseline 行為）；要不要對齊成 True 是實驗決策，見 §8 backlog。
    #
    # ⚠ 不要跟 USE_RETURN_WEIGHT 搞混——那兩件事的證據強度差很多：
    #   · use_return_weight（rank(y) 加權，即「標籤的函數」）：daily summary §10.2
    #     有 154 日配對實驗，開權重逐日 IC −0.0329 vs 關 +0.0651，ΔIC +0.0980
    #     (t=6.63, p=5.4e-10)，107/154 天關的較高，逐年同向。**結論明確：不要用**，
    #     兩邊現在都是 False。
    #   · use_vol_weight（inverse-vol，即「特徵側的穩健性加權」）：daily summary
    #     §10.6 明列為**待辦**——「尚未做 on/off 對照」，只有 154 日粗測
    #     （都不開 +0.0651 / 只開 vol +0.0729，vol 略優 +0.0078），沒有配對檢定。
    #     daily 當初開啟的理由寫的是「對齊 research 預設」，指的是
    #     objective/model.py::WalkForwardConfig.use_vol_weight 的 dataclass 預設
    #     （確實是 True），但 research **實際跑的**一直是 main_fix 寫死的 False。
    #     兩邊 summary 都沒記到這一層，所以這個分歧存在多久沒人知道。
    USE_VOL_WEIGHT: bool = False

    # ── ★ 產出目錄與亂數種子（2026-08-04，移植自 1107）──────────
    # 副本裡跑多組設定時要把每組的 predictions 分開放，否則 backtest 會把不同
    # 設定的 window 混在一起讀（backtest_0709.py 是 glob 整個目錄）。
    EXPERIMENT_DIR = Path("database/experiment")
    RANDOM_STATE:   int = 42


