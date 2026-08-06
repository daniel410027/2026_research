"""
model.py
========
Walk-forward LightGBM 訓練模組（FinLab 新專案版）。

變更（vs 舊版）：
    - WalkForwardConfig 移除 raw_data_dir / market_data_path
      （資料改由 FinLab API 提供，PreprocessConfig 自動轉換日期）
    - 所有 import 改為 objective 套件相對 import
    - 其他邏輯（walk-forward 流程、vol weight、cache 注入）完全不變

    [2025 更新] 超參數凍結加速：
    - WalkForwardConfig 新增 freeze_hyperparams / retune_every_n
    - 第一個 fold 完整 Optuna tuning，後續 fold 沿用凍結超參數
    - 沿用時改呼叫 collect_oof_prob（只跑 CV，不做 Optuna）
    - metrics.json 新增 hyperparams_tuned 欄位標記是否有重新 tune

    [2026-07-22 修正] 上述 freeze_hyperparams / retune_every_n 兩旗標已由單一
    tune_mode 取代（"every_fold" / "first_only" / "periodic" / "frozen"）：
    - 舊版 _should_tune 的第一個分支 `if not freeze_hyperparams: return True`
      會在檢查 _frozen_params 之前就短路，使「預植固定超參數」完全失效——
      Optuna 照跑、預植參數被靜默丟棄，畫面卻印「Optuna 全程停用」。
    - frozen 模式改為在 run() 開頭 fail fast（未預植直接拋 RuntimeError）。
    - 舊旗標若仍被傳入，於 __post_init__ 拋出遷移說明。
    - metrics.json 另新增 tune_mode 欄位。

    [2026 修正] CV 時間軸 bug：
    - 問題：_preprocess_window 以 ["證券代碼","年月日"] 排序（stock-major），
            train_df 沿用該順序丟進 TimeSeriesSplit，導致 CV 折實際在「按股票
            分塊」而非「按時間切」（驗證集 ~99.8% 為 train 未見股票）。
    - 修正：_run_window 在切 X/y/weight 前，先將 train_df / test_df 依「年月日」
            排序並 reset_index；並把 dates 傳入 tune_lgb / collect_oof_prob，
            由 lgb_utils 以「唯一交易日」為單位切折（見 lgb_utils._make_cv_folds），
            消除 panel 同日 cross-sectional 邊界外洩。

    [2026-06 對齊] _EXCLUDE_COLS 納入 amount / market_return_fwd：
    - 將「排除非訓練欄」收回本模組，與 daily_model.py 用同一訓練特徵集。
    - 此檔在 2026_daily 與 2026_research 兩 repo 通用（內容相同即可同步）。

    [2026-07-27 改版] 訓練特徵排除清單改由 config 傳入：
    - WalkForwardConfig 新增 extra_exclude_cols / extra_exclude_prefixes，
      模組層的 _EXCLUDE_COLS / _EXCLUDE_PREFIXES 降格為「基底預設值」。
    - 取代舊的 main_fix_0709.py::align_features_with_live() monkeypatch
      （它改的是模組全域，因此入口腳本被迫用 importlib 控制載入順序）。
    - 每個 window 另存 run_manifest.json（程式碼指紋 + 完整 config + 環境），
      讓「這份產出是什麼跑出來的」可事後查核，見 run_manifest.py 檔頭。

    [2026-08-04 移植自 1107_window_length] 訓練視窗長度可設定：
    - WalkForwardConfig 新增 train_years_n / train_year_offset /
      decay_half_life_months，**預設值 (2, 0, None) 即改版前寫死的行為**，
      換寫法不改結果（tools/regression_check_window.py 對本 repo 既有的
      2023_2025 折驗證 ic / ic_daily / rmse / n_test 逐位元相同）。
    - 訓練年一律由 cfg.train_years(test_year) 推導（唯一推導處），windows()
      只負責決定「哪些年可以當測試年」；window 目錄名仍是
      f"{訓練首年}_{測試年}"。
    - train_year_offset > 0 是「樣本數不變、只把資料變舊」的控制組。
    - decay_half_life_months 給指數時間衰減樣本權重，與 vol/return weight
      相乘後正規化。⚠ 1107 實測衰減權重**更差**，留著是為了可複現該結論，
      不是推薦設定（見 summary_research.md §9-#18）。

Walk-forward 規則（由 train_years_n / train_year_offset 決定；
研究端現行預設 4 年，對齊 2026_daily 線上的 train_years=4）：
    Windows: (訓練首年, 測試年)，例如 n=4 時 (2014,2018), (2015,2019), ...
    每個 window：
        train = 測試年往前 train_years_n 年
        test  = 測試年

輸出（每個 window 存至 database/experiment/YYYY_YYYY/）：
    predictions.csv
    metrics.json
    feature_importance.csv
    best_params.json

作者：Daniel Huang
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

from .lgb_utils import (
    tune_lgb, train_final_lgb, evaluate_oof_ic, build_imp_df, collect_oof_prob,
)
from .run_manifest import write_run_manifest

# preprocess 僅在 inject_cache 路徑使用，lazy import 避免 __init__.py 問題
def _lazy_preprocessor():
    from .preprocess import PreprocessConfig, DataPreprocessor
    return PreprocessConfig, DataPreprocessor

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
#  特徵清單
# ─────────────────────────────────────────────────────────────

_FEATURES_CSV_PATH = Path("database/features_only.csv")

_NEW_FEATURES = ["beta", "vol20", "daily_return",
                 "CAPM_Beta一月", "CAPM_Beta九月", "CAPM_Beta一年"]

# ★ 2026-07-27：以下兩個模組層常數是**基底清單（預設值）**，不再是「跑的時候
#   實際生效的那一份」。實際生效的是 WalkForwardConfig.resolved_exclude_cols() /
#   resolved_exclude_prefixes()＝基底 ∪ config 的 extra_*。
#
#   改動原因：舊版由 main_fix_0709.py 的 align_features_with_live() 直接
#   monkeypatch 這個模組全域（`_wf_model._EXCLUDE_COLS = before | extra`），而那
#   又要求入口腳本用 importlib 把 objective.model 塞進 sys.modules，才能保證
#   patch 到 trainer 真正用的那份模組。整條鏈是隱性的：新入口腳本只要沒照抄
#   那套載入順序，排除清單就靜默失效——和 §6.1「宣稱凍結、實際照跑 Optuna」
#   同一類 bug，而且因為實驗是整包複製出去的（§0.1），每份副本都繼承這個陷阱。
#   改成 config 欄位後，排除清單跟著 config 走、會進 metrics/run_manifest，
#   入口腳本用一般 import 即可。
_EXCLUDE_COLS = {
    "證券代碼", "年月日",
    "return", "return_tick", "return_tick_0",
    "excess_return", "excess_return_tick",
    "market_return_fwd",
    "amount", "close",
    "market_value",
    # ★ 2026-07-27（2026_daily 移植）：daily 端 07-17 起就排除這兩欄，research
    #   一直沒跟上。clu_id_daily 在本 repo 已被 _EXCLUDE_PREFIXES 的 "clu_id"
    #   蓋掉（列在這裡是為了與 daily 同形，不改變行為）；真正的行為變更是
    #   clu_valid——§6.4 實測 `clu_valid==1` 的集合與「clu_ 有值」的集合逐筆
    #   完全相同（2021 年 45,766/45,766 一筆不差），在 clu_ 改為保留 NaN 之後
    #   它與缺失模式 100% 冗餘（importance 恆為 0）。
    "clu_id_daily", "clu_valid",
}

# ★ 2026-07-22：前綴排除。606_corr 的群組編號欄名跨期會變（2017–2023 為
#   clu_id_local、2024 起改名 clu_id_daily），逐一列舉在 _EXCLUDE_COLS 會漏——
#   實際上 database_make_liq_filtered/ 的 2025.csv 兩個欄名同時存在，而排除清單
#   只有 clu_id_local，導致 clu_id_daily 以 importance 0.152 / rank 41 進入訓練。
#   群組編號是任意標籤（label switching），跨期無對應意義，不應作為數值特徵。
#   改用前綴比對，之後再改名也不會漏。
_EXCLUDE_PREFIXES = (
    "clu_id",
)

# ★ 2026-07-27（2026_daily 移植，修掉 §6.2）：以下前綴的欄位**不做 fillna(0)**，
#   保留 NaN 交給 LightGBM 的原生缺失分支。
#   理由：0 落在 clu_ 特徵的合法值域內（`clu_rank_ret_1` 實際值域 [0.05, 1.0]、
#   真正等於 0 的筆數為 0），fillna(0) 等於送進一個乾淨的哨兵值，模型一刀切在
#   0 附近就能還原「這筆有沒有 clu 資料」。而 §6.4 已查明那個缺失模式本身帶
#   真實資訊（⟺「這檔是當天臨時擠進流動性前 20%、不屬於 clu 穩定核心的股票」），
#   於是缺失指示變數與 clu_ 的實際內容被混在同一組欄位裡，importance 無法解讀。
#   daily 端 07-22 就改了（daily_model/model_config.py 同名常數），research 落後。
#   ⚠ 只動訓練矩陣的填補；objective/features/*.py 裡的 fillna(0) 是特徵構造的
#   一部分，語意不同，不在此列。
_KEEP_NAN_PREFIXES: tuple[str, ...] = ("clu_",)


def _fill_missing(X: pd.DataFrame) -> pd.DataFrame:
    """對 _KEEP_NAN_PREFIXES 以外的欄位做 fillna(0)，clu_ 系列保留 NaN。"""
    keep_nan = [c for c in X.columns if c.startswith(_KEEP_NAN_PREFIXES)]
    if not keep_nan:
        return X.fillna(0)
    fill_cols = X.columns.difference(keep_nan, sort=False)
    return pd.concat([X[fill_cols].fillna(0), X[keep_nan]], axis=1)[X.columns]


# [2026-07-15] 與 daily_model_0714_fix.py 的 _EXCLUDE_COLS 統一為同一組，
# 確保 walk-forward 回測與線上 DailyTrainer 用完全相同的訓練特徵集,結果才能互相比較。
#
# 與舊版本檔的差異（決策記錄，供日後回頭檢視）：
#   - market_return：舊版本檔排除（理由：per-day constant,兩年 train 僅 ~490 個唯一
#     交易日,樹模型可能靠它記住特定日期的平均 excess return,構成 date-level
#     overfitting 風險）。daily_model_0714_fix.py 現行判斷是「應該沒有資料洩漏問題」
#     而不排除。這裡選擇跟 daily_model 一致,但上述過擬合疑慮本身並未被推翻——
#     如果之後 walk-forward 的 IC／RMSE 明顯優於「排除 market_return」時的舊結果、
#     或 feature_importance 顯示 market_return 排名異常前面,要回頭檢查是不是在吃
#     日期指紋而非真實訊號。
#   - close / market_value：舊版本檔沒排除（等同讓 WalkForwardTrainer 拿這兩欄
#     當特徵）。daily_model_0714_fix.py 與 feature_mixin.py 的既有設計是把它們當
#     「保留於 CSV、不進訓練」的 conditioning / benchmark 欄位,這裡改為與其一致排除。
#   - market_index / 市場別：兩者在 database_make/ 目前分別是「不存在」與「非數值
#     字串欄」,原本的排除本來就是 no-op,拿掉不影響行為。
#
# 注意：database/experiment/*/ 內既有的 walk-forward 結果是用舊特徵集跑出來的,
# 跟本次改動後重跑的結果不可直接比較,需要的話重新跑一次 windows 取得新基準。


def _daily_ic(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    dates:       np.ndarray,
    min_per_day: int = 30,
) -> tuple[float, float, int]:
    """
    逐日 cross-sectional Spearman IC 的平均值、t 統計量與有效天數。

    ★ 2026-07-27 新增。與 pooled IC（把所有日期混在一起算一次）的差別：
      pooled 會把「日間變異」算進去，但每日 top-N 選股只用得到「日內排序」。
      可查證的量級對照（2026_research §5.1，walk-forward test set）：
      2020_2022 折 pooled 0.1655 vs 逐日 0.0780、2023_2025 折 0.1043 vs 0.0518，
      整體 pooled 平均約為逐日的 1.6 倍。評估排序品質請用本函式。
      ⚠ 檔頭另有一句「daily 端曾出現 pooled=+0.063 而逐日=-0.022（符號相反）」，
        那是 07-27 當下的臨時比對，**沒有留下對應產出**：daily 的
        database/daily_predict/*/cv_metrics.json 全部是改版前的 pooled 值，
        區間 -0.043 ~ +0.033，找不到 +0.063 那一筆。方向可信、數字待重現。

    單日有效樣本少於 min_per_day 即略過該日，避免小樣本 IC 噪音灌入平均。
    """
    daily = []
    for d in np.unique(dates):
        sel = dates == d
        if sel.sum() < min_per_day:
            continue
        ic_d, _ = spearmanr(y_true[sel], y_pred[sel])
        if not np.isnan(ic_d):
            daily.append(ic_d)

    if not daily:
        return float("nan"), float("nan"), 0

    arr  = np.asarray(daily)
    mean = float(arr.mean())
    if len(arr) > 1:
        std   = float(arr.std(ddof=1))
        tstat = mean / (std / np.sqrt(len(arr))) if std > 0 else float("nan")
    else:
        tstat = float("nan")
    return mean, tstat, len(arr)


def load_feature_list(exclude_cols: set[str] | None = None) -> list[str]:
    """讀取 features_only.csv，附加新特徵，回傳最終候選清單。

    exclude_cols 未給時退回基底 `_EXCLUDE_COLS`；由 trainer 呼叫時會傳入
    config 解析後的完整清單（含 extra_exclude_cols）。
    """
    if not _FEATURES_CSV_PATH.exists():
        raise FileNotFoundError(f"找不到特徵清單: {_FEATURES_CSV_PATH}")

    excl   = _EXCLUDE_COLS if exclude_cols is None else exclude_cols
    raw    = pd.read_csv(_FEATURES_CSV_PATH)["feature"].tolist()
    mapped = raw + _NEW_FEATURES

    seen, final = set(), []
    for f in mapped:
        if f not in seen and f not in excl:
            seen.add(f)
            final.append(f)
    return final


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    """Walk-forward 訓練設定。"""

    # ── Walk-forward 視窗 ─────────────────────────────────────
    window_start: int = 2014
    window_end:   int = 2025

    # ── ★ 訓練視窗長度（2026-08-04，移植自 1107_window_length）───
    # 測試年 Y 的訓練年 = [Y-offset-n, ..., Y-offset-1]
    #   train_years_n      = 訓練年數；None = expanding（從 window_start 起算）
    #   train_year_offset  = 把訓練視窗整體往前推幾年（0 = 緊貼測試年）
    # ⚠ 這裡刻意維持 (2, 0)＝改版前寫死的 train=[start, start+1]、test=start+2，
    #   regression_check_window.py 才有東西可對。**要對齊線上的 4 年是設定層的
    #   決定**，見 objective/research_config.py::RunConfig.TRAIN_YEARS_N = 4。
    train_years_n:     int | None = 2
    train_year_offset: int        = 0

    # ── 目標欄位 ──────────────────────────────────────────────
    target_col: str = "excess_return"

    # ── 特徵設定 ──────────────────────────────────────────────
    use_features_csv: bool = False

    # ── ★ 訓練特徵排除（2026-07-27：由 monkeypatch 改為 config）───────
    # 基底是模組層的 _EXCLUDE_COLS / _EXCLUDE_PREFIXES，下面兩欄是**額外**
    # 疊加的部分（聯集，不是取代）。實際生效清單見 resolved_exclude_cols()。
    #
    # 典型用途（objective/research_config.py::RunConfig.EXTRA_EXCLUDE_COLS）：把
    # market_value / close 這類「保留於 CSV、但不可進訓練」的外部條件變數排掉。
    # 排除清單屬於**實驗設定**——換一組就是換一個實驗——所以它該跟其他設定一樣
    # 走 config、被記進 metrics.json 與 run_manifest.json，而不是靠改全域狀態。
    extra_exclude_cols:     frozenset[str] = frozenset()
    extra_exclude_prefixes: tuple[str, ...] = ()

    # ── Optuna / CV ───────────────────────────────────────────
    n_trials:     int = 15
    n_splits:     int = 3
    random_state: int = 42

    # ── 超參數策略 ────────────────────────────────────────────
    # ★ 2026-07-22：原本用 freeze_hyperparams(bool) + retune_every_n(int) 兩個旗標
    #   表達，但兩者交互出的狀態空間有一半是無效的，且 _should_tune 的第一個分支
    #   會在檢查 _frozen_params 之前就短路，導致 freeze_hyperparams=False 時
    #   「預植固定超參數」完全失效（Optuna 照跑、預植參數被靜默丟棄，畫面卻印
    #   「Optuna 全程停用」）。改用單一 tune_mode 列舉，狀態互斥且語意明確。
    #
    #   "every_fold" ── 每個 fold 都重跑 Optuna（每年重訓超參）
    #   "first_only" ── 只有第一個 fold tune，之後所有 fold 凍結沿用
    #   "periodic"   ── 每 retune_every_n 個 fold 重新 tune 一次
    #   "frozen"     ── 完全不 tune，用 preload_params() 預植的超參數。
    #                   未預植就跑會直接拋錯（不會靜默改跑 Optuna）。
    tune_mode:      str = "first_only"
    retune_every_n: int = 3        # 僅 tune_mode="periodic" 時有意義

    # ★ 已移除的舊旗標。若舊程式碼仍傳入，於 __post_init__ 拋出遷移說明，
    #   而不是讓 dataclass 丟一個看不出原因的 TypeError。
    freeze_hyperparams: bool | None = None

    # ── Inverse-Volatility Sample Weighting ───────────────────
    use_vol_weight:  bool  = True
    vol_weight_col:  str   = "vol20"
    vol_weight_clip: float = 0.05

    # ── ★ 指數時間衰減樣本權重（2026-08-04，移植自 1107）──────
    # None = 關閉（改版前行為）。設為半衰期月數 H 時：
    #     w_i = 0.5 ** (age_days_i / (H * 30.44))
    # age_days 以該 fold 訓練集**最後一個交易日**為 0（不是預測日，因為
    # 訓練集末端與測試年之間沒有資料，用哪一端當基準只差一個常數倍率，
    # 而權重最後會正規化到 mean=1，常數倍率無影響）。
    decay_half_life_months: float | None = None

    # ── Return-Magnitude Sample Weighting（rank-based，可獨立開關）──
    # 報酬（target_col）越高，訓練時權重越大；與 vol weight 同樣採 rank-based
    # 而非用 raw 數值，避免極端報酬把 RMSE 已有的離群值敏感度再放大一次。
    # 若 use_vol_weight 也開啟，兩者 rank weight 相乘後再正規化。
    use_return_weight: bool = False

    # ── 輸出路徑 ──────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: Path("database/experiment"))

    # ── 資料庫路徑 ───────────────────────────────────────────
    db_dir: Path = field(default_factory=lambda: Path("database"))

    # ── 預計算特徵目錄（make_features.py 輸出）────────────────
    precomputed_dir: Path = field(default_factory=lambda: Path("database_make"))

    # ── Preprocess 共用設定 ──────────────────────────────────
    return_clip:     float = 0.15
    tick_threshold:  float = 0.01
    tick0_threshold: float = 0.00
    beta_window:     int   = 60

    _TUNE_MODES = ("every_fold", "first_only", "periodic", "frozen")

    def __post_init__(self):
        self.output_dir      = Path(self.output_dir)
        self.db_dir          = Path(self.db_dir)
        self.precomputed_dir = Path(self.precomputed_dir)

        # 允許傳 set / list，正規化成 hashable 且不可變的形式（config 會被
        # dump 進 run_manifest，可變預設值在 dataclass 也是常見地雷）
        self.extra_exclude_cols     = frozenset(self.extra_exclude_cols)
        self.extra_exclude_prefixes = tuple(self.extra_exclude_prefixes)

        if self.freeze_hyperparams is not None:
            raise ValueError(
                "freeze_hyperparams 已於 2026-07-22 移除，請改用 tune_mode：\n"
                "  freeze_hyperparams=False              → tune_mode='every_fold'\n"
                "  freeze_hyperparams=True, 不預植參數   → tune_mode='first_only'\n"
                "  freeze_hyperparams=True, 預植參數     → tune_mode='frozen'\n"
                "                                          + trainer.preload_params(p)\n"
                "  retune_every_n < 999                  → tune_mode='periodic'\n"
                "（舊寫法的實際行為與字面意思不符，詳見 2026_research/summary_research.md §6.1）"
            )

        if self.tune_mode not in self._TUNE_MODES:
            raise ValueError(
                f"tune_mode={self.tune_mode!r} 無效，可選：{self._TUNE_MODES}"
            )
        if self.tune_mode == "periodic" and self.retune_every_n < 1:
            raise ValueError(f"retune_every_n 需 >= 1，得到 {self.retune_every_n}")

    def train_years(self, test_year: int) -> list[int]:
        """測試年 → 訓練年清單（唯一的推導處，所有呼叫端都走這裡）。"""
        last = test_year - self.train_year_offset - 1        # 訓練視窗最後一年
        if self.train_years_n is None:                       # expanding
            first = self.window_start
        else:
            first = last - self.train_years_n + 1
        if first < self.window_start or last < first:
            return []
        return list(range(first, last + 1))

    def windows(self) -> list[tuple[int, int]]:
        """
        回傳 (訓練首年, 測試年)。第一個元素只用來組 window 目錄名，
        實際訓練年一律由 train_years(test_year) 決定——offset > 0 時
        兩者不再相鄰，看目錄名就知道這一輪是哪個設定。
        """
        out = []
        for test_year in range(self.window_start, self.window_end + 1):
            tr = self.train_years(test_year)
            if tr:
                out.append((tr[0], test_year))
        return out

    # ── ★ 訓練特徵排除：實際生效清單 ─────────────────────────
    def resolved_exclude_cols(self) -> frozenset[str]:
        """基底 `_EXCLUDE_COLS` ∪ `extra_exclude_cols`。"""
        return frozenset(_EXCLUDE_COLS) | self.extra_exclude_cols

    def resolved_exclude_prefixes(self) -> tuple[str, ...]:
        """基底 `_EXCLUDE_PREFIXES` ∪ `extra_exclude_prefixes`（去重、排序穩定）。"""
        return tuple(dict.fromkeys(_EXCLUDE_PREFIXES + self.extra_exclude_prefixes))


# ─────────────────────────────────────────────────────────────
#  TRAINER
# ─────────────────────────────────────────────────────────────

class WalkForwardTrainer:

    def __init__(self, config: WalkForwardConfig):
        self.cfg = config

        # ★ 2026-07-27：排除清單在建構時就固定下來（來源是 config，不再是可被
        #   外部 monkeypatch 的模組全域），之後 _resolve_features 只讀這兩份。
        self._exclude_cols     = config.resolved_exclude_cols()
        self._exclude_prefixes = config.resolved_exclude_prefixes()
        if config.extra_exclude_cols or config.extra_exclude_prefixes:
            print(f"  訓練排除: 基底 {len(_EXCLUDE_COLS)} 欄 + 額外 "
                  f"{sorted(config.extra_exclude_cols)}"
                  + (f" + 額外前綴 {list(config.extra_exclude_prefixes)}"
                     if config.extra_exclude_prefixes else ""))

        if config.use_features_csv:
            self.feature_list = load_feature_list(self._exclude_cols)
            print(f"  特徵模式: features_only.csv  ({len(self.feature_list)} 個候選)")
        else:
            self.feature_list = None
            print("  特徵模式: 全部數值欄位（排除 labels）")

        self._df_cache: dict[tuple[int, int], pd.DataFrame] = {}

        # 超參數凍結快取（tune_mode="frozen" 時由 preload_params() 填入）
        self._frozen_params: dict | None = None

    def inject_cache(self, start: int, end: int, df: pd.DataFrame):
        """外部注入已處理的 df，跳過重複前處理。"""
        key = (start, end)
        self._df_cache[key] = df.dropna(subset=[self.cfg.target_col])
        print(f"  ✓ 注入 df cache：window ({start}, {end})，形狀 {self._df_cache[key].shape}")

    def preload_params(self, params: dict):
        """預植固定超參數（tune_mode="frozen" 必須先呼叫）。"""
        if not params:
            raise ValueError("preload_params 收到空的超參數")
        self._frozen_params = dict(params)
        print(f"  ✓ 已預植固定超參數（{len(self._frozen_params)} 項）")

    def run(self):
        windows = self.cfg.windows()

        # ★ fail fast：frozen 模式沒預植參數就直接停，不要靜默改跑 Optuna
        #   （這正是舊 freeze_hyperparams 版本的 bug，見 2026_research/summary_research.md §6.1）
        if self.cfg.tune_mode == "frozen" and self._frozen_params is None:
            raise RuntimeError(
                "tune_mode='frozen' 但沒有預植超參數。"
                "請在 run() 之前呼叫 trainer.preload_params(best_params)，"
                "或改用 tune_mode='first_only' / 'every_fold'。"
            )

        _mode_desc = {
            "every_fold": "每個 fold 都重跑 Optuna",
            "first_only": "第一個 fold tune，之後凍結沿用",
            "periodic":   f"每 {self.cfg.retune_every_n} 個 fold 重新 tune 一次",
            "frozen":     "完全不 tune，使用預植的固定超參數",
        }[self.cfg.tune_mode]

        print(f"\n{'='*60}")
        print(f"  Walk-Forward Training")
        print(f"  Windows : {windows[0]} → {windows[-1]}  ({len(windows)} 個)")
        print(f"  Target  : {self.cfg.target_col}")
        print(f"  Output  : {self.cfg.output_dir}")
        print(f"  超參數策略 : {self.cfg.tune_mode} — {_mode_desc}")
        print(f"{'='*60}")

        all_metrics = []
        for i, (start, end) in enumerate(windows, 1):
            print(f"\n[{i}/{len(windows)}] Window {start}–{end}")
            try:
                metrics = self._run_window(start, end, fold_index=i)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"  ✗ 失敗: {e}")
                import traceback; traceback.print_exc()

        self._save_summary(all_metrics)

    def window_dir_name(self, start: int, end: int) -> str:
        """
        產出資料夾名稱：`{訓練首年}_{測試年}_L{訓練年數}`，例如 `2022_2026_L4`。

        ★ 2026-08-04 加上 `_L{n}` 後綴。原本只有 `{start}_{end}`，在單一視窗
          長度下沒問題，但視窗長度一改就會出事——實際踩過：`database/experiment/`
          裡同時躺著 L2 的 `2016_2018` 與 L4 的 `2014_2018`（同一個測試年 2018
          的兩份預測），而 backtest_0709.py 是 glob 整個目錄再去重，於是
          **2016–2017 取到 L2 的預測、2018 以後取到 L4 的**，一張圖裡混了兩個
          模型，還不會報錯。

          加上後綴之後兩者名稱不同、且 backtest 端可以偵測到多種 L 並存而擋下來
          （見 backtest_0709.py::load_all_predictions）。

        expanding（train_years_n=None）記為 `Lexp`；train_year_offset > 0 時
        另加 `_off{k}`，否則位移過的視窗會與正常視窗撞名。
        """
        n   = self.cfg.train_years_n
        tag = f"L{n}" if n is not None else "Lexp"
        if self.cfg.train_year_offset:
            tag += f"_off{self.cfg.train_year_offset}"
        # ★ 剔除行情凍結列（下市殭屍股）→ 加 z。這改的是**宇宙**不是超參數，
        #   開關前後的 IC/Sharpe 不可比，必須從目錄名就看得出來。
        #   （由 precomputed_dir 的 tag 反推，因為 WalkForwardConfig 本身不帶
        #     這個設定——它吃的是已經篩好的目錄。）
        if "_nz" in str(self.cfg.precomputed_dir):
            tag += "z"
        return f"{start}_{end}_{tag}"

    def _preprocess_window(self, start: int, end: int) -> pd.DataFrame:
        key = (start, end)
        if key in self._df_cache:
            print(f"  ✓ 命中 df cache（{start}–{end}），跳過重複前處理")
            return self._df_cache[key]

        # 只讀 train_years + test_year（滾動特徵已跨年連續，無需重複讀 start 年）
        # 經 test_year_boundary.py 驗證：vol20 / beta / CAPM_Beta 系列 jump_ratio < 2x
        # ★ train_year_offset > 0 時訓練年與測試年之間會有缺口，這裡只是把
        #   各年 CSV 疊起來（特徵已預先算好、不在本函式重算），缺年無妨。
        test_year   = end
        train_years = self.cfg.train_years(test_year)
        years_to_load = train_years + [test_year]

        frames = []
        for year in years_to_load:
            path = self.cfg.precomputed_dir / f"{year}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"找不到預計算特徵：{path}\n"
                    f"請先執行 make_features.py 產生 database_make/ 資料"
                )
            frames.append(pd.read_csv(path, low_memory=False))
            print(f"  讀取 {path}")

        df = pd.concat(frames, ignore_index=True)
        df["年月日"]   = pd.to_datetime(df["年月日"])
        df["證券代碼"] = df["證券代碼"].astype(str)
        df = df.sort_values(["證券代碼", "年月日"]).reset_index(drop=True)
        return df.dropna(subset=[self.cfg.target_col])

    def _should_tune(self, fold_index: int) -> bool:
        """判斷本 fold 是否需要執行 Optuna tuning（fold_index 由 1 起算）。"""
        mode = self.cfg.tune_mode

        if mode == "every_fold":
            return True

        if mode == "frozen":
            # 預植檢查已在 run() 開頭做過（fail fast），這裡只是永不 tune。
            return False

        # first_only / periodic：第一個 fold 一定要 tune 才有參數可用
        if self._frozen_params is None:
            return True

        if mode == "periodic":
            return (fold_index - 1) % self.cfg.retune_every_n == 0

        return False                                       # first_only，已 tune 過

    def _run_window(self, start: int, end: int, fold_index: int = 1) -> dict:
        test_year   = end
        train_years = self.cfg.train_years(test_year)

        print(f"\n► 前處理 Window {start}–{end}...")
        full_df = self._preprocess_window(start, end)
        print(f"  Window 資料形狀: {full_df.shape}")

        year_col = full_df["年月日"].dt.year

        # ── [2026 修正] 切 train / test 後，務必依「年月日」重新排序 ─────
        # _preprocess_window 回傳的是 stock-major 排序；若沿用該順序，
        # 下游 TimeSeriesSplit 會變成「按股票分塊」而非「按時間切」。
        # 這裡顯式 sort by 年月日 + reset_index，還原 CV 所需的時間軸。
        train_df = (
            full_df[year_col.isin(train_years)]
            .sort_values("年月日")
            .reset_index(drop=True)
        )
        test_df = (
            full_df[year_col == test_year]
            .sort_values("年月日")
            .reset_index(drop=True)
        )

        print(
            f"  Train {train_years}: {len(train_df):,} 筆  "
            f"mean={train_df[self.cfg.target_col].mean():.5f}  "
            f"std={train_df[self.cfg.target_col].std():.5f}"
        )
        print(
            f"  Test  {test_year} : {len(test_df):,} 筆  "
            f"mean={test_df[self.cfg.target_col].mean():.5f}  "
            f"std={test_df[self.cfg.target_col].std():.5f}"
        )

        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Train 或 Test 為空")

        feature_cols = self._resolve_features(full_df)
        print(f"  實際使用特徵: {len(feature_cols)} 個")

        X_train = _fill_missing(train_df[feature_cols])
        y_train = train_df[self.cfg.target_col]
        X_test  = _fill_missing(test_df[feature_cols])
        y_test  = test_df[self.cfg.target_col]

        # CV 切折依據（傳給 lgb_utils 以唯一交易日為單位切折）
        train_dates = train_df["年月日"]

        train_weight = self._compute_sample_weights(train_df)

        # ── 超參數決策 ────────────────────────────────────────
        do_tune = self._should_tune(fold_index)

        if do_tune:
            print(f"  [Optuna] 執行超參數搜尋（{self.cfg.n_trials} trials）...")
            best_params, oof_pred = tune_lgb(
                X_train, y_train,
                n_splits      = self.cfg.n_splits,
                n_trials      = self.cfg.n_trials,
                random_state  = self.cfg.random_state,
                sample_weight = train_weight,
                dates         = train_dates,
            )
            self._frozen_params = best_params
            if self.cfg.tune_mode in ("first_only", "periodic"):
                print(f"  [Optuna] 超參數已凍結，後續 fold 沿用")
        else:
            best_params = self._frozen_params
            print(f"  [Optuna] 沿用既有超參數（tune_mode={self.cfg.tune_mode}），"
                  f"跳過 tune，只收集 OOF 預測值")
            oof_pred = collect_oof_prob(
                X_train, y_train,
                best_params   = best_params,
                n_splits      = self.cfg.n_splits,
                sample_weight = train_weight,
                dates         = train_dates,
                random_state  = self.cfg.random_state,   # ★ 2026-08-06，見下
            )

        print("  訓練最終模型...")
        # ★ 2026-08-06 修正：這兩個呼叫端原本都沒傳 random_state。
        #
        #   `collect_oof_prob` / `train_final_lgb` 的簽名都有 `random_state: int|None
        #   = None`，內部也確實會 `params["seed"] = random_state`——但**只有在有傳的
        #   時候**。沒傳就是 None，`_LGB_BASE_PARAMS` 裡也沒有 seed，於是整條
        #   walk-forward 路徑吃的是 LightGBM 自己的預設種子：結果仍可重現，卻
        #   **完全不隨 cfg.random_state 變動**。
        #
        #   怎麼發現的：2026_tune 用 SEEDS=(42, 7, 2026) 各跑一遍同一組設定做跨 seed
        #   穩健性檢查，排名表上每一組的 ic_std 都是 0.000000——查下去發現三個 seed
        #   的 predictions.csv **byte 完全相同**。39 個 run 只有 13 組相異結果，
        #   約 4 小時是重複計算。
        #
        #   為什麼三盞燈沒照到：對拍比的是 research↔daily 的 objective/，而這個 bug
        #   兩邊**一模一樣地缺**（AST 相同 → 綠燈）。真正的對照組是
        #   daily_model/trainer.py——每日線上預測那條路徑三個呼叫端都有傳，
        #   所以**線上每日流程不受影響**，受影響的只有離線 walk-forward。
        #   對拍能抓「兩份副本漂掉」，抓不到「同一件事的兩個實作不一致」。
        #
        #   ⚠ 這個修正會改變 walk-forward 的數值（從 LightGBM 預設種子換成
        #     cfg.random_state=42）。差異是純種子差異、不是方法改變，但既有
        #     experiment/ 產出與修正後的**不可直接比較**。
        model = train_final_lgb(X_train, y_train, best_params, train_weight,
                                random_state=self.cfg.random_state)

        # ★ 2026-07-27：逐日 IC 平均（舊版 pooled 含日間變異，會系統性高估）
        oof_ic = evaluate_oof_ic(y_train, oof_pred, dates=train_dates)

        metrics, pred_df, imp_df = self._evaluate(
            model, X_test, y_test, feature_cols, test_df
        )
        metrics.update({
            "window":             f"{start}_{end}",
            "train_years":        str(train_years),
            "test_year":          test_year,
            "oof_ic":             oof_ic,
            "vol_weight_used":    self.cfg.use_vol_weight,
            "vol_weight_col":     self.cfg.vol_weight_col if self.cfg.use_vol_weight else None,
            "return_weight_used": self.cfg.use_return_weight,
            # ★ 1107_window_length：視窗設定也進 metrics，否則一堆 experiment_*
            #   目錄事後分不出誰是哪個設定跑的（複製資料夾做實驗時尤其容易混）。
            "train_years_n":      self.cfg.train_years_n,
            "train_year_offset":  self.cfg.train_year_offset,
            "decay_half_life_months": self.cfg.decay_half_life_months,
            "n_train":            len(train_df),
            "hyperparams_tuned":  do_tune,   # 本 fold 是否重新 tune
            "tune_mode":          self.cfg.tune_mode,
            "n_features":         len(feature_cols),
        })

        out_dir = self.cfg.output_dir / self.window_dir_name(start, end)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save_outputs(out_dir, metrics, best_params, pred_df, imp_df)

        # ★ 2026-07-27：記錄「這份產出是被什麼程式碼、什麼設定跑出來的」。
        #   見 objective/run_manifest.py 檔頭的動機說明。
        write_run_manifest(
            out_dir,
            config = asdict(self.cfg),
            extra  = {
                "window":         f"{start}_{end}",
                "fold_index":     fold_index,
                "train_years":    train_years,
                "test_year":      test_year,
                "features": {
                    "n_used":            len(feature_cols),
                    "exclude_cols":      sorted(self._exclude_cols),
                    "exclude_prefixes":  list(self._exclude_prefixes),
                    "used":              feature_cols,
                },
                "hyperparams": {
                    "tuned_this_fold": do_tune,
                    "n_trials":        self.cfg.n_trials if do_tune else 0,
                    "params":          best_params,
                },
            },
        )

        print(f"  Test IC={metrics['ic']:.4f}  RMSE={metrics['rmse']:.6f}  → {out_dir}")
        return metrics

    def _resolve_features(self, df: pd.DataFrame) -> list[str]:
        available = {
            c for c in set(df.columns) - self._exclude_cols
            if not c.startswith(self._exclude_prefixes)
        }
        if not self.cfg.use_features_csv:
            # ★ 2026-07-31：非數值欄會在這裡被丟掉，以前是**靜默**的。
            #   後果：排除清單算出來的欄數與實際進 LightGBM 的欄數對不上，而且完全
            #   沒有線索。實際踩過——`市場別`（object dtype: sii/otc/rotc）排除清單放行、
            #   這關擋掉，導致「表頭機械計算 79、實跑 78」查了一輪才知道差在哪
            #   （見 summary_research.md §2）。改為印出來，不改行為。
            numeric = [
                c for c in df.columns
                if c in available and pd.api.types.is_numeric_dtype(df[c])
            ]
            dropped = [c for c in df.columns if c in available and c not in set(numeric)]
            if dropped:
                print(f"  ⚠ 有 {len(dropped)} 個候選特徵因非數值 dtype 被排除："
                      f"{dropped}（如需納入請先轉成數值編碼）")
            return numeric
        feature_cols = [f for f in self.feature_list if f in available]
        missing      = [f for f in self.feature_list if f not in available]
        if missing:
            print(f"  ⚠ 候選特徵中有 {len(missing)} 個不存在於 df，已略過")
        return feature_cols

    def _compute_vol_weights(self, train_df: pd.DataFrame) -> np.ndarray | None:
        if not self.cfg.use_vol_weight:
            return None
        col = self.cfg.vol_weight_col
        if col not in train_df.columns:
            print(f"  ⚠ vol_weight_col='{col}' 不存在，改用等權訓練")
            return None
        vol   = train_df[col].fillna(train_df[col].median()).values.astype(float)
        n     = len(vol)
        ranks = pd.Series(vol).rank(method="average", ascending=False).values
        w     = ranks / n
        w     = w / w.mean()
        print(
            f"  Vol weight (rank-based) [{col}]: "
            f"min={w.min():.3f}  max={w.max():.3f}  std={w.std():.3f}  n={n:,}"
        )
        return w

    def _compute_return_weights(self, train_df: pd.DataFrame) -> np.ndarray | None:
        """
        報酬（target_col）越高，訓練權重越大。rank-based（而非 raw 數值）：
        避免單筆極端報酬把 RMSE 已有的離群值敏感度再放大一次，也不會出現
        「只認正報酬」的斷點——負報酬只是權重較低，仍有梯度貢獻。
        """
        if not self.cfg.use_return_weight:
            return None
        col = self.cfg.target_col
        if col not in train_df.columns:
            print(f"  ⚠ target_col='{col}' 不存在，改用等權訓練")
            return None
        r     = train_df[col].values.astype(float)
        n     = len(r)
        ranks = pd.Series(r).rank(method="average", ascending=True).values
        w     = ranks / n
        w     = w / w.mean()
        print(
            f"  Return weight (rank-based) [{col}]: "
            f"min={w.min():.3f}  max={w.max():.3f}  std={w.std():.3f}  n={n:,}"
        )
        return w

    def _compute_decay_weights(self, train_df: pd.DataFrame) -> np.ndarray | None:
        """
        指數時間衰減：越舊的樣本權重越小，半衰期 decay_half_life_months 個月。
        以訓練集最後一個交易日為 age=0；正規化到 mean=1（所以基準日的選擇
        只影響常數倍率，不影響相對權重）。
        """
        h = self.cfg.decay_half_life_months
        if h is None:
            return None
        if h <= 0:
            raise ValueError(f"decay_half_life_months 必須 > 0，收到 {h}")

        dates    = pd.to_datetime(train_df["年月日"])
        age_days = (dates.max() - dates).dt.days.values.astype(float)
        w = 0.5 ** (age_days / (h * 30.44))
        w = w / w.mean()
        print(
            f"  Decay weight (half-life {h}m): "
            f"min={w.min():.3f}  max={w.max():.3f}  std={w.std():.3f}  "
            f"span={age_days.max():.0f} 天"
        )
        return w

    def _compute_sample_weights(self, train_df: pd.DataFrame) -> np.ndarray | None:
        """合併 vol / return / time-decay 權重（各自可獨立開關），相乘後正規化。"""
        parts = {
            "vol":    self._compute_vol_weights(train_df),
            "return": self._compute_return_weights(train_df),
            "decay":  self._compute_decay_weights(train_df),
        }
        active = {k: v for k, v in parts.items() if v is not None}

        if not active:
            return None
        if len(active) == 1:
            return next(iter(active.values()))

        w = np.ones(len(train_df), dtype=float)
        for v in active.values():
            w = w * v
        w = w / w.mean()
        print(
            f"  Combined weight ({' × '.join(active)}): "
            f"min={w.min():.3f}  max={w.max():.3f}  std={w.std():.3f}"
        )
        return w

    def _evaluate(
        self,
        model:        lgb.Booster,
        X_test:       pd.DataFrame,
        y_test:       pd.Series,
        feature_cols: list[str],
        test_df:      pd.DataFrame,
    ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        y_pred = model.predict(X_test)

        ic, _ = spearmanr(y_test, y_pred)

        # ★ 2026-07-27：新增逐日 IC。
        #   `ic` 是把測試年所有日期混在一起的 pooled Spearman，同時吃到日間變異
        #   （哪一天大盤好）與日內變異（當天哪檔股票好），但 backtest 只用得到後者。
        #   實測 pooled 約為逐日平均的 3.6 倍（2024_2026 window：0.162 vs 0.045），
        #   會系統性高估模型品質。`ic` 保留原定義供既有 11 個 experiment 資料夾
        #   對照，新增的 `ic_daily` / `ic_daily_t` 才是與選股口徑一致的數字，
        #   評估模型請看後者。
        ic_daily, ic_daily_t, n_days = _daily_ic(
            y_test.values, y_pred, test_df["年月日"].values
        )

        metrics = {
            "rmse":        round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 6),
            "mae":         round(float(mean_absolute_error(y_test, y_pred)),         6),
            "r2":          round(float(r2_score(y_test, y_pred)),                   6),
            "ic":          round(float(ic),                                        6),
            "ic_daily":    round(float(ic_daily),                                  6),
            "ic_daily_t":  round(float(ic_daily_t),                                4),
            "ic_n_days":   int(n_days),
            "n_test":      int(len(y_test)),
        }

        pred_df = test_df[["證券代碼", "年月日", "return"]].copy().reset_index(drop=True)
        pred_df["y_true"] = y_test.values
        pred_df["y_pred"] = y_pred

        imp_df = build_imp_df(model, feature_cols)
        return metrics, pred_df, imp_df

    def _save_outputs(
        self,
        out_dir: Path,
        metrics: dict,
        params:  dict,
        pred_df: pd.DataFrame,
        imp_df:  pd.DataFrame,
    ):
        with open(out_dir / "metrics.json",     "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(params,  f, ensure_ascii=False, indent=2)
        pred_df.to_csv(out_dir / "predictions.csv",        index=False, encoding="utf-8-sig")
        imp_df.to_csv(out_dir  / "feature_importance.csv", index=False, encoding="utf-8-sig")

    def _save_summary(self, all_metrics: list[dict]):
        if not all_metrics:
            return
        summary_df = pd.DataFrame(all_metrics)
        out_path   = self.cfg.output_dir / "walk_forward_summary.csv"
        summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"\n{'='*60}")
        print(f"  Walk-Forward 完成  →  {out_path}")
        print(f"{'='*60}")
        cols = ["window", "test_year", "n_test", "ic", "ic_daily", "ic_daily_t",
                "rmse", "mae", "r2", "oof_ic", "hyperparams_tuned"]
        show = [c for c in cols if c in summary_df.columns]
        print(summary_df[show].to_string(index=False))