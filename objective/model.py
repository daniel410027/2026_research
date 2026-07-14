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

Walk-forward 規則：
    Windows: (2014,2016), (2015,2017), ..., (2022,2024)
    每個 window：
        train = 前兩年（start_year, start_year+1）
        test  = 第三年（start_year+2）

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
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

from .lgb_utils import (
    tune_lgb, train_final_lgb, evaluate_oof_ic, build_imp_df, collect_oof_prob,
)

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

_EXCLUDE_COLS = {
    "證券代碼", "年月日", "market_index",
    "return", "return_tick", "return_tick_0",
    "excess_return", "excess_return_tick",
    # [2026-07-14] market_return：backward 大盤日報酬（T-1→T 收盤），時序上無
    #   look-ahead，但為 per-day constant——同日所有股票值相同，等同「日期指紋」。
    #   Pooled panel 訓練下樹模型可據此辨識特定日期、記憶該日平均 excess return
    #   （兩年 train 僅 ~490 個唯一交易日，date-level conditioning 極易過擬合）；
    #   對每日 cross-sectional decile 排序無直接資訊。有用的交互版本
    #   （超額報酬日大盤/週大盤、beta）已在特徵集，原始值屬冗餘。
    #   欄位仍保留於 database_make/ CSV（make 階段的 beta/超額報酬計算依賴它）。
    "market_return",
    # [2026-06 對齊 live] 以下兩欄納入 base，使 WF 與 daily_model.py 共用同一訓練特徵集。
    #   排除邏輯自此集中於本模組，不再依賴各進入點（main.py / main_fix.py）以
    #   EXTRA_EXCLUDE_COLS 擴充；其既有 union 變為 idempotent 冗餘，可保留不動。
    #   注意：目前 database_make/ 並無這兩欄，故此改動不會改變現有特徵集，
    #         純為防止未來 schema 新增時的靜默 train/serve 漂移。
    "amount",            # 成交金額：僅作流動性過濾，不進訓練
    "market_return_fwd", # 前視 T+1：label 建構用，防呆排除
    # [2026-07-08] 市場別：make_new.py 自 market_type.csv merge 進來的靜態字串欄
    #   （sii/otc/rotc），供 live 端 universe 過濾與檢視用。目前因 object dtype
    #   會被 _resolve_features 的 is_numeric_dtype 靜默濾掉，但該防線依賴 dtype
    #   巧合——若未來被 encode 成數值即無聲進入特徵集。顯式排除以固定意圖。
    #   （panel 已經 sii 篩選，此欄近乎常數，本無訓練價值。）
    "市場別",
}


def load_feature_list() -> list[str]:
    """讀取 features_only.csv，附加新特徵，回傳最終候選清單。"""
    if not _FEATURES_CSV_PATH.exists():
        raise FileNotFoundError(f"找不到特徵清單: {_FEATURES_CSV_PATH}")

    raw    = pd.read_csv(_FEATURES_CSV_PATH)["feature"].tolist()
    mapped = raw + _NEW_FEATURES

    seen, final = set(), []
    for f in mapped:
        if f not in seen and f not in _EXCLUDE_COLS:
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

    # ── 目標欄位 ──────────────────────────────────────────────
    target_col: str = "excess_return"

    # ── 特徵設定 ──────────────────────────────────────────────
    use_features_csv: bool = False

    # ── Optuna / CV ───────────────────────────────────────────
    n_trials:     int = 15
    n_splits:     int = 3
    random_state: int = 42

    # ── 超參數凍結策略 ────────────────────────────────────────
    # freeze_hyperparams=True：第一個 fold 完整 tune，後續 fold 沿用
    # retune_every_n：每 N 個 fold 重新 tune 一次（999 = 實質上只 tune 一次）
    freeze_hyperparams: bool = True
    retune_every_n:     int  = 999

    # ── Inverse-Volatility Sample Weighting ───────────────────
    use_vol_weight:  bool  = True
    vol_weight_col:  str   = "vol20"
    vol_weight_clip: float = 0.05

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

    def __post_init__(self):
        self.output_dir      = Path(self.output_dir)
        self.db_dir          = Path(self.db_dir)
        self.precomputed_dir = Path(self.precomputed_dir)

    def windows(self) -> list[tuple[int, int]]:
        return [(s, s + 2) for s in range(self.window_start, self.window_end - 1)]


# ─────────────────────────────────────────────────────────────
#  TRAINER
# ─────────────────────────────────────────────────────────────

class WalkForwardTrainer:

    def __init__(self, config: WalkForwardConfig):
        self.cfg = config
        if config.use_features_csv:
            self.feature_list = load_feature_list()
            print(f"  特徵模式: features_only.csv  ({len(self.feature_list)} 個候選)")
        else:
            self.feature_list = None
            print("  特徵模式: 全部數值欄位（排除 labels）")

        self._df_cache: dict[tuple[int, int], pd.DataFrame] = {}

        # 超參數凍結快取
        self._frozen_params: dict | None = None
        self._fold_counter:  int         = 0   # 計算已執行 fold 數（用於 retune_every_n）

    def inject_cache(self, start: int, end: int, df: pd.DataFrame):
        """外部注入已處理的 df，跳過重複前處理。"""
        key = (start, end)
        self._df_cache[key] = df.dropna(subset=[self.cfg.target_col])
        print(f"  ✓ 注入 df cache：window ({start}, {end})，形狀 {self._df_cache[key].shape}")

    def run(self):
        windows = self.cfg.windows()
        print(f"\n{'='*60}")
        print(f"  Walk-Forward Training")
        print(f"  Windows : {windows[0]} → {windows[-1]}  ({len(windows)} 個)")
        print(f"  Target  : {self.cfg.target_col}")
        print(f"  Output  : {self.cfg.output_dir}")
        if self.cfg.freeze_hyperparams:
            print(f"  超參數策略 : 第一個 fold tune，後續凍結沿用"
                  f"（retune_every_n={self.cfg.retune_every_n}）")
        else:
            print(f"  超參數策略 : 每個 fold 獨立 tune")
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

    def _preprocess_window(self, start: int, end: int) -> pd.DataFrame:
        key = (start, end)
        if key in self._df_cache:
            print(f"  ✓ 命中 df cache（{start}–{end}），跳過重複前處理")
            return self._df_cache[key]

        # 只讀 train_years + test_year（滾動特徵已跨年連續，無需重複讀 start 年）
        # 經 test_year_boundary.py 驗證：vol20 / beta / CAPM_Beta 系列 jump_ratio < 2x
        test_year   = end
        train_years = [start, start + 1]
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
        """判斷本 fold 是否需要執行 Optuna tuning。"""
        if not self.cfg.freeze_hyperparams:
            return True                                    # 從不凍結
        if self._frozen_params is None:
            return True                                    # 尚無凍結參數（第一個 fold）
        if self.cfg.retune_every_n < 999:
            return (fold_index % self.cfg.retune_every_n) == 1   # 週期性 re-tune
        return False                                       # 完全凍結

    def _run_window(self, start: int, end: int, fold_index: int = 1) -> dict:
        test_year   = end
        train_years = [start, start + 1]

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

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[self.cfg.target_col]
        X_test  = test_df[feature_cols].fillna(0)
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
            if self.cfg.freeze_hyperparams:
                self._frozen_params = best_params
                print(f"  [Optuna] 超參數已凍結，後續 fold 沿用")
        else:
            best_params = self._frozen_params
            print(f"  [Optuna] 沿用凍結超參數，跳過 tune，只收集 OOF 預測值")
            oof_pred = collect_oof_prob(
                X_train, y_train,
                best_params   = best_params,
                n_splits      = self.cfg.n_splits,
                sample_weight = train_weight,
                dates         = train_dates,
            )

        print("  訓練最終模型...")
        model = train_final_lgb(X_train, y_train, best_params, train_weight)

        oof_ic = evaluate_oof_ic(y_train, oof_pred)

        metrics, pred_df, imp_df = self._evaluate(
            model, X_test, y_test, feature_cols, test_df
        )
        metrics.update({
            "window":             f"{start}_{end}",
            "train_years":        str(train_years),
            "test_year":          test_year,
            "oof_ic":             oof_ic,
            "vol_weight_used":    train_weight is not None,
            "vol_weight_col":     self.cfg.vol_weight_col if train_weight is not None else None,
            "hyperparams_tuned":  do_tune,   # 本 fold 是否重新 tune
        })

        out_dir = self.cfg.output_dir / f"{start}_{end}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save_outputs(out_dir, metrics, best_params, pred_df, imp_df)

        print(f"  Test IC={metrics['ic']:.4f}  RMSE={metrics['rmse']:.6f}  → {out_dir}")
        return metrics

    def _resolve_features(self, df: pd.DataFrame) -> list[str]:
        available = set(df.columns) - _EXCLUDE_COLS
        if not self.cfg.use_features_csv:
            return [
                c for c in df.columns
                if c in available and pd.api.types.is_numeric_dtype(df[c])
            ]
        feature_cols = [f for f in self.feature_list if f in available]
        missing      = [f for f in self.feature_list if f not in available]
        if missing:
            print(f"  ⚠ 候選特徵中有 {len(missing)} 個不存在於 df，已略過")
        return feature_cols

    def _compute_sample_weights(self, train_df: pd.DataFrame) -> np.ndarray | None:
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

        metrics = {
            "rmse":   round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 6),
            "mae":    round(float(mean_absolute_error(y_test, y_pred)),         6),
            "r2":     round(float(r2_score(y_test, y_pred)),                   6),
            "ic":     round(float(ic),                                        6),
            "n_test": int(len(y_test)),
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
        cols = ["window", "test_year", "n_test", "ic", "rmse", "mae", "r2",
                "oof_ic", "hyperparams_tuned"]
        show = [c for c in cols if c in summary_df.columns]
        print(summary_df[show].to_string(index=False))