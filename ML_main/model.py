"""
model.py
========
Walk-forward LightGBM 訓練模組。

Walk-forward 規則：
    Windows: (2014,2016), (2015,2017), ..., (2022,2024)
    每個 window：
        - train = 前兩年（start_year, start_year+1）
        - test  = 第三年（start_year+2）

前處理策略：
    每個 window 獨立呼叫 DataPreprocessor，只讀取 [start, end] 年份資料。
    確保 beta / vol20 等 rolling 特徵不含未來資訊（no look-ahead bias）。
    若外部已計算好某 window 的 df（如 EDA 步驟），可透過 inject_cache() 注入，
    避免重複前處理。

目標：excess_return_tick

特徵來源：
    1. features_only.csv 中存在於 df 的欄位
    2. CAPM_Beta六月 → 對應 preprocess 產生的 beta
    3. 新增無 data leakage 特徵：vol20, daily_return, market_return

輸出（每個 window 存至 database/experiment/YYYY_YYYY/）：
    predictions.csv        ── 證券代碼, 年月日, return, y_true, y_prob, y_pred
    metrics.json           ── 評估指標
    feature_importance.csv ── 特徵重要性（gain）
    best_params.json       ── Optuna 最佳超參數

使用方式：
    from model import WalkForwardConfig, WalkForwardTrainer
    trainer = WalkForwardTrainer(WalkForwardConfig())
    trainer.run()

作者：Daniel Huang
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from preprocess import PreprocessConfig, DataPreprocessor

# ============================================================
#  專案根目錄：2026_research/（ML_main/ 的上一層）
# ============================================================
_ROOT = Path(__file__).resolve().parent.parent

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ============================================================
#  特徵清單
# ============================================================

# features_only.csv 的原始欄位名稱
_FEATURES_CSV_PATH = _ROOT / "database/features_only.csv"

# 新增：preprocess 計算的 backward-looking 特徵，無 data leakage
# beta 與 TEJ 的 CAPM_Beta六月 是不同資料，各自獨立保留
_NEW_FEATURES = ["beta", "vol20", "daily_return", "market_return"]

# 絕對排除（labels + IDs）── 即使出現在 features_only.csv 也不使用
_EXCLUDE_COLS = {
    "證券代碼", "年月日",
    "return", "return_tick", "return_tick_0",
    "excess_return", "excess_return_tick",   # forward-looking，為 label，嚴禁使用
}


def load_feature_list() -> list[str]:
    """
    讀取 features_only.csv，套用 alias（CAPM_Beta六月 → beta），
    附加新特徵，回傳最終候選特徵清單。
    重複欄位自動去除。
    """
    if not _FEATURES_CSV_PATH.exists():
        raise FileNotFoundError(f"找不到特徵清單: {_FEATURES_CSV_PATH}")

    raw = pd.read_csv(_FEATURES_CSV_PATH)["feature"].tolist()

    # 附加新特徵（beta 與 CAPM_Beta六月 各自獨立，不做 alias）
    mapped = raw + _NEW_FEATURES

    # 去重並排除 label 欄位
    seen, final = set(), []
    for f in mapped:
        if f not in seen and f not in _EXCLUDE_COLS:
            seen.add(f)
            final.append(f)

    return final


# ============================================================
#  CONFIG
# ============================================================

@dataclass
class WalkForwardConfig:
    """Walk-forward 訓練設定，所有超參數集中於此。"""

    # ── Walk-forward 視窗 ────────────────────────────────────
    window_start: int = 2014
    window_end:   int = 2025

    # ── 目標欄位 ─────────────────────────────────────────────
    target_col: str = "excess_return_tick"

    # ── 特徵設定 ─────────────────────────────────────────────
    use_features_csv: bool = True   # True = 只用 features_only.csv（+ 新特徵）
                                     # False = 使用 df 所有數值欄位（排除 labels）

    # ── Optuna / CV ──────────────────────────────────────────
    n_trials:     int = 15
    n_splits:     int = 3
    random_state: int = 42

    # ── Inverse-Volatility Sample Weighting ──────────────────
    # 以 1/vol 對訓練樣本加權，抑制模型偏好高波動股票
    # vol_weight_col    : 用來計算 weight 的波動率欄位（需存在於 df）
    # vol_weight_clip   : 最低波動率下界（避免極端大 weight），設為訓練集該欄位的此分位數
    use_vol_weight:    bool  = True
    vol_weight_col:    str   = "vol20"
    vol_weight_clip:   float = 0.05   # 5th percentile clip，防止極端 weight

    # ── 輸出路徑 ─────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: _ROOT / "database/experiment")

    # ── Preprocess 共用設定 ──────────────────────────────────
    raw_data_dir:     Path  = field(default_factory=lambda: _ROOT / "database/raw_data")
    market_data_path: Path  = field(default_factory=lambda: _ROOT / "database/market/market.csv")
    return_clip:      float = 0.15
    tick_threshold:   float = 0.01
    tick0_threshold:  float = 0.00
    beta_window:      int   = 60

    def __post_init__(self):
        self.output_dir       = Path(self.output_dir)
        self.raw_data_dir     = Path(self.raw_data_dir)
        self.market_data_path = Path(self.market_data_path)

    def windows(self) -> list[tuple[int, int]]:
        return [(s, s + 2) for s in range(self.window_start, self.window_end - 1)]


# ============================================================
#  TRAINER
# ============================================================

class WalkForwardTrainer:

    def __init__(self, config: WalkForwardConfig):
        self.cfg = config
        if config.use_features_csv:
            self.feature_list = load_feature_list()
            print(f"  特徵模式: features_only.csv  ({len(self.feature_list)} 個候選)")
        else:
            self.feature_list = None
            print("  特徵模式: 全部數值欄位（排除 labels）")

        # ── df cache：避免重複前處理（key = (start, end)）────
        self._df_cache: dict[tuple[int, int], pd.DataFrame] = {}

    # ────────────────────────────────────────────────────────
    #  Cache 注入（供外部將已計算好的 df 傳入）
    # ────────────────────────────────────────────────────────

    def inject_cache(self, start: int, end: int, df: pd.DataFrame):
        """
        將外部已計算好的 df 注入 cache，避免重複前處理。
        典型用途：EDA 步驟與第一個 ML window 年份相同時，
        直接傳入 EDA 產生的 df，跳過重複的 DataPreprocessor.run()。

        Parameters
        ----------
        start : int
            window 起始年份
        end : int
            window 結束年份（= start + 2）
        df : pd.DataFrame
            已完成前處理、含 target_col 的 DataFrame
        """
        key = (start, end)
        self._df_cache[key] = df.dropna(subset=[self.cfg.target_col])
        print(f"  ✓ 注入 df cache：window ({start}, {end})，"
              f"形狀 {self._df_cache[key].shape}")

    # ────────────────────────────────────────────────────────
    #  公開介面
    # ────────────────────────────────────────────────────────

    def run(self):
        windows = self.cfg.windows()
        print(f"\n{'='*60}")
        print(f"  Walk-Forward Training")
        print(f"  Windows : {windows[0]} → {windows[-1]}  ({len(windows)} 個)")
        print(f"  Target  : {self.cfg.target_col}")
        print(f"  Output  : {self.cfg.output_dir}")
        print(f"{'='*60}")

        all_metrics = []
        for i, (start, end) in enumerate(windows, 1):
            print(f"\n[{i}/{len(windows)}] Window {start}–{end}")
            try:
                metrics = self._run_window(start, end)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"  ✗ 失敗: {e}")
                import traceback; traceback.print_exc()

        self._save_summary(all_metrics)

    # ────────────────────────────────────────────────────────
    #  Window 獨立前處理（每次只載入該 window 的年份）
    # ────────────────────────────────────────────────────────

    def _preprocess_window(self, start: int, end: int) -> pd.DataFrame:
        """
        只讀取 [start, end] 年的資料，確保 beta / vol20 等 rolling 特徵
        不會用到 end 年之後的未來資訊。

        優先查 cache：若 inject_cache() 已預先注入相同 (start, end)，
        直接回傳快取結果，跳過重複的 DataPreprocessor.run()。
        """
        key = (start, end)
        if key in self._df_cache:
            print(f"  ✓ 命中 df cache（{start}–{end}），跳過重複前處理")
            return self._df_cache[key]

        cfg = PreprocessConfig(
            start_year       = start,
            end_year         = end,
            raw_data_dir     = self.cfg.raw_data_dir,
            market_data_path = self.cfg.market_data_path,
            return_clip      = self.cfg.return_clip,
            tick_threshold   = self.cfg.tick_threshold,
            tick0_threshold  = self.cfg.tick0_threshold,
            beta_window      = self.cfg.beta_window,
        )
        df = DataPreprocessor(cfg).run()
        return df.dropna(subset=[self.cfg.target_col])

    # ────────────────────────────────────────────────────────
    #  單一 Window 流程
    # ────────────────────────────────────────────────────────

    def _run_window(self, start: int, end: int) -> dict:
        test_year   = end
        train_years = [start, start + 1]

        # ── Window 獨立前處理 ────────────────────────────────
        print(f"\n► 前處理 Window {start}–{end}（只含此 window 年份資料）...")
        full_df = self._preprocess_window(start, end)
        print(f"  Window 資料形狀: {full_df.shape}")

        # ── 切片 ────────────────────────────────────────────
        year_col   = full_df["年月日"].dt.year
        train_df   = full_df[year_col.isin(train_years)].copy()
        test_df    = full_df[year_col == test_year].copy()

        print(f"  Train {train_years}: {len(train_df):,} 筆  "
              f"正例率 {train_df[self.cfg.target_col].mean():.3f}")
        print(f"  Test  {test_year} : {len(test_df):,} 筆  "
              f"正例率 {test_df[self.cfg.target_col].mean():.3f}")

        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Train 或 Test 為空")

        # ── 特徵（候選清單 ∩ df 實際欄位）──────────────────
        feature_cols = self._resolve_features(full_df)
        print(f"  實際使用特徵: {len(feature_cols)} 個")

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[self.cfg.target_col]
        X_test  = test_df[feature_cols].fillna(0)
        y_test  = test_df[self.cfg.target_col]

        # ── Inverse-Volatility Sample Weight ────────────────
        train_weight = self._compute_sample_weights(train_df)

        # ── Optuna + OOF prob（同一次 CV，不重複訓練）──────
        best_params, oof_prob = self._tune(X_train, y_train, train_weight)

        # ── 最終模型 ─────────────────────────────────────────
        print(f"  訓練最終模型...")
        model = self._train_final(X_train, y_train, best_params, train_weight)

        # ── Threshold（直接用 OOF prob，不重新訓練）────────
        best_threshold = self._find_threshold(y_train, oof_prob)

        # ── 評估 ────────────────────────────────────────────
        metrics, pred_df, imp_df = self._evaluate(
            model, X_test, y_test, feature_cols, test_df, best_threshold
        )
        metrics.update({
            "window":          f"{start}_{end}",
            "train_years":     str(train_years),
            "test_year":       test_year,
            "threshold":       best_threshold,
            "vol_weight_used": train_weight is not None,
            "vol_weight_col":  self.cfg.vol_weight_col if train_weight is not None else None,
        })

        # ── 輸出 ────────────────────────────────────────────
        out_dir = self.cfg.output_dir / f"{start}_{end}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save_outputs(out_dir, metrics, best_params, pred_df, imp_df)

        print(f"  AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}  → {out_dir}")
        return metrics

    # ────────────────────────────────────────────────────────
    #  特徵解析
    # ────────────────────────────────────────────────────────

    def _resolve_features(self, df: pd.DataFrame) -> list[str]:
        """
        use_features_csv=True  : 候選清單 ∩ df 實際欄位，排除 labels
        use_features_csv=False : df 所有數值欄位，排除 labels
        印出缺失欄位供除錯。
        """
        available = set(df.columns) - _EXCLUDE_COLS

        if not self.cfg.use_features_csv:
            # 全部數值欄位
            return [
                c for c in df.columns
                if c in available and pd.api.types.is_numeric_dtype(df[c])
            ]

        # features_only.csv 模式
        feature_cols = [f for f in self.feature_list if f in available]
        missing = [f for f in self.feature_list if f not in available]
        if missing:
            print(f"  ⚠ 候選特徵中有 {len(missing)} 個不存在於 df，已略過:")
            for m in missing:
                print(f"      - {m}")

        return feature_cols

    # ────────────────────────────────────────────────────────
    #  Inverse-Volatility Sample Weighting
    # ────────────────────────────────────────────────────────

    def _compute_sample_weights(self, train_df: pd.DataFrame) -> np.ndarray | None:
        """
        Rank-based inverse-volatility sample weight。

        原理：對 vol 做降序排名，rank(-vol) / n
            - 低 vol 股票 → 高 rank → 高 weight
            - 高 vol 股票 → 低 rank → 低 weight
            - weight 線性分佈，完全不受 vol outlier 影響

        正規化：mean weight = 1，保持 LightGBM gradient scale 不變。
        """
        if not self.cfg.use_vol_weight:
            return None

        col = self.cfg.vol_weight_col
        if col not in train_df.columns:
            print(f"  ⚠ vol_weight_col='{col}' 不存在，改用等權訓練")
            return None

        vol = train_df[col].fillna(train_df[col].median()).values.astype(float)
        n = len(vol)

        # rank(-vol)：vol 最小 → rank=1（最高 weight）
        # ascending=False 代表 vol 最大 → rank=1，所以用 ascending=True 再反轉
        ranks = pd.Series(vol).rank(method="average", ascending=False).values  # vol 最大 → rank=1
        w = ranks / n   # vol 最小 → rank≈n → w≈1；vol 最大 → rank≈1 → w≈1/n

        w = w / w.mean()   # 正規化，mean weight = 1

        print(
            f"  Vol weight (rank-based) [{col}]: "
            f"min={w.min():.3f}  max={w.max():.3f}  "
            f"std={w.std():.3f}  n={n:,}"
        )
        return w

    # ────────────────────────────────────────────────────────
    #  Optuna 調參 + OOF prob（合併，不重複訓練）
    # ────────────────────────────────────────────────────────

    def _tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[dict, np.ndarray]:
        """
        回傳 (best_params, oof_prob)。
        最佳參數確定後，只再跑一輪 5-fold OOF 收集 prob，
        供 _find_threshold 使用，不額外增加訓練負擔。

        sample_weight: shape=(n_train,)，由 _compute_sample_weights 產生。
            傳入後，每個 fold 的 train/valid Dataset 都會附上對應的 weight slice。
        """
        tscv = TimeSeriesSplit(n_splits=self.cfg.n_splits)

        # ── Optuna ──────────────────────────────────────────
        def objective(trial):
            params = {
                "objective":        "binary",
                "metric":           "binary_logloss",
                "verbosity":        -1,
                "boosting_type":    "gbdt",
                "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "num_leaves":       trial.suggest_int("num_leaves", 16, 512),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq":     trial.suggest_int("bagging_freq", 1, 20),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
                "max_depth":        trial.suggest_int("max_depth", 5, 20),
            }
            fold_losses = []
            for tr_idx, va_idx in tscv.split(X_train):
                w_tr = sample_weight[tr_idx] if sample_weight is not None else None
                w_va = sample_weight[va_idx] if sample_weight is not None else None
                m = lgb.train(
                    params,
                    lgb.Dataset(X_train.iloc[tr_idx], label=y_train.iloc[tr_idx], weight=w_tr),
                    num_boost_round=1000,
                    valid_sets=[lgb.Dataset(X_train.iloc[va_idx], label=y_train.iloc[va_idx], weight=w_va)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                prob = np.clip(
                    m.predict(X_train.iloc[va_idx], num_iteration=m.best_iteration),
                    1e-7, 1 - 1e-7
                )
                fold_losses.append(log_loss(y_train.iloc[va_idx], prob))
            return np.mean(fold_losses)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.cfg.random_state),
        )
        study.optimize(objective, n_trials=self.cfg.n_trials, show_progress_bar=False)
        best_params = study.best_params
        print(f"  Best CV log-loss: {study.best_value:.6f}")

        # ── 用最佳參數跑一輪 OOF，收集 prob ────────────────
        params_final = {
            "objective":     "binary",
            "metric":        "binary_logloss",
            "verbosity":     -1,
            "boosting_type": "gbdt",
            **best_params,
        }
        oof_prob = np.zeros(len(y_train))
        for tr_idx, va_idx in tscv.split(X_train):
            w_tr = sample_weight[tr_idx] if sample_weight is not None else None
            m = lgb.train(
                params_final,
                lgb.Dataset(X_train.iloc[tr_idx], label=y_train.iloc[tr_idx], weight=w_tr),
                num_boost_round=500,
            )
            oof_prob[va_idx] = m.predict(X_train.iloc[va_idx])

        return best_params, oof_prob

    # ────────────────────────────────────────────────────────
    #  最終模型
    # ────────────────────────────────────────────────────────

    def _train_final(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        best_params: dict,
        sample_weight: np.ndarray | None = None,
    ) -> lgb.Booster:
        params = {
            "objective":     "binary",
            "metric":        "binary_logloss",
            "verbosity":     -1,
            "boosting_type": "gbdt",
            **best_params,
        }
        return lgb.train(
            params,
            lgb.Dataset(X_train, label=y_train, weight=sample_weight),
            num_boost_round=500,
        )

    # ────────────────────────────────────────────────────────
    #  Threshold 搜尋（直接用 OOF prob，零額外訓練）
    # ────────────────────────────────────────────────────────

    def _find_threshold(self, y_train: pd.Series, oof_prob: np.ndarray) -> float:
        best_thresh, best_f1 = 0.5, 0.0
        for thresh in np.arange(0.05, 0.70, 0.01):
            f1 = f1_score(y_train, (oof_prob >= thresh).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        print(f"  Best threshold: {best_thresh:.2f}  (OOF F1={best_f1:.4f})")
        return round(best_thresh, 2)

    # ────────────────────────────────────────────────────────
    #  評估
    # ────────────────────────────────────────────────────────

    def _evaluate(
        self,
        model:        lgb.Booster,
        X_test:       pd.DataFrame,
        y_test:       pd.Series,
        feature_cols: list[str],
        test_df:      pd.DataFrame,
        threshold:    float,
    ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:

        y_prob = np.clip(model.predict(X_test), 1e-7, 1 - 1e-7)
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "log_loss":           round(float(log_loss(y_test, y_prob)),                         6),
            "roc_auc":            round(float(roc_auc_score(y_test, y_prob)),                    6),
            "accuracy":           round(float(accuracy_score(y_test, y_pred)),                   6),
            "precision":          round(float(precision_score(y_test, y_pred, zero_division=0)), 6),
            "recall":             round(float(recall_score(y_test, y_pred, zero_division=0)),    6),
            "f1":                 round(float(f1_score(y_test, y_pred, zero_division=0)),        6),
            "n_test":             int(len(y_test)),
            "positive_rate_true": round(float(y_test.mean()), 4),
            "positive_rate_pred": round(float(y_pred.mean()), 4),
        }

        pred_df = test_df[["證券代碼", "年月日", "return"]].copy().reset_index(drop=True)
        pred_df["y_true"] = y_test.values
        pred_df["y_prob"] = y_prob
        pred_df["y_pred"] = y_pred

        imp_df = pd.DataFrame({
            "feature":    feature_cols,
            "importance": model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        imp_df["rank"] = range(1, len(imp_df) + 1)

        return metrics, pred_df, imp_df

    # ────────────────────────────────────────────────────────
    #  儲存輸出
    # ────────────────────────────────────────────────────────

    def _save_outputs(
        self,
        out_dir: Path,
        metrics: dict,
        params:  dict,
        pred_df: pd.DataFrame,
        imp_df:  pd.DataFrame,
    ):
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        pred_df.to_csv(out_dir / "predictions.csv",        index=False, encoding="utf-8-sig")
        imp_df.to_csv(out_dir  / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # ────────────────────────────────────────────────────────
    #  Summary
    # ────────────────────────────────────────────────────────

    def _save_summary(self, all_metrics: list[dict]):
        if not all_metrics:
            return

        summary_df = pd.DataFrame(all_metrics)
        out_path   = self.cfg.output_dir / "walk_forward_summary.csv"
        summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"\n{'='*60}")
        print(f"  Walk-Forward 完成  →  {out_path}")
        print(f"{'='*60}")
        cols = ["window", "test_year", "n_test", "roc_auc", "f1",
                "threshold", "positive_rate_true", "positive_rate_pred"]
        show = [c for c in cols if c in summary_df.columns]
        print(summary_df[show].to_string(index=False))