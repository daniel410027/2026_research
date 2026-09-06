"""
daily_main.py
=============
每日交易訊號預測 Pipeline。

流程：
    1. 掃描 database/daily_data/ 找最新 yyyymmdd.csv（預測目標日）
    2. 從 database/raw_data/ 撈最近 TRAIN_YEARS 年的季度 CSV 作為訓練歷史
    3. 將 daily_data 所有歷史 CSV 附加至歷史資料末端，執行與 preprocess.py 相同特徵工程
       （rolling 指標：RSI, beta, vol20 等皆以完整歷史計算，無 look-ahead bias）
    4. 訓練集 = 歷史資料中 excess_return_tick 有效列（含 daily_data 歷史日）
       預測集 = predict_date 當日所有股票列（label 為 NaN，僅需特徵）
    5. Optuna 調超參數 → 訓練最終 LightGBM → 搜尋最佳 threshold（OOF F1）
    6. 對預測集輸出 pred_score，依分數排序，標記 is_selected

輸出（database/daily_predict/YYYYMMDD/）：
    YYYYMMDD_prediction.csv        ── 完整分數，含所有股票
    feature_importance.csv         ── 特徵重要性（gain），同 model.py 格式
    best_params.json               ── Optuna 最佳超參數
    YYYYMMDD_shap.csv              ── SHAP 值（每股 × 每特徵），若 compute_shap=True
    YYYYMMDD_shap_summary.csv      ── 每特徵平均 |SHAP|，若 compute_shap=True
    final_day_prediction_sort.csv  ── 覆寫更新，永遠是最新預測日（根目錄）

使用方式：
    python daily_main.py

作者：Daniel Huang
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import json

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from preprocess import DataPreprocessor, PreprocessConfig
from model import _EXCLUDE_COLS, load_feature_list

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ============================================================
#  CONFIG
# ============================================================

@dataclass
class DailyConfig:
    """每日預測設定，所有超參數集中於此。"""

    # ── 資料路徑 ────────────────────────────────────────────
    daily_data_dir:   Path = field(default_factory=lambda: Path("database/daily_data"))
    raw_data_dir:     Path = field(default_factory=lambda: Path("database/raw_data"))
    market_data_path: Path = field(default_factory=lambda: Path("database/market/market.csv"))
    output_dir:       Path = field(default_factory=lambda: Path("database/daily_predict"))

    # ── 訓練設定 ─────────────────────────────────────────────
    train_years:      int  = 2              # rolling 訓練視窗（年）
    target_col:       str  = "excess_return_tick"
    use_features_csv: bool = True           # True = features_only.csv；False = 全部數值欄位

    # ── 選股設定 ─────────────────────────────────────────────
    top_n: int = 20                         # is_selected=1 的股數

    # ── 前處理參數（與 RunConfig 保持一致）──────────────────
    return_clip:     float = 0.15
    tick_threshold:  float = 0.01
    tick0_threshold: float = 0.00
    beta_window:     int   = 60

    # ── Optuna / CV ──────────────────────────────────────────
    n_trials:     int = 15
    n_splits:     int = 3
    random_state: int = 42

    # ── SHAP ─────────────────────────────────────────────────
    compute_shap:  bool     = True   # True = 輸出 SHAP 值
    shap_target:   str      = "pred"  # "pred"=預測集（快）/ "train"=訓練集（慢）
    shap_top_n:    int      = 20      # summary 只顯示 Top N 特徵
    shap_sample_n: int|None = 5000    # None = 全量；int = 對 shap_target 隨機抽樣
                                      # 建議：pred 可設 None；train 建議 5000–20000

    def __post_init__(self):
        self.daily_data_dir   = Path(self.daily_data_dir)
        self.raw_data_dir     = Path(self.raw_data_dir)
        self.market_data_path = Path(self.market_data_path)
        self.output_dir       = Path(self.output_dir)


# ============================================================
#  DAILY PREPROCESSOR
# ============================================================

class DailyPreprocessor(DataPreprocessor):
    """
    繼承 DataPreprocessor，僅覆寫 _load_and_merge。

    差異：在讀完 raw_data 季度 CSV 後，額外附加 daily_data/ 目錄內所有
    yyyymmdd.csv，使 rolling 指標（RSI、beta、vol20 等）可正確計算到預測
    目標日，同時將當季所有歷史日納入訓練集。
    其餘特徵工程步驟完全沿用 DataPreprocessor。
    """

    def __init__(self, config: PreprocessConfig, daily_data_dir: Path):
        # 繞過 end_year - start_year >= 2 的 assertion，直接設定 config
        object.__setattr__(self, "config", config)
        self.config = config
        self.df: pd.DataFrame | None = None
        self._market_df: pd.DataFrame | None = None
        self._daily_data_dir = daily_data_dir  # 整個目錄

    def _load_and_merge(self) -> pd.DataFrame:
        """讀取歷史季度 CSV（raw_data）+ 所有 daily CSV，合併去重。"""
        dfs = []

        # ── 歷史季度資料 ─────────────────────────────────────
        for year in range(self.config.start_year, self.config.end_year + 1):
            for q in range(1, 5):
                path = self.config.raw_data_dir / f"{year}{q}.csv"
                if path.exists():
                    df = self._read_csv(path)
                    dfs.append(df)
                    print(f"  讀取: {path.name}  ({len(df):,} 筆)")

        if not dfs:
            raise FileNotFoundError(
                f"raw_data 無任何資料，請確認路徑: {self.config.raw_data_dir}"
            )

        # ── 附加所有 daily_data CSV（依日期排序）────────────
        daily_csvs = sorted(self._daily_data_dir.glob("????????.csv"))
        if not daily_csvs:
            raise FileNotFoundError(
                f"找不到任何 daily_data CSV，請確認路徑: {self._daily_data_dir}"
            )
        for p in daily_csvs:
            df = self._read_csv(p)
            dfs.append(df)
            print(f"  讀取 (daily): {p.name}  ({len(df):,} 筆)")

        # ── 合併去重（daily_data 優先，keep="last"）──────────
        merged = pd.concat(dfs, ignore_index=True)
        before = len(merged)
        merged = merged.drop_duplicates(subset=["年月日", "證券代碼"], keep="last")
        print(f"\n  合併完成: {before:,} → {len(merged):,} 筆，"
              f"{merged['證券代碼'].nunique()} 檔股票")
        return merged


# ============================================================
#  DAILY TRAINER
# ============================================================

class DailyTrainer:
    """
    每日 LightGBM 訓練 + 預測器。

    流程：
        preprocess → split（train / predict）→ tune（Optuna）
        → train_final → find_threshold → predict → [shap] → save
    """

    def __init__(self, cfg: DailyConfig):
        self.cfg = cfg
        if cfg.use_features_csv:
            self.feature_list = load_feature_list()
            print(f"  特徵模式: features_only.csv  ({len(self.feature_list)} 個候選)")
        else:
            self.feature_list = None
            print("  特徵模式: 全部數值欄位（排除 labels）")

    # ────────────────────────────────────────────────────────
    #  公開介面
    # ────────────────────────────────────────────────────────

    def run(self):
        t0 = time.time()
        cfg = self.cfg

        # ── 1. 找最新 daily_data ─────────────────────────────
        predict_date, daily_csv = self._find_latest_daily(cfg.daily_data_dir)
        print(f"\n  預測目標日: {predict_date}  ({daily_csv.name})")

        # ── 2. 決定訓練年份視窗 ──────────────────────────────
        end_year   = predict_date.year
        start_year = end_year - max(cfg.train_years, 2)
        print(f"  訓練視窗  : {start_year} ~ {end_year}（含當季 daily 資料）")

        # ── 3. 前處理 ────────────────────────────────────────
        print(f"\n{'─'*60}")
        print("  ▶ 前處理")
        print(f"{'─'*60}")
        df = self._preprocess(start_year, end_year)

        # ── 4. 分離訓練集 / 預測集 ───────────────────────────
        X_train, y_train, X_pred, pred_meta = self._split(df, predict_date)

        if len(X_train) == 0:
            print("  ✗ 訓練集為空，請確認歷史資料是否存在。")
            sys.exit(1)
        if len(X_pred) == 0:
            print(f"  ✗ 預測集為空（{predict_date} 無資料）。")
            sys.exit(1)

        print(f"\n  訓練集: {len(X_train):,} 筆  正例率 {y_train.mean():.3f}")
        print(f"  預測集: {len(X_pred):,} 股")
        print(f"  特徵數: {X_train.shape[1]} 個")

        # ── 5. Optuna 調參 + OOF threshold ──────────────────
        print(f"\n{'─'*60}")
        print("  ▶ Optuna 調參")
        print(f"{'─'*60}")
        best_params, oof_prob = self._tune(X_train, y_train)
        threshold = self._find_threshold(y_train, oof_prob)

        # ── 6. 訓練最終模型 ──────────────────────────────────
        print(f"\n  ▶ 訓練最終模型...")
        model = self._train_final(X_train, y_train, best_params)

        # ── 7. 預測 + 特徵重要性 ─────────────────────────────
        pred_score = np.clip(model.predict(X_pred), 1e-7, 1 - 1e-7)
        imp_df     = self._build_imp_df(model, self._feature_cols)

        # ── 8. SHAP（可選）──────────────────────────────────
        shap_df         = None
        shap_summary_df = None
        if cfg.compute_shap:
            print(f"\n{'─'*60}")
            print("  ▶ 計算 SHAP 值")
            print(f"{'─'*60}")
            shap_df, shap_summary_df = self._compute_shap(model, X_train, X_pred, pred_meta)

        # ── 9. 輸出 ──────────────────────────────────────────
        print(f"\n{'─'*60}")
        print("  ▶ 輸出結果")
        print(f"{'─'*60}")
        self._save(
            pred_meta, pred_score, threshold, predict_date,
            best_params, imp_df, shap_df, shap_summary_df,
        )

        print(f"\n  總耗時: {time.time() - t0:.1f} 秒")

    # ────────────────────────────────────────────────────────
    #  找最新 daily_data
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _find_latest_daily(daily_data_dir: Path) -> tuple[date, Path]:
        """掃描 database/daily_data/，取 yyyymmdd.csv 最新一筆。"""
        csvs = sorted(daily_data_dir.glob("????????.csv"), reverse=True)
        if not csvs:
            raise FileNotFoundError(
                f"找不到任何 daily_data CSV，請確認路徑: {daily_data_dir}"
            )
        latest = csvs[0]
        s = latest.stem
        d = date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        return d, latest

    # ────────────────────────────────────────────────────────
    #  前處理
    # ────────────────────────────────────────────────────────

    def _preprocess(self, start_year: int, end_year: int) -> pd.DataFrame:
        cfg = self.cfg
        preprocess_cfg = PreprocessConfig(
            start_year       = start_year,
            end_year         = end_year,
            raw_data_dir     = cfg.raw_data_dir,
            market_data_path = cfg.market_data_path,
            return_clip      = cfg.return_clip,
            tick_threshold   = cfg.tick_threshold,
            tick0_threshold  = cfg.tick0_threshold,
            beta_window      = cfg.beta_window,
        )
        preprocessor = DailyPreprocessor(preprocess_cfg, cfg.daily_data_dir)
        return preprocessor.run()

    # ────────────────────────────────────────────────────────
    #  分離訓練集 / 預測集
    # ────────────────────────────────────────────────────────

    def _split(
        self,
        df: pd.DataFrame,
        predict_date: date,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
        target = self.cfg.target_col
        predict_ts = pd.Timestamp(predict_date)

        feature_cols = self._resolve_features(df)
        self._feature_cols = feature_cols

        train_mask = (df["年月日"] < predict_ts) & df[target].notna()
        train_df   = df[train_mask]

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target].astype(int)

        pred_mask = df["年月日"] == predict_ts
        pred_df   = df[pred_mask]

        X_pred    = pred_df[feature_cols].fillna(0)
        pred_meta = pred_df[["證券代碼", "年月日"]].copy().reset_index(drop=True)

        return X_train, y_train, X_pred.reset_index(drop=True), pred_meta

    def _resolve_features(self, df: pd.DataFrame) -> list[str]:
        available = set(df.columns) - _EXCLUDE_COLS

        if not self.cfg.use_features_csv or self.feature_list is None:
            return [
                c for c in df.columns
                if c in available and pd.api.types.is_numeric_dtype(df[c])
            ]

        feature_cols = [f for f in self.feature_list if f in available]
        missing = [f for f in self.feature_list if f not in available]
        if missing:
            print(f"  ⚠ {len(missing)} 個候選特徵不存在於 df，已略過")
        return feature_cols

    # ────────────────────────────────────────────────────────
    #  Optuna 調參 + OOF prob
    # ────────────────────────────────────────────────────────

    def _tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> tuple[dict, np.ndarray]:
        tscv = TimeSeriesSplit(n_splits=self.cfg.n_splits)

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
                m = lgb.train(
                    params,
                    lgb.Dataset(X_train.iloc[tr_idx], label=y_train.iloc[tr_idx]),
                    num_boost_round=1000,
                    valid_sets=[lgb.Dataset(X_train.iloc[va_idx], label=y_train.iloc[va_idx])],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                prob = np.clip(
                    m.predict(X_train.iloc[va_idx], num_iteration=m.best_iteration),
                    1e-7, 1 - 1e-7,
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

        params_final = {
            "objective":     "binary",
            "metric":        "binary_logloss",
            "verbosity":     -1,
            "boosting_type": "gbdt",
            **best_params,
        }
        oof_prob = np.zeros(len(y_train))
        for tr_idx, va_idx in tscv.split(X_train):
            m = lgb.train(
                params_final,
                lgb.Dataset(X_train.iloc[tr_idx], label=y_train.iloc[tr_idx]),
                num_boost_round=500,
            )
            oof_prob[va_idx] = m.predict(X_train.iloc[va_idx])

        return best_params, oof_prob

    # ────────────────────────────────────────────────────────
    #  Threshold 搜尋（OOF F1）
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
    #  最終模型
    # ────────────────────────────────────────────────────────

    def _train_final(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        best_params: dict,
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
            lgb.Dataset(X_train, label=y_train),
            num_boost_round=500,
        )

    # ────────────────────────────────────────────────────────
    #  特徵重要性（同 model.py _evaluate 格式）
    # ────────────────────────────────────────────────────────

    def _build_imp_df(self, model: lgb.Booster, feature_cols: list[str]) -> pd.DataFrame:
        imp_df = pd.DataFrame({
            "feature":    feature_cols,
            "importance": model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        imp_df["rank"] = range(1, len(imp_df) + 1)
        return imp_df

    # ────────────────────────────────────────────────────────
    #  SHAP
    # ────────────────────────────────────────────────────────

    def _compute_shap(
        self,
        model:     lgb.Booster,
        X_train:   pd.DataFrame,
        X_pred:    pd.DataFrame,
        pred_meta: pd.DataFrame,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        用 TreeExplainer 計算 SHAP 值。

        shap_target = "pred"  → 對預測集（當日所有股票）計算，速度快
        shap_target = "train" → 對訓練集計算，樣本多、較慢，適合全局解釋

        shap_sample_n = None  → 全量計算
        shap_sample_n = int   → 對 X_explain 隨機抽樣，大幅縮短 train 模式的耗時
                                 建議：pred 可設 None；train 建議 5000–20000

        回傳:
            shap_df         : 每股 × 每特徵的 SHAP 值（含 證券代碼 或 sample_index 欄位）
            shap_summary_df : 每特徵平均 |SHAP|，依重要性排序（Top shap_top_n）
        """
        try:
            import shap
        except ImportError:
            print("  ✗ 未安裝 shap，請執行：pip install shap")
            return None, None

        cfg = self.cfg

        # ── 選擇解釋目標 ─────────────────────────────────────
        if cfg.shap_target == "pred":
            X_explain  = X_pred.copy()
            is_pred    = True
        else:
            X_explain  = X_train.copy()
            is_pred    = False

        # ── 抽樣（shap_sample_n）────────────────────────────
        sampled = False
        if cfg.shap_sample_n is not None and len(X_explain) > cfg.shap_sample_n:
            X_explain = X_explain.sample(cfg.shap_sample_n, random_state=cfg.random_state)
            sampled   = True
            print(f"  SHAP 抽樣: {cfg.shap_sample_n:,} 筆 / {len(X_train):,} 筆訓練集")
        else:
            print(f"  目標: {cfg.shap_target}  樣本數: {len(X_explain):,}（全量）")

        print("  建立 TreeExplainer...")
        t_shap = time.time()

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        # 舊版 shap 對 binary classification 回傳 list[neg, pos]，取正類 [1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # shape: (n_samples, n_features)

        print(f"  SHAP 計算完成  耗時: {time.time() - t_shap:.1f} 秒")

        # ── shap_df（每樣本每特徵）──────────────────────────
        shap_df = pd.DataFrame(shap_values, columns=self._feature_cols)

        if is_pred and not sampled:
            # 預測集全量：附加 證券代碼
            shap_df.insert(0, "證券代碼", pred_meta["證券代碼"].values)
        elif is_pred and sampled:
            # 預測集抽樣：對齊 index 取 證券代碼
            shap_df.insert(0, "證券代碼", pred_meta.loc[X_explain.index, "證券代碼"].values)
        else:
            # 訓練集（全量或抽樣）：補 sample_index
            shap_df.insert(0, "sample_index", X_explain.index.tolist())

        # ── shap_summary_df（特徵平均 |SHAP|）───────────────
        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_summary_df = (
            pd.DataFrame({
                "feature":       self._feature_cols,
                "mean_abs_shap": mean_abs,
            })
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_summary_df["rank"] = range(1, len(shap_summary_df) + 1)

        # 顯示 Top N
        top_n = min(cfg.shap_top_n, len(shap_summary_df))
        print(f"\n  SHAP Top {top_n} 特徵（mean |SHAP|）：")
        print(shap_summary_df.head(top_n).to_string(index=False))

        return shap_df, shap_summary_df

    # ────────────────────────────────────────────────────────
    #  輸出
    # ────────────────────────────────────────────────────────

    def _save(
        self,
        pred_meta:       pd.DataFrame,
        pred_score:      np.ndarray,
        threshold:       float,
        predict_date:    date,
        best_params:     dict,
        imp_df:          pd.DataFrame,
        shap_df:         pd.DataFrame | None,
        shap_summary_df: pd.DataFrame | None,
    ):
        """
        組裝結果 DataFrame，輸出至 database/daily_predict/YYYYMMDD/：
            YYYYMMDD_prediction.csv        ── 完整預測分數（所有股票）
            feature_importance.csv         ── gain 重要性，同 model.py 格式
            best_params.json               ── Optuna 最佳超參數
            YYYYMMDD_shap.csv              ── SHAP 值（每股 × 每特徵）[若 compute_shap]
            YYYYMMDD_shap_summary.csv      ── 每特徵平均 |SHAP|          [若 compute_shap]

        另覆寫根目錄：
            final_day_prediction_sort.csv  ── 永遠是最新預測日

        is_selected 邏輯：rank <= top_n
        """
        cfg = self.cfg
        date_str = predict_date.strftime("%Y%m%d")

        out_dir = cfg.output_dir / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # ── 組裝 prediction df ───────────────────────────────
        result = pred_meta.copy()
        result["pred_score"]  = pred_score
        result = result.sort_values("pred_score", ascending=False).reset_index(drop=True)
        result["rank"]        = result.index + 1
        result["is_selected"] = (result["rank"] <= cfg.top_n).astype(int)
        result["年月日"]      = result["年月日"].dt.strftime("%Y-%m-%d")

        # ── 個別日期預測檔 ───────────────────────────────────
        pred_path = out_dir / f"{date_str}_prediction.csv"
        result.to_csv(pred_path, index=False, encoding="utf-8-sig")
        print(f"  ✓ {pred_path}  ({len(result):,} 股)")

        # ── feature_importance.csv ───────────────────────────
        imp_path = out_dir / "feature_importance.csv"
        imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")
        print(f"  ✓ {imp_path}  (Top 特徵: {imp_df.iloc[0]['feature']})")

        # ── best_params.json ─────────────────────────────────
        params_path = out_dir / "best_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {params_path}")

        # ── SHAP 輸出（若有）────────────────────────────────
        if shap_df is not None:
            shap_path = out_dir / f"{date_str}_shap.csv"
            shap_df.to_csv(shap_path, index=False, encoding="utf-8-sig")
            print(f"  ✓ {shap_path}  ({shap_df.shape[0]} 股 × {shap_df.shape[1]-1} 特徵)")

        if shap_summary_df is not None:
            summary_path = out_dir / f"{date_str}_shap_summary.csv"
            shap_summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"  ✓ {summary_path}")

        # ── final（根目錄覆寫）──────────────────────────────
        final_path = cfg.output_dir / "final_day_prediction_sort.csv"
        result.to_csv(final_path, index=False, encoding="utf-8-sig")
        print(f"  ✓ {final_path}  (覆寫)")

        # ── 選出股票概覽 ─────────────────────────────────────
        selected = result[result["is_selected"] == 1]
        print(f"\n  ► 今日選股（Top {cfg.top_n}）：")
        print(f"  預測日期: {predict_date}  threshold: {threshold:.2f}")
        print(selected[["證券代碼", "pred_score", "rank"]].to_string(index=False))


# ============================================================
#  MAIN
# ============================================================

def main():
    cfg = DailyConfig(
        train_years  = 2,
        top_n        = 20,
        n_trials     = 15,
        n_splits     = 3,
        # ── SHAP 設定 ────────────────────────────────────────
        compute_shap  = False,   # 改成 True 即啟用
        shap_target   = "pred",  # "pred"=預測集（快）/ "train"=訓練集（慢）
        shap_top_n    = 20,      # summary 顯示前幾名
        shap_sample_n = 5000,    # None=全量；離線分析 train 模式建議 5000–20000
    )

    print(f"\n{'='*60}")
    print(f"  Daily Prediction Pipeline")
    print(f"  訓練視窗  : 最近 {cfg.train_years} 年 raw_data + 當季 daily_data")
    print(f"  選股數量  : Top {cfg.top_n}")
    print(f"  Optuna    : {cfg.n_trials} trials / {cfg.n_splits} splits")
    if cfg.compute_shap:
        sample_info = f"{cfg.shap_sample_n:,} 筆抽樣" if cfg.shap_sample_n else "全量"
        print(f"  SHAP      : 開啟（{cfg.shap_target}，{sample_info}）")
    else:
        print(f"  SHAP      : 關閉")
    print(f"  輸出路徑  : {cfg.output_dir}")
    print(f"{'='*60}")

    trainer = DailyTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()