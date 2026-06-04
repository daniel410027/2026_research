"""
lgb_utils.py
============
LightGBM 共用工具函式。

WalkForwardTrainer（model.py）與 DailyTrainer（daily_model.py）
原本各自重複實作 _tune / _train_final / _find_threshold / _build_imp_df，
本模組將其統一，兩端共用同一套邏輯，避免日後改動時需同步修改兩處。

主要差異整合：
    - WalkForwardTrainer 支援 sample_weight（inverse-vol weighting）
    - DailyTrainer 不使用 sample_weight
    → tune_lgb / train_final_lgb 均以 sample_weight=None 為預設值，
      兩端呼叫時按需傳入即可，行為完全一致。

    [2025 更新] 新增 collect_oof_prob：
    - 超參數凍結模式下，跳過 Optuna，只用固定參數跑 CV 收集 OOF prob
    - 供 find_threshold 使用，耗時約為 tune_lgb 的 1/n_trials

公開函式：
    tune_lgb          Optuna TPE 調參 + OOF prob 收集
    collect_oof_prob  固定超參數下只收集 OOF prob（凍結模式用）
    train_final_lgb   最終模型訓練（全量 X_train）
    find_threshold    OOF F1 最佳 threshold 搜尋
    build_imp_df      特徵重要性 DataFrame（gain）

作者：Daniel Huang
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ============================================================
#  LightGBM 超參數搜尋空間（集中定義，避免兩端各自維護）
# ============================================================

_LGB_BASE_PARAMS = {
    "objective":     "binary",
    "metric":        "binary_logloss",
    "verbosity":     -1,
    "boosting_type": "gbdt",
}

_SEARCH_SPACE = {
    "learning_rate":    ("float_log", 0.005, 0.3),
    "num_leaves":       ("int",       16,    512),
    "feature_fraction": ("float",     0.5,   1.0),
    "bagging_fraction": ("float",     0.5,   1.0),
    "bagging_freq":     ("int",       1,     20),
    "min_data_in_leaf": ("int",       5,     100),
    "max_depth":        ("int",       5,     20),
}


def _suggest_params(trial: optuna.Trial) -> dict:
    """從 _SEARCH_SPACE 統一產生 trial 超參數。"""
    params = {}
    for name, spec in _SEARCH_SPACE.items():
        if spec[0] == "float_log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif spec[0] == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif spec[0] == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
    return params


# ============================================================
#  公開函式
# ============================================================

def tune_lgb(
    X_train:       pd.DataFrame,
    y_train:       pd.Series,
    n_splits:      int,
    n_trials:      int,
    random_state:  int,
    sample_weight: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    """
    Optuna TPE 調參 + 最佳參數 OOF prob 收集（TimeSeriesSplit）。

    最佳參數確定後，只再跑一輪 CV 收集 OOF prob，
    供 find_threshold 使用，不額外增加訓練負擔。

    Parameters
    ----------
    X_train, y_train : 訓練集
    n_splits         : TimeSeriesSplit 摺數
    n_trials         : Optuna trial 數
    random_state     : 隨機種子
    sample_weight    : 可選 shape=(n_train,)（inverse-vol weight 等）

    Returns
    -------
    best_params : dict         ── Optuna 最佳超參數（不含 base params）
    oof_prob    : np.ndarray   ── OOF 預測機率，與 y_train 等長
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # ── Optuna ──────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        params = {**_LGB_BASE_PARAMS, **_suggest_params(trial)}
        fold_losses = []
        for tr_idx, va_idx in tscv.split(X_train):
            w_tr = sample_weight[tr_idx] if sample_weight is not None else None
            w_va = sample_weight[va_idx] if sample_weight is not None else None
            m = lgb.train(
                params,
                lgb.Dataset(
                    X_train.iloc[tr_idx], label=y_train.iloc[tr_idx], weight=w_tr
                ),
                num_boost_round=1000,
                valid_sets=[
                    lgb.Dataset(
                        X_train.iloc[va_idx], label=y_train.iloc[va_idx], weight=w_va
                    )
                ],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            prob = np.clip(
                m.predict(X_train.iloc[va_idx], num_iteration=m.best_iteration),
                1e-7, 1 - 1e-7,
            )
            fold_losses.append(log_loss(y_train.iloc[va_idx], prob))
        return float(np.mean(fold_losses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    print(f"  Best CV log-loss: {study.best_value:.6f}")

    # ── 最佳參數 OOF prob ────────────────────────────────────
    oof_prob = collect_oof_prob(X_train, y_train, best_params, n_splits, sample_weight)

    return best_params, oof_prob


def collect_oof_prob(
    X_train:       pd.DataFrame,
    y_train:       pd.Series,
    best_params:   dict,
    n_splits:      int,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """
    固定超參數下，只跑 TimeSeriesSplit CV 收集 OOF prob。

    用於超參數凍結模式：跳過 Optuna，直接用 frozen params 評估 threshold，
    耗時約為 tune_lgb 的 1/n_trials。

    Parameters
    ----------
    X_train, y_train : 訓練集
    best_params      : 已凍結的超參數 dict（不含 base params）
    n_splits         : TimeSeriesSplit 摺數
    sample_weight    : 可選 shape=(n_train,)

    Returns
    -------
    oof_prob : np.ndarray  ── OOF 預測機率，與 y_train 等長
    """
    tscv     = TimeSeriesSplit(n_splits=n_splits)
    params   = {**_LGB_BASE_PARAMS, **best_params}
    oof_prob = np.zeros(len(y_train))

    for tr_idx, va_idx in tscv.split(X_train):
        w_tr = sample_weight[tr_idx] if sample_weight is not None else None
        m = lgb.train(
            params,
            lgb.Dataset(
                X_train.iloc[tr_idx], label=y_train.iloc[tr_idx], weight=w_tr
            ),
            num_boost_round=500,
        )
        oof_prob[va_idx] = m.predict(X_train.iloc[va_idx])

    return oof_prob


def train_final_lgb(
    X_train:         pd.DataFrame,
    y_train:         pd.Series,
    best_params:     dict,
    sample_weight:   np.ndarray | None = None,
    num_boost_round: int               = 500,
) -> lgb.Booster:
    """
    最終模型訓練（全量 X_train，不做 early stopping）。

    Parameters
    ----------
    best_params     : tune_lgb 回傳的超參數 dict（不含 base params）
    sample_weight   : 可選 shape=(n_train,)
    num_boost_round : 訓練輪數（預設 500）

    Returns
    -------
    lgb.Booster
    """
    params = {**_LGB_BASE_PARAMS, **best_params}
    return lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train, weight=sample_weight),
        num_boost_round=num_boost_round,
    )


def find_threshold(y_true: pd.Series, oof_prob: np.ndarray) -> float:
    """
    在 [0.05, 0.70) 以 0.01 為步距搜尋，最大化 OOF F1 的 threshold。

    Returns
    -------
    float  ── 最佳 threshold（四捨五入至小數點後 2 位）
    """
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.05, 0.70, 0.01):
        f1 = f1_score(y_true, (oof_prob >= thresh).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    print(f"  Best threshold: {best_thresh:.2f}  (OOF F1={best_f1:.4f})")
    return round(float(best_thresh), 2)


def build_imp_df(model: lgb.Booster, feature_cols: list[str]) -> pd.DataFrame:
    """
    回傳 gain 重要性 DataFrame，欄位：feature, importance, rank。
    格式與 model.py、daily_model.py 輸出相容。
    """
    imp_df = (
        pd.DataFrame({
            "feature":    feature_cols,
            "importance": model.feature_importance(importance_type="gain"),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df["rank"] = range(1, len(imp_df) + 1)
    return imp_df