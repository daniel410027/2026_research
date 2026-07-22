"""
lgb_utils.py
============
LightGBM 共用工具函式（regression 版）。

WalkForwardTrainer（model.py）與 DailyTrainer（daily_model.py）
原本各自重複實作 _tune / _train_final / _find_threshold / _build_imp_df，
本模組將其統一，兩端共用同一套邏輯，避免日後改動時需同步修改兩處。

主要差異整合：
    - WalkForwardTrainer 支援 sample_weight（inverse-vol weighting）
    - DailyTrainer 不使用 sample_weight
    → tune_lgb / train_final_lgb 均以 sample_weight=None 為預設值，
      兩端呼叫時按需傳入即可，行為完全一致。

    [2025 更新] 新增 collect_oof_prob：
    - 超參數凍結模式下，跳過 Optuna，只用固定參數跑 CV 收集 OOF 預測值
    - 供 evaluate_oof_ic 使用，耗時約為 tune_lgb 的 1/n_trials

    [2026 修正] CV 切折以「唯一交易日」為單位（date-aware）：
    - 問題①：上游 train_df 若為 stock-major 排序，row-based TimeSeriesSplit
              會退化成「按股票分塊」（已於 model.py 端先 sort by 年月日 修正）。
    - 問題②：即使依日期排序，panel 同一交易日有上千檔股票，row-based 切折
              仍可能把「同一天」拆在 train/valid 邊界兩側 → 同日 cross-sectional
              外洩。本版改為以唯一交易日為切分單位（_make_cv_folds），確保同一天
              的所有股票完整落在同一側。
    - 問題③：TimeSeriesSplit 最早一段樣本永遠不在任何 valid fold，舊版 OOF 以
              0 初始化，使這些列被誤計入下游評估。
              本版 OOF 改以 NaN 初始化，evaluate_oof_ic 自動略過未預測列。

    向後相容：dates 為 None 時退回傳統 row-based TimeSeriesSplit
              （DailyTrainer 若未傳 dates，行為與舊版一致）。

    [2026-07-14 regression 改版] target 由 binary（excess_return_tick）
    改為連續值（excess_return）：
    - _LGB_BASE_PARAMS: objective binary→regression, metric binary_logloss→rmse
    - Optuna objective() 的 CV loss 由 log_loss 改為 RMSE
    - find_threshold（OOF F1 最佳機率門檻）→ evaluate_oof_ic（Spearman IC 監控，
      不做 threshold；regression target 沒有天然的 0/1 切點，選股門檻改由
      呼叫端用預測值排序 / decile 決定）
    - oof_prob 變數名沿用（僅為 CV 折內部命名慣例），實際內容已是連續預測值，
      非機率

公開函式：
    tune_lgb          Optuna TPE 調參 + OOF 預測值收集
    collect_oof_prob  固定超參數下只收集 OOF 預測值（凍結模式用）
    train_final_lgb   最終模型訓練（全量 X_train）
    evaluate_oof_ic   OOF Spearman IC 監控（自動略過 NaN，不做 threshold）
    build_imp_df      特徵重要性 DataFrame（gain）

作者：Daniel Huang
"""

from __future__ import annotations

import os
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ============================================================
#  LightGBM 超參數搜尋空間（集中定義，避免兩端各自維護）
# ============================================================

# ★ n-2 核平行運算：保留 2 核給系統/其他行程，避免搶占整台機器
_N_JOBS = max(1, (os.cpu_count() or 1) - 2)

_LGB_BASE_PARAMS = {
    "objective":     "regression",
    "metric":        "rmse",
    "verbosity":     -1,
    "boosting_type": "gbdt",
    "num_threads":   _N_JOBS,
    # ★ 2026-07-22：max_depth 固定 -1（不限制），移出 _SEARCH_SPACE。
    #   max_depth 是硬上限（depth=d → 葉數 ≤ 2^d），與 num_leaves 同時搜會冗餘：
    #   10 折 walk-forward 實測有 3 折的 num_leaves 完全被蓋掉
    #   （如 num_leaves=957 但 max_depth=3 → 實際只有 8 個葉子），既浪費 trial
    #   預算，也會讓重要性分析把 num_leaves 誤判為不重要（它常被 max_depth 遮蔽）。
    #   改由 num_leaves 單獨控制樹複雜度，語意單純。
    "max_depth":     -1,
}

_SEARCH_SPACE = {
    # ★ 2026-07-22：learning_rate 是 300 trials 中 Spearman ρ 唯一顯著的超參數
    #   （ρ=+0.514，其餘全在雜訊等級），分箱證據顯示 >0.02 幾乎全是爛區
    #   （0.1~0.3 桶平均 rank 0.896 vs ≤0.005 桶 0.383）。原範圍 [0.005, 0.3]
    #   約一半 trial 浪費在爛區，收窄到 [0.001, 0.03] 讓預算集中在有效區間。
    "learning_rate":    ("float_log", 0.001, 0.03),
    # ρ = −0.013，與目標值幾乎無關；保留原範圍但不值得為它加 trial 預算。
    "num_leaves":       ("int",       16,    512),
    "feature_fraction": ("float",     0.5,   1.0),
    "bagging_fraction": ("float",     0.5,   1.0),
    "bagging_freq":     ("int",       1,     20),
    # ★ 2026-07-22：上界 100 → 600。ρ = −0.23，是唯一次要有訊號的參數，且舊上界
    #   確實在綁：舊空間 [5,100] 的 10 折觀測值是 41~91（看似未貼界），放寬到 600
    #   後立刻跑到 144~599。噪音大、樣本少時觀測最大值會系統性低估最適區，
    #   「沒有貼界」不代表「界沒有在綁」。
    "min_data_in_leaf": ("int",       5,     600),
    # max_depth 已移出（見 _LGB_BASE_PARAMS）
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
#  CV 切折（date-aware）
# ============================================================

def _make_cv_folds(
    X:        pd.DataFrame,
    n_splits: int,
    dates:    pd.Series | np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    產生 [(train_pos, valid_pos), ...] 折清單（expanding window）。

    dates 提供時（建議）：
        以「唯一交易日」為切分單位，先對 unique dates 跑 TimeSeriesSplit，
        再映回 row 位置。確保同一交易日的所有股票完整落在 train 或 valid
        同一側，消除 panel 同日 cross-sectional 邊界外洩。

    dates 為 None：
        退回傳統 row-based TimeSeriesSplit（向後相容）。

    Parameters
    ----------
    X        : 訓練特徵（僅用於 row 數 / row-based fallback）
    n_splits : 摺數
    dates    : 與 X 逐列對齊的日期序列（model.py 傳入 train_df["年月日"]）

    Returns
    -------
    list[(train_pos, valid_pos)]  ── 皆為 positional int 索引（供 .iloc 使用）
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    if dates is None:
        return list(tscv.split(X))

    d = pd.to_datetime(pd.Series(np.asarray(dates))).reset_index(drop=True)
    uniq = np.sort(d.unique())

    # 唯一日期數不足以切 n_splits 折 → 退回 row-based
    if len(uniq) <= n_splits:
        return list(tscv.split(X))

    pos  = np.arange(len(d))
    d_np = d.to_numpy()
    folds = []
    for tr_d_idx, va_d_idx in tscv.split(uniq):
        tr_mask = np.isin(d_np, uniq[tr_d_idx])
        va_mask = np.isin(d_np, uniq[va_d_idx])
        folds.append((pos[tr_mask], pos[va_mask]))
    return folds


# ============================================================
#  公開函式
# ============================================================

def tune_lgb(
    X_train:           pd.DataFrame,
    y_train:           pd.Series,
    n_splits:          int,
    n_trials:          int,
    random_state:      int,
    sample_weight:     np.ndarray | None = None,
    dates:             pd.Series | np.ndarray | None = None,
    return_best_iters: bool = False,
    return_cv_details: bool = False,
) -> (
    tuple[dict, np.ndarray]
    | tuple[dict, np.ndarray, list[int]]
    | tuple[dict, np.ndarray, list[float]]
    | tuple[dict, np.ndarray, list[int], list[float]]
):
    """
    Optuna TPE 調參 + 最佳參數 OOF prob 收集（date-aware CV）。

    最佳參數確定後，只再跑一輪 CV 收集 OOF prob，
    供 evaluate_oof_ic 監控使用，不額外增加訓練負擔。

    [2026-06 更新] return_best_iters=True 時：
    - OOF 收集改用「1000 輪 + ES(50)」（與 objective 內的訓練方式一致），
      並回傳 (best_params, oof_prob, best_iters)；
      best_iters 供呼叫端設定 final model 的 num_boost_round。
    - 預設 False 時行為與舊版相同（model.py 不受影響）。

    Parameters
    ----------
    X_train, y_train  : 訓練集
    n_splits          : 摺數
    n_trials          : Optuna trial 數
    random_state      : 隨機種子
    sample_weight     : 可選 shape=(n_train,)（inverse-vol weight 等）
    dates             : 可選，與 X_train 逐列對齊的日期序列；
                        提供時以唯一交易日為單位切折（建議）
    return_best_iters : True 時回傳 3-tuple（含每折 best_iteration）
    return_cv_details : True 時額外回傳最佳 trial 的逐折 CV RMSE
                        （來自 Optuna best_trial 的 objective 內部評估，
                        不重新訓練，不增加額外耗時）

    Returns
    -------
    best_params : dict         ── Optuna 最佳超參數（不含 base params）
    oof_prob    : np.ndarray   ── OOF 連續預測值（未被任何 valid fold 命中者為 NaN）
    best_iters  : list[int]    ── 僅 return_best_iters=True 時
    best_fold_losses : list[float] ── 僅 return_cv_details=True 時，
                        最佳超參數組合在 Optuna 搜尋當下、各折的 RMSE
                        （與 objective() 內計算方式一致，用於 cv_metrics.json）
    """
    cv_folds = _make_cv_folds(X_train, n_splits, dates)

    # ── Optuna ──────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        # ★ seed 固定 LightGBM 內部所有隨機來源（bagging/feature_fraction/
        #   data 等），確保同一組 trial 超參數在重跑時得到相同結果。
        params = {
            **_LGB_BASE_PARAMS,
            **_suggest_params(trial),
            "seed": random_state,
        }
        fold_losses = []
        for tr_idx, va_idx in cv_folds:
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
            pred = m.predict(X_train.iloc[va_idx], num_iteration=m.best_iteration)
            rmse = float(np.sqrt(np.mean((y_train.iloc[va_idx].to_numpy() - pred) ** 2)))
            fold_losses.append(rmse)
        # ★ 存進 trial user_attrs，供搜尋結束後從 study.best_trial 撈出，
        #   不需重新訓練即可取得最佳超參數當下的逐折 CV RMSE。
        trial.set_user_attr("fold_losses", fold_losses)
        return float(np.mean(fold_losses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_fold_losses = list(study.best_trial.user_attrs.get("fold_losses", []))
    print(f"  Best CV RMSE: {study.best_value:.6f}")

    # ── 最佳參數 OOF prob ────────────────────────────────────
    if return_best_iters:
        # 與 objective 一致：1000 輪 + ES(50)，並回收每折 best_iteration
        oof_prob, best_iters = collect_oof_prob(
            X_train, y_train, best_params, n_splits, sample_weight, dates,
            num_boost_round=1000, early_stopping_rounds=50, return_best_iters=True,
            random_state=random_state,
        )
        if return_cv_details:
            return best_params, oof_prob, best_iters, best_fold_losses
        return best_params, oof_prob, best_iters

    oof_prob = collect_oof_prob(
        X_train, y_train, best_params, n_splits, sample_weight, dates,
        random_state=random_state,
    )

    if return_cv_details:
        return best_params, oof_prob, best_fold_losses
    return best_params, oof_prob


def collect_oof_prob(
    X_train:               pd.DataFrame,
    y_train:               pd.Series,
    best_params:           dict,
    n_splits:              int,
    sample_weight:         np.ndarray | None = None,
    dates:                 pd.Series | np.ndarray | None = None,
    num_boost_round:       int = 500,
    early_stopping_rounds: int | None = None,
    return_best_iters:     bool = False,
    random_state:          int | None = None,
) -> np.ndarray | tuple[np.ndarray, list[int]]:
    """
    固定超參數下，只跑 date-aware CV 收集 OOF prob。

    用於超參數凍結模式：跳過 Optuna，直接用 frozen params 評估 threshold，
    耗時約為 tune_lgb 的 1/n_trials。

    [2026-06 更新] 新增 early stopping 與 best_iteration 回傳：
    - early_stopping_rounds 提供時，每折以 valid fold 做 ES，
      OOF prob 由 best_iteration 截斷的模型產生（與 tune_lgb objective 一致）。
    - return_best_iters=True 時回傳 (oof_prob, best_iters)，
      供呼叫端決定 final model 的 num_boost_round（取中位數等）。
    - 預設參數下行為與舊版完全相同（500 輪、無 ES、只回傳 oof_prob），
      model.py 等既有呼叫端不受影響。

    Parameters
    ----------
    X_train, y_train      : 訓練集
    best_params           : 已凍結的超參數 dict（不含 base params）
    n_splits              : 摺數
    sample_weight         : 可選 shape=(n_train,)
    dates                 : 可選，與 X_train 逐列對齊的日期序列
    num_boost_round       : 每折最大輪數（預設 500；搭配 ES 時建議 1000）
    early_stopping_rounds : 提供時啟用 early stopping
    return_best_iters     : True 時回傳 (oof_prob, best_iters)
    random_state          : 提供時固定 LightGBM 內部隨機種子（seed），
                             確保同一組 best_params 重跑時 OOF 結果一致

    Returns
    -------
    oof_prob : np.ndarray  ── 與 y_train 等長（實際為連續預測值）；
               最早一段（從不在任何 valid fold）保持 NaN，
               由 evaluate_oof_ic 自動略過，避免污染 IC 計算。
    best_iters : list[int]（僅 return_best_iters=True 時）
               每折的 best_iteration（無 ES 時為 num_boost_round）。
    """
    cv_folds   = _make_cv_folds(X_train, n_splits, dates)
    params     = {**_LGB_BASE_PARAMS, **best_params}
    if random_state is not None:
        params["seed"] = random_state
    oof_prob   = np.full(len(y_train), np.nan)   # NaN 初始化（非 0）
    best_iters = []

    for tr_idx, va_idx in cv_folds:
        w_tr = sample_weight[tr_idx] if sample_weight is not None else None
        w_va = sample_weight[va_idx] if sample_weight is not None else None
        train_set = lgb.Dataset(
            X_train.iloc[tr_idx], label=y_train.iloc[tr_idx], weight=w_tr
        )
        if early_stopping_rounds is not None:
            m = lgb.train(
                params,
                train_set,
                num_boost_round=num_boost_round,
                valid_sets=[
                    lgb.Dataset(
                        X_train.iloc[va_idx], label=y_train.iloc[va_idx], weight=w_va
                    )
                ],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            best_iters.append(int(m.best_iteration))
            oof_prob[va_idx] = m.predict(
                X_train.iloc[va_idx], num_iteration=m.best_iteration
            )
        else:
            m = lgb.train(params, train_set, num_boost_round=num_boost_round)
            best_iters.append(num_boost_round)
            oof_prob[va_idx] = m.predict(X_train.iloc[va_idx])

    if return_best_iters:
        return oof_prob, best_iters
    return oof_prob


def train_final_lgb(
    X_train:         pd.DataFrame,
    y_train:         pd.Series,
    best_params:     dict,
    sample_weight:   np.ndarray | None = None,
    num_boost_round: int               = 500,
    random_state:    int | None        = None,
) -> lgb.Booster:
    """
    最終模型訓練（全量 X_train，不做 early stopping）。

    Parameters
    ----------
    best_params     : tune_lgb 回傳的超參數 dict（不含 base params）
    sample_weight   : 可選 shape=(n_train,)
    num_boost_round : 訓練輪數（預設 500）
    random_state    : 提供時固定 LightGBM 內部隨機種子（seed），
                       確保相同資料 + 相同超參數重跑時模型完全一致

    Returns
    -------
    lgb.Booster
    """
    params = {**_LGB_BASE_PARAMS, **best_params}
    if random_state is not None:
        params["seed"] = random_state
    return lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train, weight=sample_weight),
        num_boost_round=num_boost_round,
    )


def evaluate_oof_ic(y_true: pd.Series, oof_pred: np.ndarray) -> float:
    """
    計算 OOF 預測值與真實值的 Spearman IC（rank correlation），供監控用。

    自動略過 oof_pred 為 NaN 的列（最早一段未被任何 valid fold 命中者），
    避免將「未預測」誤計入 IC 計算。

    不做 threshold 搜尋——regression target 下沒有天然的 0/1 切點，
    改用 IC 監控模型排序品質，實際選股門檻交由呼叫端（如 decile / top-K）決定。

    Returns
    -------
    float  ── Spearman IC（rank correlation），無有效樣本時回傳 np.nan
    """
    y_true   = np.asarray(y_true)
    oof_pred = np.asarray(oof_pred, dtype=float)

    mask = ~np.isnan(oof_pred)
    n_valid = int(mask.sum())
    if n_valid == 0:
        print("  ⚠ 無有效 OOF 預測值，IC 無法計算")
        return float("nan")

    ic, _ = spearmanr(y_true[mask], oof_pred[mask])
    print(f"  OOF Spearman IC: {ic:.4f}  (n={n_valid:,})")
    return float(ic)


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