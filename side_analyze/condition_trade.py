"""
strategy.py
===========
條件交易策略模組（Walk-forward 版）。

取代 model.py 的機器學習方法，改以技術指標條件式評分產生訊號，
輸出格式與 model.py 完全相容，可直接銜接 report.py 進行回測。

【交易假設（沿用 report.py 設定）】
    訊號日 t  →  t+1 開盤買入  →  t+2 開盤賣出
    return = open_{t+2} / open_{t+1} - 1

【輸出格式（對齊 model.py predictions.csv）】
    證券代碼, 年月日, return, y_true, y_prob, y_pred

    y_prob : 複合策略評分，介於 [0, 1]
             數值越高代表訊號越強（買入傾向越高）
    y_pred : y_prob >= threshold → 1（買入訊號），否則 0

【內建策略（可組合）】
    MomentumStrategy     : RSI 動能 + MACD 多頭 + 價格在 SMA 之上
    MeanReversionStrategy: RSI 超賣 + Bollinger Band 觸底
    MultifactorStrategy  : 多因子加權（動能 + 均值回歸 + 成交量 + 波動度）
    CompositeStrategy    : 依設定比例混合上述策略

【Walk-forward 規則（同 model.py）】
    Windows: (2014,2016), (2015,2017), ..., (2022,2024)
    每個 window:
        - in-sample  = 前兩年（用於計算閾值自適應）
        - test       = 第三年（產生訊號並輸出）

作者：Daniel Huang
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from preprocess import PreprocessConfig, DataPreprocessor

warnings.filterwarnings("ignore")


# ============================================================
#  CONFIG
# ============================================================

@dataclass
class StrategyConfig:
    """條件交易策略設定，所有超參數集中於此。"""

    # ── Walk-forward 視窗 ────────────────────────────────────
    window_start: int = 2014
    window_end:   int = 2025

    # ── 策略選擇 ─────────────────────────────────────────────
    strategy: Literal[
        "momentum",       # RSI 動能 + MACD + 價格趨勢
        "mean_reversion", # RSI 超賣 + Bollinger 觸底
        "multifactor",    # 多因子加權評分
        "composite",      # 混合策略（依下方比例）
    ] = "multifactor"

    # ── 複合策略比例（strategy="composite" 時有效）───────────
    composite_weights: dict = field(default_factory=lambda: {
        "momentum":       0.35,
        "mean_reversion": 0.25,
        "multifactor":    0.40,
    })

    # ── 訊號閾值 ─────────────────────────────────────────────
    # y_prob >= signal_threshold → y_pred = 1（買入）
    # None = 自動以 in-sample 最佳化決定（以 top N% 為閾值）
    signal_threshold: float | None = None
    top_pct: float = 0.20   # 取評分前 20% 為買入訊號

    # ── 目標欄位 ─────────────────────────────────────────────
    target_col: str = "excess_return_tick"

    # ── 輸出路徑 ─────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: Path("database/condition_trade"))

    # ── Preprocess 設定（同 model.py）───────────────────────
    raw_data_dir:     Path  = field(default_factory=lambda: Path("database/raw_data"))
    market_data_path: Path  = field(default_factory=lambda: Path("database/market/market.csv"))
    return_clip:      float = 0.15
    tick_threshold:   float = 0.01
    tick0_threshold:  float = 0.00
    beta_window:      int   = 60

    def __post_init__(self):
        self.output_dir       = Path(self.output_dir)
        self.raw_data_dir     = Path(self.raw_data_dir)
        self.market_data_path = Path(self.market_data_path)

    def windows(self) -> list[tuple[int, int]]:
        """回傳所有 walk-forward window，格式為 (start, end)。"""
        return [(s, s + 2) for s in range(self.window_start, self.window_end - 1)]


# ============================================================
#  策略評分函式（均回傳 pd.Series，index 同輸入 df，值域 [0,1]）
# ============================================================

class SignalScorer:
    """
    各策略的評分計算器。
    所有方法接受完整 df（含技術指標），回傳截面標準化至 [0,1] 的評分 Series。
    截面標準化：每日 rank 正規化，消除跨日絕對值差異。
    """

    # ────────────────────────────────────────────────────────
    #  截面 Rank 正規化（每日）
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _cross_section_rank(series: pd.Series, date_col: pd.Series) -> pd.Series:
        """
        每日截面 rank 正規化到 [0, 1]。
        相較 z-score，rank 對 outlier 更穩健，且與 quantile 策略直觀對齊。
        """
        result = series.copy().astype(float)
        for date, group_idx in series.groupby(date_col).groups.items():
            vals = series.loc[group_idx]
            n = vals.notna().sum()
            if n < 2:
                result.loc[group_idx] = 0.5
                continue
            ranks = rankdata(vals.fillna(vals.median()), method="average")
            result.loc[group_idx] = (ranks - 1) / (n - 1)
        return result

    # ────────────────────────────────────────────────────────
    #  1. Momentum Strategy
    # ────────────────────────────────────────────────────────

    @staticmethod
    def momentum_score(df: pd.DataFrame) -> pd.Series:
        """
        動能策略評分。

        訊號邏輯（三者皆成立時評分高）：
            - RSI_20 在 40~70 之間（動能適中，非超買/超賣）
            - MACD_Hist > 0（MACD 柱狀圖翻正，上升動能）
            - Price_SMA20_Ratio > 0（價格在 20 日均線以上）

        評分方式：三個子訊號加權 → 截面 rank 正規化
        """
        score = pd.Series(0.0, index=df.index)
        date_col = df["年月日"]

        # 子訊號 1：RSI 動能帶（30~70 最佳，越接近 50 加 1 分，超買超賣減分）
        if "RSI_20" in df.columns:
            rsi = df["RSI_20"]
            rsi_score = np.where(
                (rsi >= 40) & (rsi <= 65),   1.0,   # 最佳動能區間
                np.where(
                    (rsi >= 30) & (rsi < 40), 0.5,   # 輕微超賣，仍可接受
                    np.where(rsi > 65,        0.3,   # 趨近超買，動能衰竭風險
                    0.0)                              # 嚴重超賣或極端超買
                )
            )
            score += 0.35 * rsi_score

        # 子訊號 2：MACD Histogram 正負（正 = 多頭加速）
        if "MACD_Hist" in df.columns:
            score += 0.35 * (df["MACD_Hist"] > 0).astype(float)

        # 子訊號 3：KD 多頭排列（K > D，且 K 從低位回升）
        if "K" in df.columns and "D" in df.columns:
            kd_bull = ((df["K"] > df["D"]) & (df["D"] < 80)).astype(float)
            score += 0.15 * kd_bull

        # 子訊號 4：價格在 SMA20 以上（中期趨勢確認）
        if "Price_SMA20_Ratio" in df.columns:
            price_above_sma = (df["Price_SMA20_Ratio"] > 0).astype(float)
            score += 0.15 * price_above_sma

        return SignalScorer._cross_section_rank(score, date_col)

    # ────────────────────────────────────────────────────────
    #  2. Mean Reversion Strategy
    # ────────────────────────────────────────────────────────

    @staticmethod
    def mean_reversion_score(df: pd.DataFrame) -> pd.Series:
        """
        均值回歸策略評分。

        訊號邏輯（股票被過度超賣，等待反彈）：
            - RSI_10 < 30（短期超賣）
            - BB_PctB_20 < 0.2（價格接近布林下軌）
            - Williams_R_14 < -80（短期嚴重超賣）
            - 成交量異常放大（大量下跌後底部確認訊號）

        注意：此策略適合震盪盤，趨勢下跌市場效果較差。
        """
        score = pd.Series(0.0, index=df.index)
        date_col = df["年月日"]

        # 子訊號 1：RSI 超賣（越低評分越高，但非常極端的低值可能是崩跌）
        if "RSI_10" in df.columns:
            rsi10 = df["RSI_10"].clip(0, 100)
            rsi_score = np.where(
                rsi10 < 20,  0.9,   # 嚴重超賣，高反彈機率
                np.where(
                    rsi10 < 30, 0.7,  # 超賣
                    np.where(rsi10 < 40, 0.4, 0.0)  # 接近超賣 / 中性
                )
            )
            score += 0.35 * rsi_score

        elif "RSI_20" in df.columns:
            # fallback 使用 RSI_20
            rsi20 = df["RSI_20"].clip(0, 100)
            rsi_score = np.where(rsi20 < 30, 0.8, np.where(rsi20 < 40, 0.4, 0.0))
            score += 0.35 * rsi_score

        # 子訊號 2：Bollinger Band 觸底（%B 越低，越接近下軌）
        if "BB_PctB_20" in df.columns:
            pct_b = df["BB_PctB_20"]
            bb_score = np.where(
                pct_b < 0.0,  1.0,   # 突破下軌
                np.where(pct_b < 0.2, 0.7,  # 接近下軌
                np.where(pct_b < 0.4, 0.3, 0.0))
            )
            score += 0.30 * bb_score

        # 子訊號 3：Williams %R 超賣
        if "Williams_R_14" in df.columns:
            wr = df["Williams_R_14"]
            wr_score = np.where(wr < -90, 1.0, np.where(wr < -80, 0.6, 0.0))
            score += 0.20 * wr_score

        # 子訊號 4：成交量放大（底部確認）
        if "Volume_Ratio_5" in df.columns:
            vol_surge = (df["Volume_Ratio_5"] > 1.5).astype(float)
            score += 0.15 * vol_surge

        return SignalScorer._cross_section_rank(score, date_col)

    # ────────────────────────────────────────────────────────
    #  3. Multifactor Strategy
    # ────────────────────────────────────────────────────────

    @staticmethod
    def multifactor_score(df: pd.DataFrame) -> pd.Series:
        """
        多因子加權策略評分（仿 Barra 多因子模型的條件版）。

        因子構成：
            動能因子   (30%) : ROC_20 截面 rank
            波動因子   (20%) : 低波動偏好（vol20 inverse rank）
            成交量因子 (20%) : 成交量動能（OBV_ROC 截面 rank）
            價值因子   (15%) : CCI 反轉訊號（CCI < -100 超賣）
            趨勢因子   (15%) : 多均線多頭排列（SMA20 > SMA60 > SMA120）

        每個子因子獨立截面 rank 後加權求和，最終再做截面 rank。
        """
        date_col = df["年月日"]
        sub_scores = []

        # 因子 1：動能（ROC_20）
        if "ROC_20" in df.columns:
            roc_rank = SignalScorer._cross_section_rank(df["ROC_20"], date_col)
            sub_scores.append(("momentum", 0.30, roc_rank))

        # 因子 2：低波動（vol20 inverse，低波動 = 高評分）
        if "vol20" in df.columns:
            inv_vol = -df["vol20"]   # 取負值，低波動排名靠前
            vol_rank = SignalScorer._cross_section_rank(inv_vol, date_col)
            sub_scores.append(("low_vol", 0.20, vol_rank))

        # 因子 3：成交量動能（OBV_ROC）
        if "OBV_ROC" in df.columns:
            obv_rank = SignalScorer._cross_section_rank(df["OBV_ROC"], date_col)
            sub_scores.append(("volume", 0.20, obv_rank))

        # 因子 4：反轉因子（CCI 超賣反轉，CCI 越低 → 評分越高）
        if "CCI_14" in df.columns:
            inv_cci = -df["CCI_14"]
            cci_rank = SignalScorer._cross_section_rank(inv_cci, date_col)
            sub_scores.append(("reversal", 0.15, cci_rank))

        # 因子 5：趨勢一致性（多均線多頭排列）
        trend_signal = pd.Series(0.0, index=df.index)
        if all(c in df.columns for c in ["SMA_20", "SMA_60", "SMA_120"]):
            trend_signal += (df["SMA_20"] > df["SMA_60"]).astype(float) * 0.5
            trend_signal += (df["SMA_60"] > df["SMA_120"]).astype(float) * 0.5
        elif "Price_SMA20_Ratio" in df.columns:
            trend_signal = (df["Price_SMA20_Ratio"] > 0).astype(float)
        sub_scores.append(("trend", 0.15, trend_signal))

        if not sub_scores:
            return pd.Series(0.5, index=df.index)

        # 加權合成
        total_weight = sum(w for _, w, _ in sub_scores)
        composite = sum(w * s for _, w, s in sub_scores) / total_weight

        return SignalScorer._cross_section_rank(composite, date_col)

    # ────────────────────────────────────────────────────────
    #  4. Composite Strategy
    # ────────────────────────────────────────────────────────

    @staticmethod
    def composite_score(
        df: pd.DataFrame,
        weights: dict[str, float] | None = None,
    ) -> pd.Series:
        """
        混合策略評分（依比例加權三個子策略）。

        Parameters
        ----------
        weights : dict
            {"momentum": float, "mean_reversion": float, "multifactor": float}
        """
        if weights is None:
            weights = {"momentum": 0.35, "mean_reversion": 0.25, "multifactor": 0.40}

        scores = {
            "momentum":       SignalScorer.momentum_score(df),
            "mean_reversion": SignalScorer.mean_reversion_score(df),
            "multifactor":    SignalScorer.multifactor_score(df),
        }

        total = sum(weights.values())
        composite = sum(weights[k] * s for k, s in scores.items() if k in weights) / total

        return SignalScorer._cross_section_rank(composite, df["年月日"])


# ============================================================
#  Walk-forward Backtester
# ============================================================

class WalkForwardBacktester:
    """
    Walk-forward 條件策略回測器。

    每個 window：
        - in-sample  (前兩年) : 計算閾值自適應（若 signal_threshold=None）
        - test       (第三年) : 產生訊號，輸出 predictions.csv
    """

    def __init__(self, config: StrategyConfig):
        self.cfg = config
        self._df_cache: dict[tuple[int, int], pd.DataFrame] = {}

    # ────────────────────────────────────────────────────────
    #  Cache 注入（對齊 model.py 介面）
    # ────────────────────────────────────────────────────────

    def inject_cache(self, start: int, end: int, df: pd.DataFrame):
        self._df_cache[(start, end)] = df
        print(f"  ✓ 注入 df cache：window ({start}, {end})，形狀 {df.shape}")

    # ────────────────────────────────────────────────────────
    #  資料取得
    # ────────────────────────────────────────────────────────

    def _get_df(self, start: int, end: int) -> pd.DataFrame:
        key = (start, end)
        if key in self._df_cache:
            print(f"  [cache] 使用已快取 df：({start}, {end})")
            return self._df_cache[key]

        cfg = self.cfg
        preprocess_config = PreprocessConfig(
            start_year       = start,
            end_year         = end,
            raw_data_dir     = cfg.raw_data_dir,
            market_data_path = cfg.market_data_path,
            beta_window      = cfg.beta_window,
            return_clip      = cfg.return_clip,
            tick_threshold   = cfg.tick_threshold,
            tick0_threshold  = cfg.tick0_threshold,
        )
        df = DataPreprocessor(preprocess_config).run()
        self._df_cache[key] = df
        return df

    # ────────────────────────────────────────────────────────
    #  策略評分（依 config 選擇）
    # ────────────────────────────────────────────────────────

    def _compute_score(self, df: pd.DataFrame) -> pd.Series:
        strategy = self.cfg.strategy
        if strategy == "momentum":
            return SignalScorer.momentum_score(df)
        elif strategy == "mean_reversion":
            return SignalScorer.mean_reversion_score(df)
        elif strategy == "multifactor":
            return SignalScorer.multifactor_score(df)
        elif strategy == "composite":
            return SignalScorer.composite_score(df, self.cfg.composite_weights)
        else:
            raise ValueError(f"未知策略: {strategy}")

    # ────────────────────────────────────────────────────────
    #  閾值自適應（in-sample）
    # ────────────────────────────────────────────────────────

    def _adaptive_threshold(
        self,
        in_sample_scores: pd.Series,
    ) -> float:
        """
        以 in-sample 評分的 (1 - top_pct) 分位數作為訊號閾值。
        例：top_pct=0.20 → 取前 20% → 閾值 = 80th percentile。
        """
        threshold = float(np.nanpercentile(in_sample_scores, 100 * (1 - self.cfg.top_pct)))
        print(f"  自適應閾值: {threshold:.4f} (top {self.cfg.top_pct:.0%})")
        return threshold

    # ────────────────────────────────────────────────────────
    #  單一 Window
    # ────────────────────────────────────────────────────────

    def _run_window(self, start: int, end: int) -> pd.DataFrame | None:
        """
        執行單一 walk-forward window，回傳 predictions DataFrame。

        Parameters
        ----------
        start : int   window 起始年
        end   : int   window 結束年（= start + 2）

        Returns
        -------
        pd.DataFrame 含欄位：證券代碼, 年月日, return, y_true, y_prob, y_pred
        """
        test_year = end   # 第三年為 test

        print(f"\n  [Window] {start}~{end-1} (in-sample) / {test_year} (test)")

        df = self._get_df(start, end)
        df = df.dropna(subset=[self.cfg.target_col, "return"])

        # ── 分割 in-sample / test ──────────────────────────
        df["_year"] = pd.to_datetime(df["年月日"]).dt.year
        in_sample = df[df["_year"] < test_year].copy()
        test_df   = df[df["_year"] == test_year].copy()

        if len(test_df) == 0:
            print(f"  ⚠ test 年 {test_year} 無資料，跳過")
            return None

        print(f"  in-sample: {len(in_sample):,} 筆 | test: {len(test_df):,} 筆")

        # ── 計算評分 ───────────────────────────────────────
        in_scores   = self._compute_score(in_sample)
        test_scores = self._compute_score(test_df)

        # ── 閾值決定 ───────────────────────────────────────
        if self.cfg.signal_threshold is not None:
            threshold = self.cfg.signal_threshold
            print(f"  固定閾值: {threshold:.4f}")
        else:
            threshold = self._adaptive_threshold(in_scores)

        # ── 建立輸出 DataFrame ─────────────────────────────
        result = pd.DataFrame({
            "證券代碼": test_df["證券代碼"].values,
            "年月日":   test_df["年月日"].dt.strftime("%Y%m%d").values,
            "return":  test_df["return"].values,
            "y_true":  test_df[self.cfg.target_col].values,
            "y_prob":  test_scores.values,
            "y_pred":  (test_scores >= threshold).astype(int).values,
        })

        # ── 統計輸出 ───────────────────────────────────────
        n_buy   = (result["y_pred"] == 1).sum()
        n_total = len(result)
        pos_rate = (result["y_true"] == 1).mean()
        buy_precision = (
            result.loc[result["y_pred"] == 1, "y_true"].mean()
            if n_buy > 0 else float("nan")
        )
        print(f"  買入訊號: {n_buy:,}/{n_total:,} ({n_buy/n_total:.1%})")
        print(f"  y_true 正樣本率: {pos_rate:.2%}")
        print(f"  訊號精確率 (Precision): {buy_precision:.2%}" if not np.isnan(buy_precision) else "  訊號精確率: N/A")

        return result

    # ────────────────────────────────────────────────────────
    #  公開介面
    # ────────────────────────────────────────────────────────

    def run(self):
        """執行所有 walk-forward windows，輸出 predictions.csv 至各 window 資料夾。"""
        cfg     = self.cfg
        windows = cfg.windows()

        print(f"\n{'='*60}")
        print(f"  Strategy: {cfg.strategy.upper()}")
        print(f"  Walk-forward Windows: {windows[0][0]}~{windows[-1][1]}")
        print(f"  signal_threshold: {'auto (top ' + f'{cfg.top_pct:.0%})' if cfg.signal_threshold is None else cfg.signal_threshold}")
        print(f"  目標欄位: {cfg.target_col}")
        print(f"{'='*60}")

        all_results = []
        metrics_summary = []

        for start, end in windows:
            result = self._run_window(start, end)
            if result is None:
                continue

            # ── 儲存 predictions.csv ───────────────────────
            window_dir = cfg.output_dir / f"{start}_{end}"
            window_dir.mkdir(parents=True, exist_ok=True)
            out_path = window_dir / "predictions.csv"
            result.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  ✓ 輸出: {out_path}")

            # ── 儲存 metrics.json（對齊 model.py 格式）──────
            n_buy = (result["y_pred"] == 1).sum()
            n_total = len(result)
            y_true = result["y_true"].dropna()
            y_pred = result.loc[y_true.index, "y_pred"]
            y_prob = result.loc[y_true.index, "y_prob"]

            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score,
                recall_score, roc_auc_score,
            )

            # ── 交易績效指標 ────────────────────────────────
            long_returns = result.loc[result["y_pred"] == 1, "return"]

            # 勝率 Win Rate：買入訊號中報酬 > 0 的比例
            win_rate = float((long_returns > 0).mean()) if n_buy > 0 else float("nan")

            # 平均報酬 Avg Return：買入訊號的平均單筆報酬
            avg_return = float(long_returns.mean()) if n_buy > 0 else float("nan")

            # Sharpe Ratio（年化，無風險利率 = 0）
            # 以每日截面等權 long portfolio 日報酬序列計算
            daily_long_ret = (
                result.loc[result["y_pred"] == 1]
                .groupby("年月日")["return"]
                .mean()
            )
            if len(daily_long_ret) > 1 and daily_long_ret.std() > 0:
                sharpe = float(daily_long_ret.mean() / daily_long_ret.std() * np.sqrt(252))
            else:
                sharpe = float("nan")

            # 最大虧損 Max Drawdown：daily long portfolio 累積報酬最大回撤
            if len(daily_long_ret) > 1:
                cum_ret  = (1 + daily_long_ret).cumprod()
                roll_max = cum_ret.cummax()
                max_dd   = float(((cum_ret - roll_max) / roll_max).min())
            else:
                max_dd = float("nan")

            metrics = {
                "strategy":     cfg.strategy,
                "window":       f"{start}_{end}",
                "test_year":    end,
                "n_signals":    int(n_buy),
                "n_total":      int(n_total),
                "signal_rate":  float(n_buy / n_total) if n_total > 0 else 0.0,
                "threshold":    float(cfg.signal_threshold or self.cfg.top_pct),
                # ── 交易績效 ──
                "win_rate":     win_rate,
                "avg_return":   avg_return,
                "sharpe":       sharpe,
                "max_drawdown": max_dd,
                "ls_spread":    float(long_returns.mean() - result.loc[result["y_pred"] == 0, "return"].mean()) if n_buy > 0 else 0.0,
                # ── 分類指標 ──
                "accuracy":     float(accuracy_score(y_true, y_pred)),
                "precision":    float(precision_score(y_true, y_pred, zero_division=0)),
                "recall":       float(recall_score(y_true, y_pred, zero_division=0)),
                "f1":           float(f1_score(y_true, y_pred, zero_division=0)),
                "roc_auc":      float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else float("nan"),
                "pos_rate":     float(y_true.mean()),
            }

            metrics_path = window_dir / "metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4, ensure_ascii=False)
            print(f"  ✓ 指標: {metrics_path}")

            all_results.append(result)
            metrics_summary.append(metrics)

        # ── 彙總輸出 ───────────────────────────────────────
        if metrics_summary:
            self._print_summary(metrics_summary)

    def _print_summary(self, metrics_summary: list[dict]):
        W = 78
        print(f"\n{'='*W}")
        print(f"  Walk-forward 彙總 ({self.cfg.strategy})")
        print(f"{'─'*W}")
        header = (
            f"  {'Window':<12} {'勝率':>8} {'平均報酬':>10} "
            f"{'Sharpe':>8} {'MaxDD':>9} {'LS Spread':>10} {'訊號率':>7}"
        )
        print(header)
        print(f"{'─'*W}")

        def _fmt(v, fmt=".4f"):
            return f"{v:{fmt}}" if (v is not None and not np.isnan(v)) else "  N/A  "

        for m in metrics_summary:
            print(
                f"  {m['window']:<12} "
                f"{_fmt(m['win_rate'], '.2%'):>8} "
                f"{_fmt(m['avg_return'], '.4f'):>10} "
                f"{_fmt(m['sharpe'], '.3f'):>8} "
                f"{_fmt(m['max_drawdown'], '.4f'):>9} "
                f"{_fmt(m['ls_spread'], '.4f'):>10} "
                f"{m['signal_rate']:>7.1%}"
            )

        print(f"{'─'*W}")
        avgs = {
            "win_rate":    np.nanmean([m["win_rate"]    for m in metrics_summary]),
            "avg_return":  np.nanmean([m["avg_return"]  for m in metrics_summary]),
            "sharpe":      np.nanmean([m["sharpe"]      for m in metrics_summary]),
            "max_drawdown":np.nanmean([m["max_drawdown"]for m in metrics_summary]),
            "ls_spread":   np.nanmean([m["ls_spread"]   for m in metrics_summary]),
            "signal_rate": np.nanmean([m["signal_rate"] for m in metrics_summary]),
        }
        print(
            f"  {'平均':<12} "
            f"{_fmt(avgs['win_rate'], '.2%'):>8} "
            f"{_fmt(avgs['avg_return'], '.4f'):>10} "
            f"{_fmt(avgs['sharpe'], '.3f'):>8} "
            f"{_fmt(avgs['max_drawdown'], '.4f'):>9} "
            f"{_fmt(avgs['ls_spread'], '.4f'):>10} "
            f"{avgs['signal_rate']:>7.1%}"
        )
        print(f"{'='*W}")


# ============================================================
#  直接執行
# ============================================================

if __name__ == "__main__":
    import time

    cfg = StrategyConfig(
        window_start = 2014,
        window_end   = 2025,
        strategy     = "multifactor",   # "momentum" | "mean_reversion" | "multifactor" | "composite"
        top_pct      = 0.20,            # 取前 20% 評分為買入訊號
        target_col   = "excess_return_tick",
    )

    t0 = time.time()
    backtester = WalkForwardBacktester(cfg)
    backtester.run()
    print(f"\n  總耗時: {time.time() - t0:.1f} 秒")




#     ► Return Labels
#   market_return_fwd dropped（look-ahead bias 防護）
#   return_tick        (>1.0%) 正樣本率: 23.13%
#   return_tick_0      (>0.0%) 正樣本率: 44.11%
#   excess_return_tick (>1.0%) 正樣本率: 22.63%

# ============================================================
#   ✓ 完成
#   形狀          : (693801, 680)
#   股票數量      : 958
#   日期範圍      : 2023-01-03 ~ 2025-12-31
#   技術指標欄位數: 57
#   Look-ahead 防護: ✓ market_return_fwd 已 drop
# ============================================================

#   in-sample: 458,514 筆 | test: 233,371 筆
#   自適應閾值: 0.7979 (top 20%)
#   買入訊號: 47,246/233,371 (20.2%)
#   y_true 正樣本率: 23.77%
#   訊號精確率 (Precision): 23.60%
#   ✓ 輸出: database\condition_trade\2023_2025\predictions.csv
#   ✓ 指標: database\condition_trade\2023_2025\metrics.json

# ==============================================================================
#   Walk-forward 彙總 (multifactor)
# ──────────────────────────────────────────────────────────────────────────────
#   Window             勝率       平均報酬   Sharpe     MaxDD  LS Spread     訊號率
# ──────────────────────────────────────────────────────────────────────────────
#   2023_2025      44.44%     0.0003    0.537   -0.1600     0.0003   20.2%
# ──────────────────────────────────────────────────────────────────────────────
#   平均             44.44%     0.0003    0.537   -0.1600     0.0003   20.2%
# ==============================================================================