"""
ta_indicators.py
================
技術指標純函數庫（無狀態）。

所有函式皆為 module-level pure functions，接受 pd.Series 輸入、
回傳 pd.Series（或 tuple），供 FeatureMixin 的 _add_* 方法呼叫。

移出 DataPreprocessor @staticmethod 的原因：
  - 可獨立測試（pytest 直接 import）
  - 不需要 class 實例，memory footprint 更低
  - 可供其他模組（如 strategy.py）直接使用

包含：
  rsi, sma, ema, macd, kd, williams_r, roc, momentum,
  atr, bollinger, obv, vwap, cci, signed_log1p

作者：Daniel Huang
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
#  Trend Indicators
# ============================================================

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average。"""
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average（span 參數）。"""
    return series.ewm(span=window, adjust=False).mean()


def macd(
    series: pd.Series,
    fast:   int = 12,
    slow:   int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD 三線：(macd_line, signal_line, histogram)。
    macd_line = EMA(fast) - EMA(slow)
    signal    = EMA(macd_line, signal)
    histogram = macd_line - signal
    """
    line = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    sig  = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


# ============================================================
#  Oscillators
# ============================================================

def rsi(series: pd.Series, window: int) -> pd.Series:
    """
    Wilder RSI（用 EWM 模擬，alpha = 1/window）。
    前 window 期設為 NaN（warmup）。
    """
    series = pd.to_numeric(series, errors="coerce")
    delta  = series.diff()
    alpha  = 1.0 / window
    gain   = delta.clip(lower=0).ewm(alpha=alpha, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(alpha=alpha, adjust=False).mean()
    r      = 100 - 100 / (1 + gain / (loss + 1e-9))
    r.iloc[:window] = np.nan
    return r


def kd(
    high:     pd.Series,
    low:      pd.Series,
    close:    pd.Series,
    k_window: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """
    KD 指標（K 值、D 值）。
    使用 2/3 EWM 平滑，初始 KP=DP=50。
    """
    ll  = low.rolling(k_window, min_periods=k_window).min()
    hh  = high.rolling(k_window, min_periods=k_window).max()
    rsv = 100 * (close - ll) / (hh - ll + 1e-9)

    k_vals, d_vals = [], []
    kp, dp = 50.0, 50.0
    for v in rsv:
        if pd.isna(v):
            k_vals.append(np.nan)
            d_vals.append(np.nan)
        else:
            kp = 2 / 3 * kp + 1 / 3 * v
            dp = 2 / 3 * dp + 1 / 3 * kp
            k_vals.append(kp)
            d_vals.append(dp)

    return (
        pd.Series(k_vals, index=close.index),
        pd.Series(d_vals, index=close.index),
    )


def williams_r(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    window: int = 14,
) -> pd.Series:
    """Williams %R，範圍 [-100, 0]。"""
    hh = high.rolling(window, min_periods=window).max()
    ll = low.rolling(window, min_periods=window).min()
    return -100 * (hh - close) / (hh - ll + 1e-9)


def cci(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    window: int = 20,
) -> pd.Series:
    """Commodity Channel Index。"""
    tp   = (high + low + close) / 3
    sma_ = tp.rolling(window, min_periods=window).mean()
    mad  = tp.rolling(window, min_periods=window).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (tp - sma_) / (0.015 * mad + 1e-9)


# ============================================================
#  Momentum Indicators
# ============================================================

def roc(series: pd.Series, window: int) -> pd.Series:
    """Rate of Change（百分比）。"""
    return 100 * (series - series.shift(window)) / (series.shift(window) + 1e-9)


def momentum(series: pd.Series, window: int) -> pd.Series:
    """Price Momentum（絕對差）。"""
    return series - series.shift(window)


# ============================================================
#  Volatility Indicators
# ============================================================

def atr(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    window: int = 14,
) -> pd.Series:
    """Average True Range（EWM 平滑）。"""
    pc = close.shift(1)
    tr = pd.concat(
        [high - low, (high - pc).abs(), (low - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=window, adjust=False).mean()


def bollinger(
    series:  pd.Series,
    window:  int   = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands。
    Returns: (upper, middle, lower, pct_b)
    pct_b = (price - lower) / (upper - lower)
    """
    mid   = series.rolling(window, min_periods=window).mean()
    std   = series.rolling(window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
    return upper, mid, lower, pct_b


# ============================================================
#  Volume Indicators
# ============================================================

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume。"""
    vals, prev = [0], 0
    for i in range(1, len(close)):
        if   close.iloc[i] > close.iloc[i - 1]: prev += volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]: prev -= volume.iloc[i]
        vals.append(prev)
    return pd.Series(vals, index=close.index)


def vwap(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Cumulative VWAP（全期間累積）。"""
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / (volume.cumsum() + 1e-9)


# ============================================================
#  Utility
# ============================================================

def consecutive_cumsum(s: pd.Series) -> pd.Series:
    """
    符號轉換重置累計（連續買賣超）。

    規則：
      - 正負號改變時，cumsum 歸零重算（從當日值開始）。
      - 當日值為 0 時，視為延續前一方向（不觸發重置）。
      - NaN 保留為 NaN，不影響後續計算。

    範例：
        輸入：[500, 300, -200, -400, 100]
        輸出：[500, 800, -200, -600, 100]
    """
    result = np.empty(len(s), dtype=float)
    values = s.to_numpy(dtype=float)
    running   = 0.0
    prev_sign = 0
    for i, v in enumerate(values):
        if np.isnan(v):
            result[i] = np.nan
            continue
        curr_sign = int(np.sign(v)) if v != 0.0 else prev_sign
        if curr_sign != prev_sign and prev_sign != 0:
            running = v          # 符號翻轉 → 重置
        else:
            running += v
        prev_sign = curr_sign
        result[i] = running
    return pd.Series(result, index=s.index)


def signed_log1p(s: pd.Series) -> pd.Series:
    """
    保號 log1p 轉換：sign(x) × log1p(|x|)。
    適用於含負值的流量欄位（如借券庫存異動）。
    """
    return np.sign(s) * np.log1p(np.abs(s))