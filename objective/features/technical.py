"""
technical.py
============
傳統技術指標特徵：RSI / 均線 / MACD / KD / Williams %R / ROC / Momentum /
ATR / 布林通道 / OBV / VWAP / 成交量比 / CCI。
"""

from __future__ import annotations

import pandas as pd

from .. import ta_indicators as ta
from ._base import _FeatureHelperMixin, _CLOSE, _HIGH, _LOW, _VOL


class TechnicalFeaturesMixin(_FeatureHelperMixin):
    """所有傳統技術指標 _add_* 方法。繼承類別需提供：self.df, self.config"""

    # ──────────────────────────────────────────────────────────
    #  RSI
    # ──────────────────────────────────────────────────────────

    def _add_rsi_features(self):
        if not self._require_cols(_CLOSE):
            return False
        for w in [5, 10, 20, 60, 120, 240]:
            self.df[f"RSI_{w}"] = (
                self.df.groupby("證券代碼")[_CLOSE]
                .transform(lambda x, _w=w: ta.rsi(x, _w))
                .fillna(-1)
            )

    # ──────────────────────────────────────────────────────────
    #  Moving Averages（SMA / EMA / Price-to-MA Ratio）
    # ──────────────────────────────────────────────────────────

    def _add_moving_average_features(self):
        if not self._require_cols(_CLOSE):
            return False
        for w in [5, 10, 20, 60, 120, 240]:
            sma_raw = self.df.groupby("證券代碼")[_CLOSE].transform(lambda x, _w=w: ta.sma(x, _w))
            ema_raw = self.df.groupby("證券代碼")[_CLOSE].transform(lambda x, _w=w: ta.ema(x, _w))
            # 暖機期（資料不足 window 日）以 adj_close 填補：
            #   SMA = close  →  Price_SMA_Ratio = 0（中性，優於填 0 造成極端比値）
            sma_ = sma_raw.fillna(self.df[_CLOSE])
            ema_ = ema_raw.fillna(self.df[_CLOSE])
            self.df[f"SMA_{w}"]            = sma_
            self.df[f"EMA_{w}"]            = ema_
            self.df[f"Price_SMA{w}_Ratio"] = ((self.df[_CLOSE] / (sma_ + 1e-9)) - 1) * 100
            self.df[f"Price_SMA{w}_Ratio"] = self.df[f"Price_SMA{w}_Ratio"].fillna(0)

    # ──────────────────────────────────────────────────────────
    #  MACD
    # ──────────────────────────────────────────────────────────

    def _add_macd_features(self):
        if not self._require_cols(_CLOSE):
            return False

        def _calc(g):
            line, sig, hist = ta.macd(g)
            return pd.DataFrame({"MACD": line, "MACD_Signal": sig, "MACD_Hist": hist}, index=g.index)

        result = (
            self.df.groupby("證券代碼")[_CLOSE]
            .apply(_calc)
            .reset_index(level=0, drop=True)
        )
        for col in ["MACD", "MACD_Signal", "MACD_Hist"]:
            self.df[col] = result[col].fillna(0)

    # ──────────────────────────────────────────────────────────
    #  KD
    # ──────────────────────────────────────────────────────────

    def _add_kd_features(self):
        if not self._require_cols(_HIGH, _LOW, _CLOSE):
            return False

        def _calc(g):
            k, d = ta.kd(g[_HIGH], g[_LOW], g[_CLOSE])
            return pd.DataFrame({"K": k, "D": d}, index=g.index)

        result = (
            self.df.groupby("證券代碼")
            .apply(_calc, include_groups=False)
            .reset_index(level=0, drop=True)
        )
        self.df["K"]       = result["K"].fillna(50)
        self.df["D"]       = result["D"].fillna(50)
        self.df["KD_Diff"] = self.df["K"] - self.df["D"]

    # ──────────────────────────────────────────────────────────
    #  Williams %R
    # ──────────────────────────────────────────────────────────

    def _add_williams_r_features(self):
        if not self._require_cols(_HIGH, _LOW, _CLOSE):
            return False
        for w in [14, 28]:
            self.df[f"Williams_R_{w}"] = (
                self.df.groupby("證券代碼")
                .apply(lambda g, _w=w: ta.williams_r(g[_HIGH], g[_LOW], g[_CLOSE], _w),
                       include_groups=False)
                .reset_index(level=0, drop=True)
                .fillna(-50)
            )

    # ──────────────────────────────────────────────────────────
    #  ROC / Momentum
    # ──────────────────────────────────────────────────────────

    def _add_roc_momentum_features(self):
        if not self._require_cols(_CLOSE):
            return False
        for w in [5, 10, 20]:
            self.df[f"ROC_{w}"] = (
                self.df.groupby("證券代碼")[_CLOSE]
                .transform(lambda x, _w=w: ta.roc(x, _w))
                .fillna(0)
            )
            self.df[f"Momentum_{w}"] = (
                self.df.groupby("證券代碼")[_CLOSE]
                .transform(lambda x, _w=w: ta.momentum(x, _w))
                .fillna(0)
            )

    # ──────────────────────────────────────────────────────────
    #  ATR / Bollinger Bands
    # ──────────────────────────────────────────────────────────

    def _add_volatility_features(self):
        if not self._require_cols(_HIGH, _LOW, _CLOSE):
            return False
        for w in [14, 28]:
            atr_ = (
                self.df.groupby("證券代碼")
                .apply(lambda g, _w=w: ta.atr(g[_HIGH], g[_LOW], g[_CLOSE], _w),
                       include_groups=False)
                .reset_index(level=0, drop=True)
            )
            self.df[f"ATR_{w}"]     = atr_.fillna(0)
            self.df[f"ATR_{w}_Pct"] = (atr_ / (self.df[_CLOSE] + 1e-9) * 100).fillna(0)

        def _calc_bb(g):
            upper, mid, lower, pct_b = ta.bollinger(g, 20)
            return pd.DataFrame(
                {"BB_Upper_20": upper, "BB_Middle_20": mid,
                 "BB_Lower_20": lower, "BB_PctB_20": pct_b},
                index=g.index,
            )

        bb = (
            self.df.groupby("證券代碼")[_CLOSE]
            .apply(_calc_bb)
            .reset_index(level=0, drop=True)
        )
        for col in bb.columns:
            self.df[col] = bb[col].fillna(0)

    # ──────────────────────────────────────────────────────────
    #  OBV / VWAP / Volume Ratio
    # ──────────────────────────────────────────────────────────

    def _add_volume_features(self):
        if not self._require_cols(_HIGH, _LOW, _CLOSE, _VOL):
            return False

        self.df["OBV"] = (
            self.df.groupby("證券代碼")
            .apply(lambda g: ta.obv(g[_CLOSE], g[_VOL]), include_groups=False)
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        self.df["OBV_ROC"] = (
            self.df.groupby("證券代碼")["OBV"]
            .transform(lambda x: x.pct_change(5, fill_method=None) * 100)
            .fillna(0)
        )
        self.df["VWAP"] = (
            self.df.groupby("證券代碼")
            .apply(lambda g: ta.vwap(g[_HIGH], g[_LOW], g[_CLOSE], g[_VOL]),
                   include_groups=False)
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        self.df["Price_VWAP_Ratio"] = (self.df[_CLOSE] / (self.df["VWAP"] + 1e-9) - 1) * 100

        for w in [5, 20]:
            sma_v = self.df.groupby("證券代碼")[_VOL].transform(
                lambda x, _w=w: x.rolling(_w, min_periods=1).mean()
            )
            self.df[f"Volume_SMA_{w}"]   = sma_v.fillna(0)
            self.df[f"Volume_Ratio_{w}"] = (self.df[_VOL] / (sma_v + 1e-9)).fillna(0)

    # ──────────────────────────────────────────────────────────
    #  CCI
    # ──────────────────────────────────────────────────────────

    def _add_cci_features(self):
        if not self._require_cols(_HIGH, _LOW, _CLOSE):
            return False
        for w in [14, 20]:
            self.df[f"CCI_{w}"] = (
                self.df.groupby("證券代碼")
                .apply(lambda g, _w=w: ta.cci(g[_HIGH], g[_LOW], g[_CLOSE], _w),
                       include_groups=False)
                .reset_index(level=0, drop=True)
                .fillna(0)
            )
