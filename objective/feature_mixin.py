"""
feature_mixin.py
================
FeatureMixin：所有特徵工程方法（_add_*）。

新專案版本變更：
    - 欄位名稱從 TEJ 中文 → FinLab 標準內部名稱
        收盤價元  → adj_close    （技術指標使用還原後收盤價）
        最高價元  → high
        最低價元  → low
        開盤價元  → adj_open     （還原後開盤，報酬率標籤用）
        成交量千股→ volume        （股，非千股）
        成交金額  → amount
    - 新增方法：
        _add_price_derived_features      高低價差 / 實際週轉率 / 現股成交比重
        _add_return_momentum_extended    多期間報酬率 / YTD / MTD / QTD
        _add_capm_beta_extended          CAPM Beta 1m / 9m / 1y
        _add_institutional_flow_features 三大法人 / 融資融券衍生特徵
        _add_fundamental_features        月營收 / EPS 衍生特徵
    - _add_short_selling_features 改用 FinLab security_lending 欄位

作者：Daniel Huang
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ta_indicators as ta

# ─────────────────────────────────────────────────────────────
#  欄位常數（FinLab 內部標準名）
# ─────────────────────────────────────────────────────────────
_CLOSE  = "adj_close"    # 還原後收盤（TA 指標）
_RCLOSE = "close"        # 未還原收盤（比率類特徵）
_HIGH   = "high"
_LOW    = "low"
_OPEN   = "adj_open"     # adj_open（報酬率標籤）
_VOL    = "volume"       # 成交股數（股）
_AMT    = "amount"       # 成交金額


class FeatureMixin:
    """
    所有 _add_* 特徵工程方法的 Mixin。
    繼承類別需提供：self.df, self.config
    """

    # ──────────────────────────────────────────────────────────
    #  輔助
    # ──────────────────────────────────────────────────────────

    def _require_cols(self, *cols: str) -> bool:
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            print(f"  ✗ 缺少欄位: {missing}")
            return False
        return True

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

    # ──────────────────────────────────────────────────────────
    #  【新增】價格衍生特徵
    #  高低價差 / 實際週轉率 / 一般現股成交比重
    # ──────────────────────────────────────────────────────────

    def _add_price_derived_features(self):
        """
        1. 高低價差         = (high - low) / close
        2. 實際週轉率       = volume / (fii_shares - insider_shares) × 100%
        3. 一般現股成交比重  = (volume - intraday_shares) / volume
        """
        # ① 高低價差
        if self._require_cols(_HIGH, _LOW, _RCLOSE):
            self.df["高低價差"] = (
                (self.df[_HIGH] - self.df[_LOW]) / (self.df[_RCLOSE] + 1e-9)
            ).clip(0, 1)
            print("    高低價差 ✓")

        # ② 實際週轉率（董監持股月頻，已在 ffill 後對齊日頻）
        if self._require_cols(_VOL, "fii_shares", "insider_shares"):
            float_shares = (
                self.df["fii_shares"] - self.df["insider_shares"]
            ).clip(lower=1)
            self.df["實際週轉率"] = (
                self.df[_VOL] / float_shares * 100
            ).clip(0, 100).fillna(0)
            print("    實際週轉率 ✓")

        # ③ 一般現股成交比重
        if self._require_cols(_VOL, "intraday_shares"):
            self.df["一般現股成交比重"] = (
                (self.df[_VOL] - self.df["intraday_shares"])
                / (self.df[_VOL] + 1e-9)
            ).clip(0, 1).fillna(0)
            print("    一般現股成交比重 ✓")

        # ④ 當日沖銷佔市場比重（直接取 intraday_trading_stat）
        if "intraday_sell_pct" in self.df.columns:
            self.df["當日沖銷交易總賣出成交金額占市場比重"] = (
                self.df["intraday_sell_pct"].fillna(0)
            )
            print("    當日沖銷交易總賣出成交金額占市場比重 ✓")

        # ⑤ 股價漲跌元（日漲跌金額，未還原收盤）
        if _RCLOSE in self.df.columns:
            self.df["股價漲跌元"] = (
                self.df.groupby("證券代碼")[_RCLOSE]
                .transform(lambda x: x.diff())
                .fillna(0)
            )
            print("    股價漲跌元 ✓")

    # ──────────────────────────────────────────────────────────
    #  【新增】多期間報酬率 / YTD / MTD / QTD
    # ──────────────────────────────────────────────────────────

    def _add_return_momentum_extended(self):
        """
        報酬率Ln      = log(adj_close_t / adj_close_{t-1})
        近一月報酬率  = close_t / close_{t-21} - 1
        近一季報酬率  = close_t / close_{t-63} - 1
        近一年報酬率  = close_t / close_{t-252} - 1
        本月以來報酬率 = close / MTD起始close - 1
        本週以來報酬率 = close / WTD起始close - 1
        本年以來報酬率 = close / YTD起始close - 1
        本季以來報酬率 = close / QTD起始close - 1
        近3月個股報酬率= close_t / close_{t-63} - 1 (同 近一季報酬率)
        """
        if not self._require_cols(_CLOSE):
            return False

        c = _CLOSE

        # Log return
        self.df["報酬率Ln"] = (
            self.df.groupby("證券代碼")[c]
            .transform(lambda x: np.log(x / x.shift(1) + 1e-9))
            .fillna(0)
        )

        # Rolling period returns
        periods = {
            "近一月報酬率": 21,
            "近一季報酬率": 63,
            "近一年報酬率": 252,
        }
        for name, lag in periods.items():
            self.df[name] = (
                self.df.groupby("證券代碼")[c]
                .transform(lambda x, l=lag: (x / x.shift(l) - 1).clip(-0.99, 10))
                .fillna(0)
            )

        # 個股月報酬率（與 monthly_revenue 對齊的版本）
        self.df["個股月報酬率％_rev"] = self.df["近一月報酬率"] * 100

        # 本月以來報酬率（MTD）
        def _mtd(group):
            dt    = group["年月日"]
            ym    = dt.dt.to_period("M")
            first = group.groupby(ym)[c].transform("first")
            return group[c] / (first + 1e-9) - 1

        self.df["本月以來報酬率"] = (
            self.df.groupby("證券代碼", group_keys=False)
            .apply(_mtd, include_groups=False)
            .fillna(0)
        )

        # 本週以來報酬率（WTD）
        def _wtd(group):
            dt    = group["年月日"]
            yw    = dt.dt.isocalendar().year * 100 + dt.dt.isocalendar().week
            yw    = yw.values
            first = pd.Series(yw).map(
                dict(zip(*np.unique(yw, return_index=True)))
            )  # index of first occurrence per week
            # 使用 groupby week
            week_key = pd.to_datetime(group["年月日"]).dt.to_period("W")
            first_c  = group.groupby(week_key)[c].transform("first")
            return group[c] / (first_c + 1e-9) - 1

        self.df["本週以來報酬率"] = (
            self.df.groupby("證券代碼", group_keys=False)
            .apply(_wtd, include_groups=False)
            .fillna(0)
        )

        # 本年以來報酬率（YTD）
        def _ytd(group):
            year  = group["年月日"].dt.year
            first = group.groupby(year)[c].transform("first")
            return group[c] / (first + 1e-9) - 1

        self.df["本年以來報酬率"] = (
            self.df.groupby("證券代碼", group_keys=False)
            .apply(_ytd, include_groups=False)
            .fillna(0)
        )

        # 本季以來報酬率（QTD）
        def _qtd(group):
            q     = group["年月日"].dt.to_period("Q")
            first = group.groupby(q)[c].transform("first")
            return group[c] / (first + 1e-9) - 1

        self.df["本季以來報酬率"] = (
            self.df.groupby("證券代碼", group_keys=False)
            .apply(_qtd, include_groups=False)
            .fillna(0)
        )

        print("    多期間報酬率: 報酬率Ln / 近一月/季/年 / YTD / MTD / WTD / QTD ✓")

    # ──────────────────────────────────────────────────────────
    #  【新增】CAPM Beta 多視窗
    #  需在 _add_beta_vol_features 後呼叫（依賴 daily_return）
    # ──────────────────────────────────────────────────────────

    def _add_capm_beta_extended(self):
        """
        CAPM_Beta一月  rolling(21)  OLS beta
        CAPM_Beta九月  rolling(189) OLS beta
        CAPM_Beta一年  rolling(252) OLS beta
        """
        if not self._require_cols("daily_return", "market_return"):
            return False

        def _rolling_beta(group: pd.DataFrame, window: int, min_p: int) -> pd.Series:
            r   = group["daily_return"]
            mkt = group["market_return"]
            cov = r.rolling(window, min_periods=min_p).cov(mkt)
            var = mkt.rolling(window, min_periods=min_p).var()
            return (cov / (var + 1e-9)).clip(-5, 5).fillna(1.0)

        windows = {
            "CAPM_Beta一月": (21,  10),
            "CAPM_Beta九月": (189, 60),
            "CAPM_Beta一年": (252, 90),
        }
        for col_name, (w, mp) in windows.items():
            beta = (
                self.df.groupby("證券代碼", group_keys=False)
                .apply(lambda g, _w=w, _mp=mp: _rolling_beta(g, _w, _mp),
                       include_groups=False)
            )
            self.df[col_name] = beta.values
        print("    CAPM Beta: 一月 / 九月 / 一年 ✓")

    # ──────────────────────────────────────────────────────────
    #  【新增】三大法人 / 融資融券衍生特徵
    # ──────────────────────────────────────────────────────────

    def _add_institutional_flow_features(self):
        """
        合計買賣超金額千元      = (fii_net + dealer_net + trust_net) × close / 1000
        投信買賣超市值百萬       = trust_net × close / 1,000,000
        融資買賣成交量           = margin_buy + margin_sell
        融資買進千元             = margin_buy × close × 1000 / 1000
        整戶維持率（近似）       = (融資擔保市值 + 融券擔保) / (原融資金額 + 融券市值) × 100
                                   融資成數 60%；融券保證金+擔保價款 190%（以現價近似成本）
        融資增減千元             = (margin_balance - margin_balance.shift(1)) × close / 1000
        尚可投資比率TSE          = fii_investable
        董監持股數               = insider_shares
        外資本年以來買賣超千股   = YTD cumulative fii_net / 1000
        自營本週以來買賣超千股   = WTD cumulative dealer_net / 1000
        自營本月以來買賣超千股   = MTD cumulative dealer_net / 1000
        自營連續累計買賣超千     = cumulative dealer_net / 1000
        信用交易比重             = (margin_buy + margin_sell + ms_balance) / volume
        """
        df = self.df

        # ① 合計買賣超金額千元（三大法人）
        if all(c in df.columns for c in ["fii_net", "dealer_net", "trust_net", _RCLOSE]):
            total_net = df["fii_net"] + df["dealer_net"] + df["trust_net"]
            df["合計買賣超金額千元"] = (total_net * df[_RCLOSE] / 1000).fillna(0)
            print("    合計買賣超金額千元 ✓")

        # ② 投信買賣超市值百萬
        if all(c in df.columns for c in ["trust_net", _RCLOSE]):
            df["投信買賣超市值百萬"] = (df["trust_net"] * df[_RCLOSE] / 1e6).fillna(0)
            print("    投信買賣超市值百萬 ✓")

        # ③ 融資買賣成交量（張）
        if all(c in df.columns for c in ["margin_buy", "margin_sell"]):
            df["融資買賣成交量"] = (df["margin_buy"] + df["margin_sell"]).fillna(0)
            print("    融資買賣成交量 ✓")

        # ④ 融資買進千元
        if all(c in df.columns for c in ["margin_buy", _RCLOSE]):
            df["融資買進千元"] = (df["margin_buy"] * df[_RCLOSE]).fillna(0)
            print("    融資買進千元 ✓")

        # ⑤ 融資使用率（直接取用）
        if "margin_util" in df.columns:
            df["融資使用率"] = df["margin_util"].fillna(0)
            print("    融資使用率 ✓")

        # ⑥ 融資增減千元（餘額日變動）
        if all(c in df.columns for c in ["margin_balance", _RCLOSE]):
            margin_chg = (
                df.groupby("證券代碼")["margin_balance"]
                .transform(lambda x: x.diff())
            )
            df["融資增減千元"] = (margin_chg * df[_RCLOSE] / 1000).fillna(0)
            print("    融資增減千元 ✓")

        # ⑦ 整戶維持率（近似公式）
        # = (融資擔保證券市值 + 原融券擔保價款及保證金) / (原融資金額 + 融券證券市值) × 100
        # 近似假設：融資成數 60%；融券保證金+擔保價款 = 190%（以現價代替成本價）
        if all(c in df.columns for c in ["margin_balance", "ms_balance", _RCLOSE]):
            close     = df[_RCLOSE]
            mkt_long  = df["margin_balance"].clip(lower=0) * close   # 融資擔保證券市值
            mkt_short = df["ms_balance"].clip(lower=0) * close       # 融券證券市值
            原融資    = mkt_long * 0.6                                # 融資成數 60%
            融券擔保  = mkt_short * 1.9                               # 保證金 90% + 擔保價款 100%
            df["整戶維持率"] = (
                (mkt_long + 融券擔保) / (原融資 + mkt_short + 1e-9) * 100
            ).clip(0, 500).fillna(130.0)
            print("    整戶維持率（近似：TWSE 公式）✓")

        # ⑧ 信用交易比重
        if all(c in df.columns for c in ["margin_buy", "margin_sell", "ms_balance", _VOL]):
            credit_vol = df["margin_buy"] + df["margin_sell"] + df["ms_balance"]
            df["信用交易比重"] = (credit_vol / (df[_VOL] + 1e-9)).clip(0, 1).fillna(0)
            print("    信用交易比重 ✓")

        # ⑨ 尚可投資比率TSE
        if "fii_investable" in df.columns:
            df["尚可投資比率TSE"] = df["fii_investable"].fillna(0)
            print("    尚可投資比率TSE ✓")

        # ⑩ 董監持股數（千股單位標準化）
        if "insider_shares" in df.columns:
            df["董監持股數"] = (df["insider_shares"] / 1000).fillna(0)
            print("    董監持股數 ✓")

        # ⑪ 自營累計買賣超（連續 / 本月 / 本週）
        if "dealer_net" in df.columns:
            # 連續累計（符號翻轉重置）
            df["自營連續累計買賣超千"] = (
                df.groupby("證券代碼")["dealer_net"]
                .transform(lambda x: ta.consecutive_cumsum(x) / 1000)
                .fillna(0)
            )
            # 本週以來
            week_key = df["年月日"].dt.to_period("W")
            df["dealer_net_wtd"] = df["dealer_net"].copy()
            df["自營本週以來買賣超千股"] = (
                df.groupby(["證券代碼", week_key])["dealer_net"]
                .transform("cumsum") / 1000
            ).fillna(0)
            # 本月以來
            month_key = df["年月日"].dt.to_period("M")
            df["自營本月以來買賣超千股"] = (
                df.groupby(["證券代碼", month_key])["dealer_net"]
                .transform("cumsum") / 1000
            ).fillna(0)
            print("    自營累計買賣超（連續sign-reset/週/月）✓")

        # ⑫ 外資買賣超累計（連續 sign-reset + YTD 兩版本）
        if "fii_net" in df.columns:
            # ── 連續累計（符號翻轉重置）→ 對應 finlab「外資連續累計買賣超張1」
            df["外資連續累計買賣超張1"] = (
                df.groupby("證券代碼")["fii_net"]
                .transform(lambda x: ta.consecutive_cumsum(x) / 1000)
                .fillna(0)
            )
            # ── YTD 累計（每年元旦重置）→ 另存備用
            year_key = df["年月日"].dt.year
            df["外資本年以來買賣超千股"] = (
                df.groupby(["證券代碼", year_key])["fii_net"]
                .transform("cumsum") / 1000
            ).fillna(0)
            print("    外資連續累計買賣超張1（sign-reset）& 外資本年以來買賣超千股（YTD）✓")

        # ⑬ 三大法人買超張 / 賣超張
        if all(c in df.columns for c in ["fii_net", "dealer_net", "trust_net"]):
            total = df["fii_net"] + df["dealer_net"] + df["trust_net"]
            df["三大法人買超張"]  = total.clip(lower=0).fillna(0)
            df["三大法人賣超張"]  = (-total).clip(lower=0).fillna(0)

        # ⑭ 股價淨值比
        if "pbr" in df.columns:
            df["股價淨值比TEJ"] = df["pbr"].fillna(0)

        # ⑮ 融資買進張（margin_buy 直接，單位：千股）
        if "margin_buy" in df.columns:
            df["融資買進張"] = df["margin_buy"].fillna(0)

        # ⑯ 外資賣出張數（fii_net 負值部分取絕對值）
        if "fii_net" in df.columns:
            df["外資賣出張數"] = (-df["fii_net"]).clip(lower=0).fillna(0)

        # ⑰ 投信成交比重 = |trust_net| / volume
        if all(c in df.columns for c in ["trust_net", _VOL]):
            df["投信成交比重"] = (
                df["trust_net"].abs() / (df[_VOL] + 1e-9)
            ).clip(0, 1).fillna(0)

        # ⑱ 法人買賣超日數1：近 20 日三大法人合計買超天數
        if all(c in df.columns for c in ["fii_net", "dealer_net", "trust_net"]):
            is_buy = (
                (df["fii_net"] + df["dealer_net"] + df["trust_net"]) > 0
            ).astype(int)
            df["法人買賣超日數1"] = (
                is_buy.groupby(df["證券代碼"])
                .transform(lambda x: x.rolling(20, min_periods=1).sum())
                .fillna(0)
            )

        # ⑲ 自營自行買賣超_年千股（YTD 累計，與 外資本年以來 對應）
        if "dealer_net" in df.columns:
            year_key = df["年月日"].dt.year
            df["自營自行買賣超_年千股"] = (
                df.groupby(["證券代碼", year_key])["dealer_net"]
                .transform("cumsum") / 1000
            ).fillna(0)

        # ⑳ 自營自行買賣超_月千股（MTD 累計，同 自營本月以來買賣超千股，補別名）
        if "自營本月以來買賣超千股" in df.columns:
            df["自營自行買賣超_月千股"] = df["自營本月以來買賣超千股"]
        elif "dealer_net" in df.columns:
            month_key = df["年月日"].dt.to_period("M")
            df["自營自行買賣超_月千股"] = (
                df.groupby(["證券代碼", month_key])["dealer_net"]
                .transform("cumsum") / 1000
            ).fillna(0)

        # ㉑ 合計本月買賣超金額千元（MTD 累計三大法人 × 收盤）
        if all(c in df.columns for c in ["fii_net", "dealer_net", "trust_net", _RCLOSE]):
            total_net_val = (df["fii_net"] + df["dealer_net"] + df["trust_net"]) * df[_RCLOSE]
            month_key = df["年月日"].dt.to_period("M")
            df["合計本月買賣超金額千元"] = (
                total_net_val.groupby([df["證券代碼"], month_key])
                .transform("cumsum") / 1000
            ).fillna(0)

        # ㉒ 外資本週以來買賣超千股（WTD 累計）
        if "fii_net" in df.columns:
            week_key = df["年月日"].dt.to_period("W")
            df["外資本週以來買賣超千股"] = (
                df.groupby(["證券代碼", week_key])["fii_net"]
                .transform("cumsum") / 1000
            ).fillna(0)
            print("    外資本週以來買賣超千股（WTD）✓")

        # ㉓ 自營本週買賣超金額千元（WTD dealer_net 張數 × 收盤價，累計後 /1000 → 千元）
        if all(c in df.columns for c in ["dealer_net", _RCLOSE]):
            dealer_amt = df["dealer_net"] * df[_RCLOSE]
            week_key = df["年月日"].dt.to_period("W")
            df["自營本週買賣超金額千元"] = (
                dealer_amt.groupby([df["證券代碼"], week_key])
                .transform("cumsum") / 1000
            ).fillna(0)
            print("    自營本週買賣超金額千元（WTD）✓")

        # ㉔ 合計本週買賣超金額千元（WTD 三大法人合計張數 × 收盤價，累計後 /1000 → 千元）
        if all(c in df.columns for c in ["fii_net", "dealer_net", "trust_net", _RCLOSE]):
            total_net_val_wtd = (df["fii_net"] + df["dealer_net"] + df["trust_net"]) * df[_RCLOSE]
            week_key = df["年月日"].dt.to_period("W")
            df["合計本週買賣超金額千元"] = (
                total_net_val_wtd.groupby([df["證券代碼"], week_key])
                .transform("cumsum") / 1000
            ).fillna(0)
            print("    合計本週買賣超金額千元（WTD）✓")

        self.df = df.copy()   # 整合碎片化欄位

    # ──────────────────────────────────────────────────────────
    #  【新增】月營收 / EPS 衍生特徵
    # ──────────────────────────────────────────────────────────

    def _add_fundamental_features(self):
        """
        去年累計營收(千元)_rev         直接取用 monthly_revenue_ly_cum
        單月營收成長率％_rev            直接取用 monthly_revenue_mom
        近3月累計營收變動率％_rev       rolling(3).sum() / rolling(3).sum().shift(3) - 1
        近12月累計營收成長率_rev        rolling(12).sum() / rolling(12).sum().shift(12) - 1
        與歷史最低單月營收比%_rev       monthly_revenue / expanding_min - 1
        累計稅後盈餘成長率％_rev        YTD EPS / YTD EPS.shift(252) - 1
        單月每股稅後盈餘(WA)_rev        eps 直接取用
        單月每股營收(元)_rev            monthly_revenue / fii_shares（每股）
        """
        df = self.df

        # 去年累計營收
        if "monthly_revenue_ly_cum" in df.columns:
            df["去年累計營收(千元)_rev"] = (df["monthly_revenue_ly_cum"] / 1000).fillna(0)
            print("    去年累計營收 ✓")

        # 單月營收成長率（MoM）
        if "monthly_revenue_mom" in df.columns:
            df["單月營收成長率％_rev"] = df["monthly_revenue_mom"].fillna(0)
            print("    單月營收成長率 ✓")

        # 近3月 / 12月累計營收成長率
        # ⚠ monthly_revenue 為月頻 ffill 至日頻，必須先降回月頻再做 rolling，
        #   否則 rolling(3) 會在日頻上跑，語意變成「3個交易日」而非「3個月」。
        if "monthly_revenue" in df.columns:

            # ⚠ pandas 2.x groupby.apply pivot 陷阱：
            #   若每個 group 回傳的 Series 以「共用日期索引」為 index（所有股票
            #   交易日相同），pandas 會把這些 Series 展開成寬表 DataFrame，導致
            #   後續 assign 觸發「Cannot set a DataFrame with multiple columns」。
            #   解法：全程保留原始 df 整數索引，apply 結束後直接對齊原 df。

            def _rev_growth(group: pd.DataFrame) -> pd.DataFrame:
                orig_index = group.index                       # 原始 df 整數索引
                g = group.set_index("年月日")["monthly_revenue"]
                date_idx = g.index
                # 月頻：取每月最後一個非 NaN 值
                m = g.resample("ME").last().dropna()
                if len(m) < 4:
                    out = pd.DataFrame(
                        {
                            "近3月累計營收變動率％_rev": np.nan,
                            "近12月累計營收成長率_rev":  np.nan,
                        },
                        index=orig_index,
                    )
                    return out
                # 近3月累計 vs 前3月累計
                r3     = m.rolling(3, min_periods=3).sum()
                r3_lag = r3.shift(3)
                chg3   = (r3 / (r3_lag.abs() + 1e-9) - 1).clip(-2, 10)
                # 近12月累計 vs 前12月累計
                r12     = m.rolling(12, min_periods=6).sum()
                r12_lag = r12.shift(12)
                chg12   = (r12 / (r12_lag.abs() + 1e-9) - 1).clip(-2, 10)
                # reindex 回日頻 ffill，再把索引還原成原始 df 整數索引
                out = pd.DataFrame(
                    {
                        "近3月累計營收變動率％_rev": chg3.reindex(date_idx, method="ffill").to_numpy(),
                        "近12月累計營收成長率_rev":  chg12.reindex(date_idx, method="ffill").to_numpy(),
                    },
                    index=orig_index,
                )
                return out

            rev_result = df.groupby("證券代碼", group_keys=False).apply(
                _rev_growth, include_groups=False
            )
            # rev_result 以原始 df 整數索引對齊，reindex 確保順序一致
            rev_result = rev_result.reindex(df.index)
            df["近3月累計營收變動率％_rev"] = rev_result["近3月累計營收變動率％_rev"].fillna(0)
            df["近12月累計營收成長率_rev"]  = rev_result["近12月累計營收成長率_rev"].fillna(0)

            # 與歷史最低單月營收比
            # ⚠ 需在月頻做 expanding().min()，並 shift(1) 避免分母含當月自身
            # ⚠ resample 用 "BME"（Business Month End）確保 reindex ffill 能正確對齊
            def _hist_low_ratio(group: pd.DataFrame) -> pd.Series:
                orig_index = group.index                       # 原始 df 整數索引
                g = group.set_index("年月日")["monthly_revenue"]
                date_idx = g.index
                if g.dropna().empty:
                    return pd.Series(np.nan, index=orig_index,
                                     name="與歷史最低單月營收比%_rev")
                m = g.resample("BME").last().dropna()
                if m.empty:
                    return pd.Series(np.nan, index=orig_index,
                                     name="與歷史最低單月營收比%_rev")
                hist_min = m.expanding().min().shift(1)         # 截至上個月的歷史最低
                ratio = (m / (hist_min.abs() + 1e-9) - 1).clip(0, 100)
                vals = ratio.reindex(date_idx, method="ffill").to_numpy()
                # 還原成原始 df 整數索引，避免共用日期索引被 pivot 成寬表
                return pd.Series(vals, index=orig_index,
                                 name="與歷史最低單月營收比%_rev")

            hist_low = df.groupby("證券代碼", group_keys=False).apply(
                _hist_low_ratio, include_groups=False
            )
            # 防呆：若 pandas 仍意外回傳 DataFrame，取第一欄壓平
            if isinstance(hist_low, pd.DataFrame):
                hist_low = hist_low.iloc[:, 0]
            df["與歷史最低單月營收比%_rev"] = hist_low.reindex(df.index).fillna(0)
            print("    月營收衍生特徵（3m/12m成長率月頻正確版 / 歷史最低比月頻+shift）✓")

        # 單月每股營收
        if all(c in df.columns for c in ["monthly_revenue", "fii_shares"]):
            df["單月每股營收(元)_rev"] = (
                df["monthly_revenue"] / (df["fii_shares"] + 1e-9)
            ).fillna(0)
            print("    單月每股營收 ✓")

        # EPS 衍生
        if "eps" in df.columns:
            df["單月每股稅後盈餘(WA)_rev"] = df["eps"].fillna(0)

            # 累計 EPS YoY 成長率（TTM / TTM shift 252 trading days）
            ttm = df.groupby("證券代碼")["eps"].transform(
                lambda x: x.rolling(4, min_periods=2).sum()  # 4季 TTM
            )
            ttm_lag = df.groupby("證券代碼")["eps"].transform(
                lambda x: x.rolling(4, min_periods=2).sum().shift(4)
            )
            df["累計稅後盈餘成長率％_rev"] = (
                (ttm / (ttm_lag.abs() + 1e-9) - 1).clip(-2, 10).fillna(0)
            )
            print("    EPS 衍生特徵 ✓")

        self.df = df.copy()   # 整合碎片化欄位

    # ──────────────────────────────────────────────────────────
    #  借券 / 融券特徵（FinLab 版）
    # ──────────────────────────────────────────────────────────

    def _add_short_selling_features(self):
        """
        FinLab 版本的借券 / 融券特徵。

        使用欄位（來自 DBLoader / database/YYYY.csv）：
            ms_balance      融券今日餘額（張）
            ms_limit        融券限額（張）
            sl_sell_balance 借券賣出餘額
            sl_balance      借券餘額（總）
            volume          成交股數

        計算：
            short_sell_volume_ratio = ms_balance / volume
            short_headroom          = 1 - ms_balance / ms_limit
            days_to_cover           = sl_sell_balance / avg_volume_20
            days_to_cover_rank      = 截面 percentile rank
            inst_retail_ratio       = sl_sell_balance / (sl_balance + 1e-9)
            short_net_flow_sec      = sl_sell_balance - sl_sell_balance.shift(1)
            借券賣出可使用額度_log  = log1p(ms_limit - ms_balance)
        """
        cfg = self.config
        df  = self.df

        # short_sell_volume_ratio
        if all(c in df.columns for c in ["ms_balance", _VOL]):
            df["short_sell_volume_ratio"] = (
                df["ms_balance"] / (df[_VOL] + 1e-9)
            ).clip(0, 1).fillna(0)

        # short_headroom：1 - 融券餘額 / 融券限額
        if all(c in df.columns for c in ["ms_balance", "ms_limit"]):
            df["short_headroom"] = (
                1 - df["ms_balance"] / (df["ms_limit"] + 1e-9)
            ).clip(0, 1).fillna(1.0)
            # 可使用額度 log
            headroom_lots = (df["ms_limit"] - df["ms_balance"]).clip(lower=0)
            df["借券賣出可使用額度_log"] = np.log1p(headroom_lots).fillna(0)

        # days_to_cover（借券賣出餘額 / 20日均量）
        if all(c in df.columns for c in ["sl_sell_balance", _VOL]):
            avg_vol_20 = (
                df.groupby("證券代碼")[_VOL]
                .transform(lambda x: x.rolling(20, min_periods=5).mean())
                .fillna(df[_VOL])
            )
            df["days_to_cover"] = (
                df["sl_sell_balance"] / (avg_vol_20 + 1e-9)
            ).clip(0, 250).fillna(0)
            df["days_to_cover_rank"] = (
                df.groupby("年月日")["days_to_cover"]
                .rank(method="average", pct=True)
            ).fillna(0.5)

        # inst_retail_ratio：借券賣出 / 借券總餘額
        if all(c in df.columns for c in ["sl_sell_balance", "sl_balance"]):
            df["inst_retail_ratio"] = (
                df["sl_sell_balance"] / (df["sl_balance"] + 1e-9)
            ).clip(0, 1).fillna(0)

        # short_net_flow_sec：借券賣出餘額日變動
        if "sl_sell_balance" in df.columns:
            df["short_net_flow_sec"] = (
                df.groupby("證券代碼")["sl_sell_balance"]
                .transform(lambda x: x.diff())
                .fillna(0)
            )

        # Rolling 特徵
        if "sl_sell_balance" in df.columns:
            for w in cfg.short_windows:
                min_p = max(w // 2, 3)
                df[f"short_balance_chg_{w}d"] = (
                    df.groupby("證券代碼")["sl_sell_balance"]
                    .transform(lambda x, _w=w: x.pct_change(_w, fill_method=None).clip(-2, 2))
                    .fillna(0)
                )
            roll_count = len(cfg.short_windows)
            print(f"    short_balance_chg rolling: {roll_count} 欄  windows={cfg.short_windows}")

        if cfg.short_log_transform:
            for col in ["sl_sell_balance", "sl_balance", "ms_balance"]:
                if col in df.columns:
                    df[f"{col}_log"] = np.log1p(df[col].clip(lower=0)).fillna(0)

        self.df = df

        exist = [c for c in [
            "short_sell_volume_ratio", "short_headroom", "days_to_cover",
            "days_to_cover_rank", "inst_retail_ratio", "short_net_flow_sec",
            "借券賣出可使用額度_log",
        ] if c in self.df.columns]
        print(f"    借券/融券衍生特徵: {len(exist)} 欄 ✓")

    # ──────────────────────────────────────────────────────────
    #  Beta（Vasicek Shrinkage）& vol20（年化）
    # ──────────────────────────────────────────────────────────

    def _add_beta_vol_features(self):
        """
        daily_return = adj_close pct_change（backward-looking）
        vol20        = 20日滾動 std × √252（年化）
        beta         = Vasicek shrinkage（60日窗口）

        ⚠ daily_return 與 market_return 必須在本步驟前已存在。
        """
        if not self._require_cols(_CLOSE, "market_return"):
            return False

        window = self.config.beta_window
        min_p  = max(window // 2, 10)

        # daily_return（backward-looking）
        self.df["daily_return"] = (
            self.df.groupby("證券代碼")[_CLOSE]
            .transform(lambda x: x.pct_change(fill_method=None))
            .fillna(0)
        )

        # vol20（年化）
        self.df["vol20"] = (
            self.df.groupby("證券代碼")["daily_return"]
            .transform(lambda x: x.rolling(20, min_periods=10).std() * np.sqrt(252))
            .fillna(0)
        )

        # Step A：rolling beta_raw
        def _rolling_beta_raw(group: pd.DataFrame) -> pd.Series:
            r   = group["daily_return"]
            mkt = group["market_return"]
            cov = r.rolling(window, min_periods=min_p).cov(mkt)
            var = mkt.rolling(window, min_periods=min_p).var()
            return (cov / (var + 1e-9)).clip(-5, 5)

        beta_raw = (
            self.df.groupby("證券代碼", group_keys=False)
            .apply(_rolling_beta_raw, include_groups=False)
        )
        self.df["_beta_raw"] = beta_raw.values

        # Step B：per-stock 估計誤差
        var_estimation = (
            self.df.groupby("證券代碼")["_beta_raw"]
            .transform(lambda x: x.rolling(window, min_periods=min_p).std() ** 2)
            .bfill().fillna(0.25)
        )

        # Step C：截面方差
        var_cross = (
            self.df.groupby("年月日")["_beta_raw"]
            .transform("var").fillna(0)
        )

        # Step D & E：Vasicek shrinkage
        w = (var_cross / (var_cross + var_estimation + 1e-9)).clip(0, 1)
        self.df["beta"] = (
            (w * self.df["_beta_raw"] + (1 - w) * 1.0)
            .clip(-5, 5).fillna(1.0)
        )
        self.df = self.df.drop(columns=["_beta_raw"], errors="ignore")

        print(
            f"  beta  視窗:{window}日 | "
            f"Vasicek 後均值:{self.df['beta'].mean():.3f} | "
            f"std:{self.df['beta'].std():.3f}"
        )
        print(
            f"  vol20 年化  | "
            f"均值:{self.df['vol20'].mean():.4f} | "
            f"std:{self.df['vol20'].std():.4f}"
        )

    # ──────────────────────────────────────────────────────────
    #  Return Labels（含 excess_return）
    # ──────────────────────────────────────────────────────────

    def _add_return_features(self):
        """
        return = (adj_open[T+2] − adj_open[T+1]) / adj_open[T+1]

        Labels：
            return_tick      : return > tick_threshold  → 1
            return_tick_0    : return > 0               → 1
            excess_return    : return − beta × market_return_fwd
            excess_return_tick

        ⚠ market_return_fwd 在本步驟後強制 drop。
        """
        if not self._require_cols(_OPEN, "market_return_fwd", "beta"):
            return False

        cfg = self.config

        def _calc(g):
            t1 = g[_OPEN].shift(-1)
            t2 = g[_OPEN].shift(-2)
            return (t2 - t1) / (t1 + 1e-9)

        ret = (
            self.df.groupby("證券代碼")
            .apply(_calc, include_groups=False)
            .reset_index(level=0, drop=True)
        )
        self.df["return"] = ret.clip(-cfg.return_clip, cfg.return_clip)

        valid = self.df["return"].notna()
        self.df["return_tick"] = np.where(
            valid, (self.df["return"] > cfg.tick_threshold).astype(int), np.nan
        )
        self.df["return_tick_0"] = np.where(
            valid, (self.df["return"] > cfg.tick0_threshold).astype(int), np.nan
        )

        self.df["excess_return"] = (
            self.df["return"] - self.df["beta"] * self.df["market_return_fwd"]
        ).clip(-cfg.return_clip, cfg.return_clip)

        excess_valid = self.df["excess_return"].notna()
        self.df["excess_return_tick"] = np.where(
            excess_valid,
            (self.df["excess_return"] > cfg.tick_threshold).astype(int),
            np.nan,
        )

        # ⚠ drop market_return_fwd（look-ahead bias 防護）
        self.df = self.df.drop(columns=["market_return_fwd"], errors="ignore")
        print("  market_return_fwd dropped（look-ahead bias 防護）")

        pos_rate        = (self.df["return_tick"]        == 1).sum() / valid.sum()
        pos_rate0       = (self.df["return_tick_0"]      == 1).sum() / valid.sum()
        pos_excess_rate = (self.df["excess_return_tick"] == 1).sum() / excess_valid.sum()

        print(f"  return_tick        (>{cfg.tick_threshold:.1%}) 正樣本率: {pos_rate:.2%}")
        print(f"  return_tick_0      (>{cfg.tick0_threshold:.1%}) 正樣本率: {pos_rate0:.2%}")
        print(f"  excess_return_tick (>{cfg.tick_threshold:.1%}) 正樣本率: {pos_excess_rate:.2%}")

    # ──────────────────────────────────────────────────────────
    #  【新增】超額報酬特徵
    # ──────────────────────────────────────────────────────────

    def _add_excess_return_features(self):
        """
        報酬率1       = adj_close.pct_change()（日簡單報酬率）
        超額報酬日大盤 = 個股日報酬 - 大盤日報酬
        超額報酬週大盤 = 個股週報酬 - 大盤週報酬

        ⚠ 依賴 market_return；需在 benchmark 計算後呼叫。
        ⚠ 報酬率1 與 daily_return 含義相同，另存此欄遵循 finlab_dataset_check.csv 規範。
        """
        if not self._require_cols(_CLOSE, "market_return"):
            return False

        # ① 日簡單報酬率
        self.df["報酬率1"] = (
            self.df.groupby("證券代碼")[_CLOSE]
            .transform(lambda x: x.pct_change(fill_method=None))
            .clip(-0.5, 0.5)
            .fillna(0)
        )

        # ② 超額報酬日大盤
        self.df["超額報酬日大盤"] = (
            self.df["報酬率1"] - self.df["market_return"]
        ).fillna(0)

        # ③ 超額報酬週大盤（週首末收盤報酬 vs 大盤週複利報酬）
        def _weekly_excess(group):
            wk = group["年月日"].dt.to_period("W")
            first_c = group.groupby(wk)[_CLOSE].transform("first")
            last_c  = group.groupby(wk)[_CLOSE].transform("last")
            stock_w = last_c / (first_c + 1e-9) - 1
            mkt_w   = group.groupby(wk)["market_return"].transform(
                lambda x: (1 + x).prod() - 1
            )
            return (stock_w - mkt_w).fillna(0)

        self.df["超額報酬週大盤"] = (
            self.df.groupby("證券代碼", group_keys=False)
            .apply(_weekly_excess, include_groups=False)
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        print("    報酬率1 / 超額報酬日大盤 / 超額報酬週大盤 ✓")

    # ──────────────────────────────────────────────────────────
    #  【新增】估值特徵
    # ──────────────────────────────────────────────────────────

    def _add_valuation_features(self):
        """
        股價營收比     = market_value / monthly_revenue_ly_cum
                       （P/S Ratio，以去年累計營收為分母）
        淨值(千元)_rev = equity / 1000
                       （股東權益總額，季頻 ffill 後使用）
        """
        # ① P/S Ratio（去年累計營收）
        if self._require_cols("market_value", "monthly_revenue_ly_cum"):
            rev_ly = self.df["monthly_revenue_ly_cum"].replace(0, np.nan)
            self.df["股價營收比"] = (
                self.df["market_value"] / rev_ly
            ).clip(0, 200).fillna(0)
            print("    股價營收比 ✓")

        # ② 淨值（千元）
        if "equity" in self.df.columns:
            self.df["淨值(千元)_rev"] = (
                self.df["equity"] / 1000
            ).fillna(0)
            print("    淨值(千元)_rev ✓")

    # ──────────────────────────────────────────────────────────
    #  【新增】累計 EPS / 去年稅前盈餘
    # ──────────────────────────────────────────────────────────

    def _add_eps_cumulative_features(self):
        """
        累計每股稅後盈餘(WA)_rev = TTM EPS（trailing-twelve-month 4季加總）
        去年累計稅前盈餘_rev      = 上一完整年度 4季稅前淨利合計

        資料特性（2025.csv 實測）：
            eps          → 稀疏型（公告日才有值，非連續）
            pretax_profit→ 已 ffill（季頻 4–5 次更新後 ffill 至日頻）

        計算策略：
            Step 1：per-stock ffill
            Step 2：resample QE .last() 取各季末代表值
            Step 3：rolling(4).sum() for TTM
                   groupby(year).sum().shift(1 year) for 去年全年
            Step 4：reindex 回日頻 + ffill
        """
        df = self.df

        # ─── 累計每股稅後盈餘(WA)_rev（TTM）────────────────────
        if "eps" in df.columns:

            # ⚠ 同 _add_fundamental_features：apply 必須回傳「原始 df 整數索引」
            #   的 Series，否則共用日期索引會被 pandas pivot 成寬表 DataFrame。
            def _ttm_eps(group: pd.DataFrame) -> pd.Series:
                orig_index = group.index
                g = group.set_index("年月日")["eps"]
                date_idx = g.index
                g_filled = g.ffill()
                q_vals = g_filled.resample("QE").last().dropna()
                if q_vals.empty:
                    return pd.Series(np.nan, index=orig_index,
                                     name="累計每股稅後盈餘(WA)_rev")
                ttm = q_vals.rolling(4, min_periods=4).sum()
                vals = ttm.reindex(date_idx, method="ffill").to_numpy()
                return pd.Series(vals, index=orig_index,
                                 name="累計每股稅後盈餘(WA)_rev")

            ttm_result = df.groupby("證券代碼", group_keys=False).apply(
                _ttm_eps, include_groups=False
            )
            if isinstance(ttm_result, pd.DataFrame):   # 防呆壓平
                ttm_result = ttm_result.iloc[:, 0]
            df["累計每股稅後盈餘(WA)_rev"] = ttm_result.reindex(df.index).fillna(0)
            print("    累計每股稅後盈餘(WA)_rev (TTM 4Q) ✓")

        # ─── 去年累計稅前盈餘_rev（上一完整年度合計）──────────
        if "pretax_profit" in df.columns:

            def _prev_year_pretax(group: pd.DataFrame) -> pd.Series:
                orig_index = group.index
                g = group.set_index("年月日")["pretax_profit"]
                date_idx = g.index
                g_filled = g.ffill()
                q_vals = g_filled.resample("QE").last().dropna()
                if q_vals.empty:
                    return pd.Series(np.nan, index=orig_index,
                                     name="去年累計稅前盈餘_rev")
                # 各年合計
                annual = q_vals.groupby(q_vals.index.year).sum()
                vals = np.array(
                    [annual.get(y - 1, np.nan) for y in date_idx.year],
                    dtype=float,
                )
                return pd.Series(vals, index=orig_index,
                                 name="去年累計稅前盈餘_rev")

            pretax_result = df.groupby("證券代碼", group_keys=False).apply(
                _prev_year_pretax, include_groups=False
            )
            if isinstance(pretax_result, pd.DataFrame):   # 防呆壓平
                pretax_result = pretax_result.iloc[:, 0]
            df["去年累計稅前盈餘_rev"] = pretax_result.reindex(df.index).fillna(0)
            print("    去年累計稅前盈餘_rev ✓")

        self.df = df.copy()   # 整合碎片化欄位