"""
price_return.py
================
價格衍生特徵與報酬率特徵：高低價差 / 週轉率 / 多期間報酬率（YTD/MTD/WTD/QTD）/
return label（含 excess_return）/ 超額報酬特徵。

Look-ahead Bias 修正歷史請見各方法 docstring。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._base import _FeatureHelperMixin, _CLOSE, _RCLOSE, _HIGH, _LOW, _OPEN, _VOL, _AMT


class PriceReturnFeaturesMixin(_FeatureHelperMixin):
    """價格衍生 / 報酬率相關 _add_* 方法。繼承類別需提供：self.df, self.config"""

    # ──────────────────────────────────────────────────────────
    #  價格衍生特徵
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
    #  多期間報酬率 / YTD / MTD / QTD
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
            # ★ 2026-07-13 清理：移除未使用的 isocalendar first-occurrence
            #   dead code（yw/first 計算後從未被引用）。邏輯不變。
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
    #  delay1 反轉／微結構／流動性因子
    # ──────────────────────────────────────────────────────────

    def _add_delay1_microstructure_features(self):
        """delay1 反轉／微結構／流動性因子（2026-07-23 由 1003_delay1 IC/corr 篩選 +
        重訓對照選入；D1 Sharpe 3.73→3.91、OOF IC 0.092→0.097、MDD −30%→−27%）。

        動機：D1 day-1 P&L 有八成與純個股 1 日反轉同向（載荷 0.76），但控制反轉後
        仍留 95%/yr 正交 alpha → 補既有 報酬率1（≈收盤對收盤反轉）沒涵蓋的維度。
        5 個因子皆只用 ≤ 收盤[T] 資訊（無前視）：

          rev_5d        = -(adj_open_t / adj_open_{t-5} - 1)              5 日開盤反轉（不同 horizon）
          ov_night_1d   = adj_open_t / adj_close_{t-1} - 1               隔夜跳空（隔夜 SOX/美股微結構）
          intraday_1d   = adj_close_t / adj_open_t - 1                   日內（日內反轉遠強於隔夜）
          rev_volscaled = -(adj_open_t/adj_open_{t-1}-1)/(|vol20|+eps)   波動正規化反轉（stat-arb 標準）
          illiq_amihud  = rolling5 mean(|cc_ret| / log1p(amount))        Amihud 流動性（反轉放大器）

        依賴 vol20（_add_beta_vol_features）與 amount → 須排在 _add_beta_vol_features 之後。
        不 fillna（保留 NaN，LightGBM 原生處理；與 1003_delay1 重訓驗證版一致）。
        """
        if not self._require_cols(_OPEN, _CLOSE, _AMT, "vol20"):
            return False

        g = self.df.groupby("證券代碼")
        o, c = self.df[_OPEN], self.df[_CLOSE]

        self.df["rev_5d"] = (
            g[_OPEN].transform(lambda x: -(x / x.shift(5) - 1)).clip(-0.5, 0.5)
        )
        self.df["ov_night_1d"] = (o / g[_CLOSE].shift(1) - 1).clip(-0.5, 0.5)
        self.df["intraday_1d"] = (c / (o + 1e-9) - 1).clip(-0.5, 0.5)

        rev_oo = g[_OPEN].transform(lambda x: -(x / x.shift(1) - 1))
        self.df["rev_volscaled"] = (rev_oo / (self.df["vol20"].abs() + 1e-6)).clip(-25, 25)

        cc_ret = g[_CLOSE].transform(lambda x: (x / x.shift(1) - 1)).abs()
        amihud = cc_ret / np.log1p(self.df[_AMT].clip(lower=0))
        self.df["illiq_amihud"] = (
            amihud.groupby(self.df["證券代碼"])
                  .transform(lambda s: s.rolling(5, min_periods=3).mean())
        )
        print("    delay1 微結構因子: rev_5d / ov_night_1d / intraday_1d / "
              "rev_volscaled / illiq_amihud ✓")

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

        ★ 2026-06-25：market_return_fwd 改為保留（不再強制 drop）。
          原因：backtest.py 需要這個欄位作為「與策略 return 同窗（open[T+1]→open[T+2]）
          對齊」的大盤 benchmark,單純 drop 掉會讓 backtest.py 拿不到正確對齊的大盤序列。
          look-ahead 防護改由訓練端負責：main_fix.py 的 EXTRA_EXCLUDE_COLS 已將
          market_return_fwd 排除於模型訓練特徵之外（與 amount / market_value / close
          同樣是「保留於 CSV、不進訓練」的 conditioning / benchmark 欄位）。
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

        pos_rate        = (self.df["return_tick"]        == 1).sum() / valid.sum()
        pos_rate0       = (self.df["return_tick_0"]      == 1).sum() / valid.sum()
        pos_excess_rate = (self.df["excess_return_tick"] == 1).sum() / excess_valid.sum()

        print(f"  return_tick        (>{cfg.tick_threshold:.1%}) 正樣本率: {pos_rate:.2%}")
        print(f"  return_tick_0      (>{cfg.tick0_threshold:.1%}) 正樣本率: {pos_rate0:.2%}")
        print(f"  excess_return_tick (>{cfg.tick_threshold:.1%}) 正樣本率: {pos_excess_rate:.2%}")
        print("  market_return_fwd 保留（訓練端由 main_fix.py EXTRA_EXCLUDE_COLS 排除）")

    # ──────────────────────────────────────────────────────────
    #  超額報酬特徵
    # ──────────────────────────────────────────────────────────

    def _add_excess_return_features(self):
        """
        報酬率1       = adj_close.pct_change()（日簡單報酬率，backward）
        超額報酬日大盤 = 個股日報酬 - 大盤日報酬（backward）
        超額報酬週大盤 = 個股 WTD 報酬 - 大盤 WTD 報酬（backward，到當日為止）

        ⚠ 依賴 market_return；需在 benchmark 計算後呼叫。
        ⚠ 報酬率1 與 daily_return 含義相同，另存此欄遵循 finlab_dataset_check.csv 規範。

        ─────────────────────────────────────────────
        ★ 2026-06-23 look-ahead 修正（超額報酬週大盤）
        ─────────────────────────────────────────────
          舊版以 groupby(週).transform("last") 取「當週最後一日（週五）收盤」，
          並賦值給該週每一列 → 週一~週四的列含有未來收盤（最多 T+4），
          且 label excess_return_tick 的報酬視窗 open[T+1]→open[T+2] 落在同一週內
          → 直接洩漏 label（leak_probe.py Part A 會以 ★★★/★★ 命中）。
          現改為 week-to-date（WTD）：
            個股腳：分子用「當列收盤 group[_CLOSE]」而非整週最後收盤；
            大盤腳：用 cumprod（週內到當日的累乘）而非整週 prod。
          兩腳皆只用「週起點 ~ 當日」資訊，無未來函數。
        """
        if not self._require_cols(_CLOSE, "market_return"):
            return False

        # ① 日簡單報酬率（backward）
        self.df["報酬率1"] = (
            self.df.groupby("證券代碼")[_CLOSE]
            .transform(lambda x: x.pct_change(fill_method=None))
            .clip(-0.5, 0.5)
            .fillna(0)
        )

        # ② 超額報酬日大盤（backward）
        self.df["超額報酬日大盤"] = (
            self.df["報酬率1"] - self.df["market_return"]
        ).fillna(0)

        # ③ 超額報酬週大盤（WTD：週起點 → 當日，無未來函數）
        def _weekly_excess(group):
            wk = group["年月日"].dt.to_period("W")
            # 個股：當列收盤 / 當週第一日收盤 - 1（到當日為止的 WTD）
            first_c   = group.groupby(wk)[_CLOSE].transform("first")
            stock_wtd = group[_CLOSE] / (first_c + 1e-9) - 1
            # 大盤：週內 expanding 累乘（cumprod），同樣只到當日。
            # ★ 2026-07-13 leg-alignment 修正：
            #   個股腳以「週首日收盤」為基準 → 不含週首日自身的報酬；
            #   舊版大盤腳 cumprod 從週首日開始累乘 → 多含一天
            #   （上週末收 → 週首日收），兩腳窗口差一天，
            #   特徵被注入 −market_return[週首日] 的系統性偏移
            #   （週首日當天：個股腳=0、大盤腳=當日大盤報酬 → 特徵=純大盤噪音）。
            #   修法：cumprod 除掉週首日的 (1+r)，使大盤腳同樣以
            #   「週首日收盤」為基準（週首收 → 當日收），與個股腳完全同窗。
            #   修正後週首日兩腳皆為 0。
            mkt_wtd   = group.groupby(wk)["market_return"].transform(
                lambda x: (1 + x).cumprod() / (1 + x.iloc[0]) - 1
            )
            return (stock_wtd - mkt_wtd).fillna(0)

        self.df["超額報酬週大盤"] = (
            self.df.groupby("證券代碼", group_keys=False)
            .apply(_weekly_excess, include_groups=False)
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        print("    報酬率1 / 超額報酬日大盤 / 超額報酬週大盤(WTD, 無未來) ✓")
