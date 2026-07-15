"""
fundamental.py
===============
基本面特徵：月營收 / EPS / 估值衍生特徵（純日頻 rolling，無 resample）。

Look-ahead Bias 修正歷史請見各方法 docstring。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._base import _FeatureHelperMixin, _W3M, _W12M


class FundamentalFeaturesMixin(_FeatureHelperMixin):
    """基本面 / 估值相關 _add_* 方法。繼承類別需提供：self.df, self.config"""

    # ──────────────────────────────────────────────────────────
    #  月營收 / EPS 衍生特徵
    # ──────────────────────────────────────────────────────────

    def _add_fundamental_features(self):
        """
        月營收衍生特徵（純日頻 rolling，無 resample）

        欄位說明：
          去年累計營收(千元)_rev      直接取用上游 monthly_revenue_ly_cum
                                      （FinLab 原始欄位，公告日 trigger，無穿越）
          單月營收成長率％_rev        直接取用上游 monthly_revenue_mom
          近3月累計營收變動率％_rev   rolling(63).sum() vs shift(63) rolling(63).sum()
          近12月累計營收成長率_rev    rolling(252).sum() vs shift(252) rolling(252).sum()
          與歷史最低單月營收比%_rev   expanding().min().shift(1) 純日頻
          單月每股營收(元)_rev        monthly_revenue / fii_shares
          單月每股稅後盈餘(WA)_rev    eps 直接取用（make.py 已 ffill）
          累計稅後盈餘成長率％_rev    → 移至 _add_eps_cumulative_features 統一管理

        修正重點（2026-06）：
          ① 廢除所有 resample("ME") / resample("BME")
          ② monthly_revenue 已是公告日 ffill 日頻，直接 rolling 即可
             rolling(63)  ≈ 3 個月；rolling(252) ≈ 12 個月
          ③ expanding min 使用 shift(1) 嚴格落後對齊
          ④ eps rolling(4)（日頻語意錯誤：非 4 季）移除，TTM 由 _add_eps_cumulative_features 負責
        """
        df = self.df

        # ── ① 去年累計營收（FinLab 原始欄位，直接取用）─────────
        if "monthly_revenue_ly_cum" in df.columns:
            df["去年累計營收(千元)_rev"] = (df["monthly_revenue_ly_cum"] / 1000).fillna(0)
            print("    去年累計營收 ✓")

        # ── ② 單月營收成長率 MoM（FinLab 原始欄位，直接取用）───
        if "monthly_revenue_mom" in df.columns:
            df["單月營收成長率％_rev"] = df["monthly_revenue_mom"].fillna(0)
            print("    單月營收成長率 ✓")

        # ── ③ 近3月 / 近12月累計營收成長率（純日頻 rolling）────
        # monthly_revenue：FinLab 公告日 ffill 日頻序列，可直接 rolling。
        # rolling(63).sum()  ≈ 近 63 交易日（≈3個月）營收加總
        # 對比 shift(63) 後的前 63 日加總 → 計算成長率
        # ⚠ pandas 2.x groupby.apply pivot 陷阱：
        #   若每個 group 回傳的 Series 以「共用日期索引」為 index，
        #   pandas 會把這些 Series 展開成寬表 DataFrame。
        #   解法：保留原始 df 整數索引，apply 結束後 reindex 對齊。
        if "monthly_revenue" in df.columns:

            def _rev_growth_daily(group: pd.DataFrame) -> pd.DataFrame:
                orig_index = group.index
                rev = group["monthly_revenue"]   # 已是日頻，不需 set_index

                if rev.dropna().shape[0] < _W3M:
                    return pd.DataFrame(
                        {
                            "近3月累計營收變動率％_rev": np.nan,
                            "近12月累計營收成長率_rev":  np.nan,
                        },
                        index=orig_index,
                    )

                # 近 3 月（63 交易日）
                r3     = rev.rolling(_W3M, min_periods=_W3M // 2).sum()
                r3_lag = r3.shift(_W3M)
                chg3   = (r3 / (r3_lag.abs() + 1e-9) - 1).clip(-2, 10)

                # 近 12 月（252 交易日）
                r12     = rev.rolling(_W12M, min_periods=_W12M // 2).sum()
                r12_lag = r12.shift(_W12M)
                chg12   = (r12 / (r12_lag.abs() + 1e-9) - 1).clip(-2, 10)

                return pd.DataFrame(
                    {
                        "近3月累計營收變動率％_rev": chg3.to_numpy(),
                        "近12月累計營收成長率_rev":  chg12.to_numpy(),
                    },
                    index=orig_index,
                )

            rev_result = df.groupby("證券代碼", group_keys=False).apply(
                _rev_growth_daily, include_groups=False
            ).reindex(df.index)

            df["近3月累計營收變動率％_rev"] = (
                rev_result["近3月累計營收變動率％_rev"].fillna(0)
            )
            df["近12月累計營收成長率_rev"] = (
                rev_result["近12月累計營收成長率_rev"].fillna(0)
            )
            print("    近3月/12月累計營收成長率（日頻 rolling，無 resample）✓")

            # ── ④ 與歷史最低單月營收比（純日頻 expanding + shift(1)）─
            # expanding().min()：截至前一日的歷史最低值
            # shift(1)：嚴格落後，分母絕不含當日及未來值
            def _hist_low_ratio_daily(group: pd.DataFrame) -> pd.Series:
                orig_index = group.index
                rev = group["monthly_revenue"]
                if rev.dropna().empty:
                    return pd.Series(np.nan, index=orig_index,
                                     name="與歷史最低單月營收比%_rev")
                hist_min = rev.expanding().min().shift(1)   # 截至昨日歷史最低
                ratio = (rev / (hist_min.abs() + 1e-9) - 1).clip(0, 100)
                return pd.Series(ratio.to_numpy(), index=orig_index,
                                 name="與歷史最低單月營收比%_rev")

            hist_low = df.groupby("證券代碼", group_keys=False).apply(
                _hist_low_ratio_daily, include_groups=False
            )
            if isinstance(hist_low, pd.DataFrame):
                hist_low = hist_low.iloc[:, 0]
            df["與歷史最低單月營收比%_rev"] = hist_low.reindex(df.index).fillna(0)
            print("    與歷史最低單月營收比（日頻 expanding min + shift(1)）✓")

        # ── ⑤ 單月每股營收 ───────────────────────────────────────
        if all(c in df.columns for c in ["monthly_revenue", "fii_shares"]):
            df["單月每股營收(元)_rev"] = (
                df["monthly_revenue"] / (df["fii_shares"] + 1e-9)
            ).fillna(0)
            print("    單月每股營收 ✓")

        # ── ⑥ 單月每股稅後盈餘（直接取用）──────────────────────
        # ⚠ TTM 累計 EPS 與 累計稅後盈餘成長率％_rev 已移至
        #   _add_eps_cumulative_features，避免稀疏 eps 在日頻 rolling(4) 的語意錯誤
        if "eps" in df.columns:
            df["單月每股稅後盈餘(WA)_rev"] = df["eps"].fillna(0)
            print("    單月每股稅後盈餘(WA)_rev ✓（TTM 成長率見 _add_eps_cumulative_features）")

        self.df = df.copy()   # 整合碎片化欄位

    # ──────────────────────────────────────────────────────────
    #  估值特徵
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
    #  累計 EPS / 去年稅前盈餘
    # ──────────────────────────────────────────────────────────

    def _add_eps_cumulative_features(self):
        """
        累計 EPS / 稅前盈餘特徵（純日頻 rolling，無 resample("QE")）

        欄位說明：
          累計每股稅後盈餘(WA)_rev  TTM EPS ≈ rolling(252).sum()（≈4季）
          累計稅後盈餘成長率％_rev   TTM YoY = TTM / TTM.shift(252) - 1
          去年累計稅前盈餘_rev       ≈ rolling(252).sum().shift(252)

        資料特性：
          eps          → make.py preprocess 已做 per-stock ffill（進入此方法時為日頻連續值）
          pretax_profit→ 同上，已 ffill 至日頻

        修正重點（2026-06，真實 lookahead bug）：
          原始問題：
            eps.ffill() → resample("QE").last()
            稀疏 eps 在公告日（如 11/14）公告 Q3 EPS → ffill 覆蓋 Q3 季末 9/30
            → resample("QE").last() 取 9/30 的值 = 11/14 公告數字
            → 模型在 9/30 就看到了 11/14 的 EPS → 嚴重穿越 ✗

          修正策略：
            eps 進入時已是日頻連續值（preprocess ffill），
            shift(1) 確保公告當日不可用（隔日才生效），
            再 ffill 補齊 shift 產生的第一列 NaN，
            rolling(252, min_periods=63).sum() ≈ TTM 4 季加總。

          ⚠ min_periods=63（≈1季）容許早期 warmup 不足時仍輸出部分值；
            若要求嚴格 4 季完整，改為 min_periods=252。
        """
        df = self.df

        # ── ① TTM EPS & YoY 成長率 ──────────────────────────────
        if "eps" in df.columns:

            def _ttm_eps_daily(group: pd.DataFrame) -> pd.DataFrame:
                """
                eps 進入時已是 preprocess ffill 的日頻連續值。
                shift(1)：公告當日不可用，隔日才生效。
                ffill()：補齊 shift(1) 產生的第一列 NaN（保持日頻連續）。
                rolling(252).sum() ≈ TTM 4 季加總。
                """
                orig_index = group.index
                eps_safe = group["eps"].shift(1).ffill()

                ttm     = eps_safe.rolling(_W12M, min_periods=_W3M).sum()
                ttm_lag = ttm.shift(_W12M)    # 去年同期 TTM
                yoy     = (ttm / (ttm_lag.abs() + 1e-9) - 1).clip(-2, 10)

                return pd.DataFrame(
                    {
                        "累計每股稅後盈餘(WA)_rev": ttm.to_numpy(),
                        "累計稅後盈餘成長率％_rev":  yoy.to_numpy(),
                    },
                    index=orig_index,
                )

            eps_result = df.groupby("證券代碼", group_keys=False).apply(
                _ttm_eps_daily, include_groups=False
            )
            if isinstance(eps_result, pd.Series):    # 防呆壓平
                eps_result = eps_result.to_frame()
            eps_result = eps_result.reindex(df.index)
            df["累計每股稅後盈餘(WA)_rev"] = (
                eps_result["累計每股稅後盈餘(WA)_rev"].fillna(0)
            )
            df["累計稅後盈餘成長率％_rev"] = (
                eps_result["累計稅後盈餘成長率％_rev"].fillna(0)
            )
            print("    累計每股稅後盈餘(WA)_rev（TTM rolling252 shift(1)）✓")
            print("    累計稅後盈餘成長率％_rev（TTM YoY）✓")

        # ── ② 去年累計稅前盈餘（上一完整年度代理值）────────────
        # rolling(252).sum()        ≈ 過去 1 年稅前盈餘合計（TTM）
        # .shift(252)               ≈ 再往前推 1 年 = 去年全年合計
        # 語意與原始「上一完整年度 4 季合計」高度等效，且無 resample 邊界問題。
        if "pretax_profit" in df.columns:

            def _prev_year_pretax_daily(group: pd.DataFrame) -> pd.Series:
                orig_index = group.index
                # shift(1)：季報公告當日不可用；ffill：補齊 shift 產生的 NaN
                pt_safe = group["pretax_profit"].shift(1).ffill()
                prev_annual = (
                    pt_safe.rolling(_W12M, min_periods=_W3M).sum()
                           .shift(_W12M)
                )
                return pd.Series(
                    prev_annual.to_numpy(),
                    index=orig_index,
                    name="去年累計稅前盈餘_rev",
                )

            pretax_result = df.groupby("證券代碼", group_keys=False).apply(
                _prev_year_pretax_daily, include_groups=False
            )
            if isinstance(pretax_result, pd.DataFrame):    # 防呆壓平
                pretax_result = pretax_result.iloc[:, 0]
            df["去年累計稅前盈餘_rev"] = (
                pretax_result.reindex(df.index).fillna(0)
            )
            print("    去年累計稅前盈餘_rev（rolling252 shift252 + shift(1)）✓")

        self.df = df.copy()   # 整合碎片化欄位
