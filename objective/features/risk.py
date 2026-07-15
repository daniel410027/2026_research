"""
risk.py
=======
風險特徵：CAPM Beta 多視窗、Vasicek shrinkage Beta 與年化波動率（vol20）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._base import _FeatureHelperMixin, _CLOSE


class RiskFeaturesMixin(_FeatureHelperMixin):
    """Beta / 波動率相關 _add_* 方法。繼承類別需提供：self.df, self.config"""

    # ──────────────────────────────────────────────────────────
    #  CAPM Beta 多視窗
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
