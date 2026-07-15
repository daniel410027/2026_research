"""
objective/features/
====================
FeatureMixin：所有特徵工程方法（_add_*），依主題拆分為子模組：
    technical.py      傳統技術指標（RSI / MA / MACD / KD / Williams %R / ROC /
                       Momentum / ATR / Bollinger / OBV / VWAP / Volume / CCI）
    price_return.py   價格衍生 / 多期間報酬率 / return label / 超額報酬
    risk.py            CAPM Beta / Vasicek shrinkage Beta / vol20
    institutional.py  三大法人 / 融資融券 / 借券衍生特徵
    fundamental.py    月營收 / EPS / 估值衍生特徵

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

Look-ahead Bias 修正（2026-06）：
    _add_fundamental_features：
        - 廢除 resample("ME") / resample("BME")
        - monthly_revenue 已是公告日 ffill 日頻，改用純日頻 rolling：
            近3月累計營收變動率  → rolling(63).sum()  vs shift(63) rolling(63).sum()
            近12月累計營收成長率 → rolling(252).sum() vs shift(252) rolling(252).sum()
            與歷史最低單月營收比 → expanding().min().shift(1) 純日頻
        - eps 在 _add_fundamental_features 的 rolling(4)（日頻語意錯誤）已移除，
          TTM 統一由 _add_eps_cumulative_features 負責
    _add_eps_cumulative_features：
        - 廢除 resample("QE")（真實穿越：稀疏 eps ffill 後季末點含未來公告值）
        - 改為：eps.shift(1)（公告隔日生效）→ ffill → rolling(252).sum()（TTM）
          注意：make.py preprocess 已對 eps 做 per-stock ffill，
                進入此方法時 eps 已是日頻連續值，shift(1) 確保公告當日不可用
        - pretax_profit 同樣改為 shift(1) + ffill + rolling(252).sum().shift(252)
        - 累計稅後盈餘成長率％_rev 從 _add_fundamental_features 移至此方法統一計算

作者：Daniel Huang
"""

from __future__ import annotations

from .technical import TechnicalFeaturesMixin
from .price_return import PriceReturnFeaturesMixin
from .risk import RiskFeaturesMixin
from .institutional import InstitutionalFeaturesMixin
from .fundamental import FundamentalFeaturesMixin


class FeatureMixin(
    TechnicalFeaturesMixin,
    PriceReturnFeaturesMixin,
    RiskFeaturesMixin,
    InstitutionalFeaturesMixin,
    FundamentalFeaturesMixin,
):
    """
    所有 _add_* 特徵工程方法的 Mixin（組合自本目錄下各主題子 Mixin）。
    繼承類別需提供：self.df, self.config
    """


__all__ = [
    "FeatureMixin",
    "TechnicalFeaturesMixin",
    "PriceReturnFeaturesMixin",
    "RiskFeaturesMixin",
    "InstitutionalFeaturesMixin",
    "FundamentalFeaturesMixin",
]
