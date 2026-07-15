"""
_base.py
========
objective/features/ 下各主題 Mixin（technical / price_return / risk /
institutional / fundamental）共用的欄位常數與輔助方法。
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
#  交易日視窗常數
# ─────────────────────────────────────────────────────────────
_W1M  = 21    # ≈ 1 個月
_W3M  = 63    # ≈ 1 季（3 個月）
_W12M = 252   # ≈ 1 年（12 個月）

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


class _FeatureHelperMixin:
    """所有主題 Mixin 共用的輔助方法。繼承類別需提供：self.df"""

    def _require_cols(self, *cols: str) -> bool:
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            print(f"  ✗ 缺少欄位: {missing}")
            return False
        return True
