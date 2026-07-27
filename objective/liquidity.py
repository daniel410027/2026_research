"""
liquidity.py
============
流動性篩選 mask（805_group2 formula3_amount_mv_turnover）。

★ 2026-07-27：本檔為 `2026_daily` 與 `2026_research` **共用的單一權威副本**。
  在此之前同一份公式有三個副本各自維護：
      - 2026_daily/daily_model/liquidity.py
      - 2026_daily/daily_model_0713_hyper.py（舊單檔版，已凍結不動）
      - 2026_research/main_fix_0709.py（內嵌於入口腳本）
  research 那份躲在入口腳本裡，`objective/` 的對拍檢查照不到它 —— 一旦公式改動
  只改一邊，兩端的訓練 universe 會靜默分歧，而 universe 分歧會讓所有 IC/Sharpe
  比較失去意義。收進 `objective/` 之後，它與其他模型邏輯一起被同步檢查。

  `daily_model/liquidity.py` 現為 re-export shim（保留舊 import 路徑）。

⚠ 兩個 repo 的這個檔案應保持 **byte 一致**；要改公式請兩邊一起改。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_liquidity_mask(
    df:         pd.DataFrame,
    w1:         float,
    w2:         float,
    keep_ratio: float,
    date_col:   str = "年月日",
) -> pd.Series:
    """
    formula3_amount_mv_turnover 流動性篩選。

    score = w1·Z(ln(amount)) + w2·Z(ln(market_value))
            + (1-w1-w2)·Z(ln(amount/market_value))

    - Z(·) 為每日 cross-sectional z-score
    - 每日依 score 由大到小排名，保留前 keep_ratio 比例
    - amount / market_value 缺失、<=0、或當日 std=0（如僅 1 檔）
      → score = NaN → 明確不合格（沿用 805_group2 教訓：NaN 不得意外通過篩選）

    Returns
    -------
    pd.Series[bool]，index 與 df 對齊；True = 通過篩選。
    """
    for col in ("amount", "market_value"):
        if col not in df.columns:
            raise KeyError(
                f"流動性篩選需要 '{col}' 欄，但 database_make/ 資料中找不到。"
                f"請確認 make_new.py 有保留該欄。"
            )

    amt = pd.to_numeric(df["amount"],       errors="coerce").where(lambda s: s > 0)
    mv  = pd.to_numeric(df["market_value"], errors="coerce").where(lambda s: s > 0)

    ln_amt = np.log(amt)
    ln_mv  = np.log(mv)
    ln_to  = np.log(amt / mv)          # turnover = amount / market_value

    grp = df[date_col]

    def _z(s: pd.Series) -> pd.Series:
        g   = s.groupby(grp)
        std = g.transform("std")
        return (s - g.transform("mean")) / std.where(std > 0)

    w3    = 1.0 - w1 - w2
    score = w1 * _z(ln_amt) + w2 * _z(ln_mv) + w3 * _z(ln_to)

    # 每日排名（大→小）取前 keep_ratio；NaN score 的 rank 為 NaN → 比較為 False
    rank_pct = score.groupby(grp).rank(ascending=False, pct=True, method="first")
    mask     = score.notna() & (rank_pct <= keep_ratio)
    return mask


def liq_tag(w1: float, w2: float, keep_ratio: float) -> str:
    """
    篩選參數 → tag 字串（與 805_group2 summary 命名慣例一致）。

    e.g. w1=0.1004, w2=0.1896, kr=0.2000 → 'f3_w101004_w201896_kr02000'

    研究端用它當快取目錄名（見 main_fix_0709.py::build_liquidity_filtered_dir）；
    daily 端不做快取，但保留同一個命名函式，兩邊講同一組參數時用同一個名字。
    """
    def _fmt(x: float) -> str:
        return f"{x:.4f}".replace("0.", "").zfill(5)
    return f"f3_w1{_fmt(w1)}_w2{_fmt(w2)}_kr{_fmt(keep_ratio)}"
