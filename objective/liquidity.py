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


#: 行情凍結判定用的欄位（research 的 database_make/ 只有這兩欄；daily 端另有成交股數）
STALE_CHECK_COLS: tuple[str, ...] = ("close", "amount")


def find_stale_rows(
    df,
    window: int = 5,
    date_col: str = "年月日",
    code_col: str = "證券代碼",
    cols: tuple[str, ...] = STALE_CHECK_COLS,
):
    """
    標出「行情凍結」的列：同一檔股票連續 `window` 個交易日，`cols` 每一欄都逐字相同。

    ★ 2026-08-04 從 2026_daily 的 `stock_list/trading_config.py::find_stale_codes`
      移植過來（daily commit 7bdf5a5，2026-08-03）。

    為什麼需要：daily 的下載端過去對 `price:` 類 dataset 做 ffill，已下市／長期
    停牌的股票會把最後一筆真實行情**無限延續**下去，製造出「永遠有量、永遠不漲
    不跌」的殭屍股。實例 6806 於 2026-06-23 終止上市，最後交易日的
    close 3.51 / amount 23,068,336 被逐字複製到 07-30，而且**通過了流動性篩選**
    （amount 凍結在下市前的水準，排名照樣進前 20%）。research 的
    `database_make/` 是那次修正之前從 daily 搬過來的，所以仍帶著這批列。

    對研究的傷害有兩層：
      · 訓練端——凍結列的報酬恆為 0、特徵恆定，是純雜訊樣本
      · 測試端——它們會進入 decile 排序，模型若把它們排進 D1，回測會拿到一個
        現實中根本賣不掉的部位（daily 那邊實測會被選進應持有清單）

    與 daily 版本的兩點差異，都是刻意的：
      1. daily 回傳「代碼集合」（它只關心「今天能不能買這檔」）；這裡回傳
         **逐列布林**——同一檔股票可能只有某一段時間凍結（實測 2327 在 2025 年
         只有部分期間被標記），整檔剔除會誤殺正常交易的日子。
      2. daily 用 收盤價/成交股數/成交金額 三欄，research 的資料只有 close 與
         amount。兩欄同時逐字凍結已足以判定——amount 是大數，真實交易連續 5 天
         完全相同的機率可忽略。

    已知限制（與 daily 相同）：凍結的**前 window-1 天標不出來**，要湊滿視窗才
    能確認。NaN 不算「相同」（`NaN == NaN` 為 False），所以真正沒有資料的列不會
    被誤判為凍結——它們本來就會被流動性篩選擋掉。

    回傳：與 `df` 同 index 的布林 Series，True = 該列行情凍結。
    """
    import pandas as pd

    if window < 2:
        raise ValueError(f"window 需 >= 2，得到 {window}")
    use = [c for c in cols if c in df.columns]
    if not use:
        return pd.Series(False, index=df.index)

    order = df.sort_values([code_col, date_col]).index
    s = df.loc[order]

    # 與「前一列」相比每一欄都相同（且同一檔股票）
    same_as_prev = pd.Series(True, index=order)
    for c in use:
        prev = s[c].shift(1)
        same_as_prev &= (s[c] == prev)          # NaN == NaN → False，符合上面的說明
    same_as_prev &= (s[code_col] == s[code_col].shift(1))

    # 連續 window-1 次「與前一列相同」＝ window 列逐字相同
    run = (same_as_prev.groupby(s[code_col].values)
           .rolling(window - 1, min_periods=window - 1).sum()
           .reset_index(level=0, drop=True))
    stale = (run == window - 1).fillna(False)

    return stale.reindex(df.index).fillna(False).astype(bool)


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
