"""
test.py
=======
驗證 database_make/ 目錄下各 CSV 的 market_return 欄位：
  1. 同一天所有股票的 market_return 是否完全相同（std == 0）
  2. 跨 CSV 檔案的同日 market_return 是否一致
  3. 輸出最終可用的 market_series（DatetimeIndex → float）

執行方式：
    python test.py
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

DATABASE_MAKE_DIR = Path("database_make")
DATE_COL          = "年月日"
MKT_COL           = "market_return"


def load_market_from_dir(directory: Path) -> dict[str, pd.Series]:
    """讀取目錄下所有 CSV，回傳 {filename: market_series}。"""
    result = {}
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{directory} 下找不到任何 .csv")

    for path in csv_files:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print(f"  [SKIP] {path.name}：讀取失敗 ({e})")
            continue

        if DATE_COL not in df.columns or MKT_COL not in df.columns:
            print(f"  [SKIP] {path.name}：缺少欄位 {DATE_COL!r} 或 {MKT_COL!r}")
            continue

        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df[MKT_COL]  = pd.to_numeric(df[MKT_COL], errors="coerce")

        # 每日 market_return 應該相同 → 取 first（並記錄 std 供檢驗）
        daily = df.groupby(DATE_COL)[MKT_COL]
        daily_std   = daily.std().fillna(0)
        daily_first = daily.first()

        n_inconsistent = (daily_std > 1e-10).sum()
        if n_inconsistent > 0:
            bad_dates = daily_std[daily_std > 1e-10].index.tolist()[:5]
            print(f"  [WARN] {path.name}：{n_inconsistent} 天 market_return 不一致（前5筆：{bad_dates}）")
        else:
            print(f"  [OK]   {path.name}：{len(daily_first)} 天，market_return 同日完全一致")

        result[path.name] = daily_first

    return result


def cross_file_check(series_dict: dict[str, pd.Series]) -> pd.Series:
    """
    比較不同 CSV 在重疊日期的 market_return 是否相同。
    回傳合併後的 market_series（有衝突時取中位數並警告）。
    """
    if not series_dict:
        raise ValueError("沒有可用的 market_return 序列")

    combined = pd.concat(series_dict.values(), axis=1)
    combined.columns = list(series_dict.keys())

    # 跨檔一致性
    row_std = combined.std(axis=1).fillna(0)
    n_conflict = (row_std > 1e-10).sum()
    if n_conflict > 0:
        conflict_dates = row_std[row_std > 1e-10].index.tolist()[:5]
        print(f"\n  [WARN] 跨 CSV 共 {n_conflict} 天 market_return 不一致（前5筆：{conflict_dates}）")
        print(         "         → 改用中位數合併")
        market_series = combined.median(axis=1)
    else:
        print(f"\n  [OK]   跨 CSV market_return 完全一致（共 {len(combined)} 天）")
        market_series = combined.mean(axis=1)

    market_series.name = MKT_COL
    market_series = market_series.sort_index()
    return market_series


def main():
    print(f"\n{'='*60}")
    print(f"  market_return 一致性測試")
    print(f"  來源目錄：{DATABASE_MAKE_DIR}")
    print(f"{'='*60}\n")

    series_dict = load_market_from_dir(DATABASE_MAKE_DIR)
    market_series = cross_file_check(series_dict)

    # 摘要統計
    print(f"\n  市場報酬摘要：")
    print(f"    日期範圍：{market_series.index.min().date()} ~ {market_series.index.max().date()}")
    print(f"    交易日數：{len(market_series)}")
    print(f"    年化報酬：{(1 + market_series).prod() ** (252 / len(market_series)) - 1:.2%}")
    print(f"    年化波動：{market_series.std() * np.sqrt(252):.2%}")
    print(f"    NaN 數量：{market_series.isna().sum()}")
    print(f"\n  前5筆：\n{market_series.head().to_string()}")

    # 輸出供 backtest.py 使用
    out_path = Path("database_make/market_return_series.csv")
    market_series.reset_index().rename(columns={"index": DATE_COL}).to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )
    print(f"\n  ✓ 輸出：{out_path}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()