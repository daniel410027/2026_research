"""
test.py — Walk-Forward 資料處理診斷腳本（self-contained）
=========================================================
目的：不依賴 objective 套件，直接複刻 model.py 的資料流，
      驗證以下問題並印出 PASS / FAIL：

  [T1] 環境 / 路徑檢查
  [T2] TimeSeriesSplit 排序 bug（核心）：
       train_df 為 stock-major 排序，丟進 TimeSeriesSplit 後
       CV 折是「按股票分塊」而非「按時間切」。
       → 同時示範修正（先 sort by 年月日）後的差異。
  [T3] 每年最後交易日的 label NaN（邊界掉資料）
  [T4] walk-forward window 的 train/test 年份是否乾淨不重疊
  [T5] look-ahead 快速掃描（market_return_fwd 等前視欄位是否外洩）

執行：
    python test.py
    python test.py --dir database_make --target excess_return_tick

把整段輸出貼回來即可。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

ID_COLS = ["證券代碼", "年月日"]
LABEL_COLS = ["return", "return_tick", "return_tick_0",
              "excess_return", "excess_return_tick"]
# 任何「不應該出現在特徵裡」的前視欄位
LOOKAHEAD_SUSPECTS = ["market_return_fwd", "fwd", "future", "next_", "_t1", "_t+1"]


def _hr(title: str):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _ok(msg):   print(f"  ✓ PASS  {msg}")
def _bad(msg):  print(f"  ✗ FAIL  {msg}")
def _warn(msg): print(f"  ⚠ WARN  {msg}")


# ─────────────────────────────────────────────────────────────
#  載入工具
# ─────────────────────────────────────────────────────────────

def available_years(d: Path) -> list[int]:
    ys = []
    for p in sorted(d.glob("*.csv")):
        try:
            ys.append(int(p.stem))
        except ValueError:
            continue
    return ys


def load_year(d: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(d / f"{year}.csv", low_memory=False)
    df["年月日"] = pd.to_datetime(df["年月日"])
    df["證券代碼"] = df["證券代碼"].astype(str)
    return df


def load_window_stockmajor(d: Path, years: list[int], target: str) -> pd.DataFrame:
    """複刻 model.py._preprocess_window 的排序方式：stock-major。"""
    frames = [load_year(d, y) for y in years]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["證券代碼", "年月日"]).reset_index(drop=True)  # ← model.py 原樣
    return df.dropna(subset=[target])


# ─────────────────────────────────────────────────────────────
#  [T1] 環境
# ─────────────────────────────────────────────────────────────

def t1_env(make_dir: Path):
    _hr("[T1] 環境 / 路徑檢查")
    for label, p in [
        ("database_make/ (預計算特徵, main.py 來源)", make_dir),
        ("database/ (db_loader 來源)", Path("database")),
        ("database/raw_data/ (make.py SRC_DIR)", Path("database/raw_data")),
        ("database/features_only.csv (model.py 特徵清單)",
         Path("database/features_only.csv")),
    ]:
        print(f"  {'存在' if p.exists() else '不存在':<4}  {label}  ->  {p}")

    ys = available_years(make_dir)
    if not ys:
        _bad(f"{make_dir} 下沒有任何 YYYY.csv，後續測試無法進行。")
        return None
    print(f"\n  database_make/ 可用年份: {ys}")
    return ys


# ─────────────────────────────────────────────────────────────
#  [T2] TimeSeriesSplit 排序 bug（核心）
# ─────────────────────────────────────────────────────────────

def _analyze_folds(X: pd.DataFrame, dates: pd.Series, codes: pd.Series,
                   n_splits: int, tag: str):
    """印出每折 train/valid 的日期範圍與股票交集，回傳是否為「時間可分離」。"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    print(f"\n  ── {tag} ──")
    print(f"  {'fold':<5}{'train日期':<25}{'valid日期':<25}"
          f"{'時間可分?':<9}{'valid新股票%':<12}")
    time_separable_all = True
    for i, (tr, va) in enumerate(tscv.split(X), 1):
        tr_d, va_d = dates.iloc[tr], dates.iloc[va]
        tr_c, va_c = set(codes.iloc[tr]), set(codes.iloc[va])
        # valid 日期是否整體晚於 train（時間切的特徵）
        separable = tr_d.max() <= va_d.min()
        time_separable_all &= separable
        # valid 中完全沒在 train 出現過的股票比例（股票切的特徵）
        new_stock_rows = codes.iloc[va].isin(tr_c).eq(False).mean()
        print(f"  {i:<5}"
              f"{str(tr_d.min().date())+'~'+str(tr_d.max().date()):<25}"
              f"{str(va_d.min().date())+'~'+str(va_d.max().date()):<25}"
              f"{'是' if separable else '否':<9}"
              f"{new_stock_rows:>9.1%}")
    return time_separable_all


def t2_timeseries_split(make_dir: Path, years: list[int], target: str,
                        n_splits: int = 3):
    _hr("[T2] TimeSeriesSplit 排序 bug 驗證（核心）")

    # 找一組連續 3 年當作一個 window（train 前兩年，沿用 model.py）
    win = None
    for s in years:
        if s + 1 in years and s + 2 in years:
            win = (s, s + 1, s + 2)
            break
    if win is None:
        _warn("找不到連續 3 年，跳過 T2。")
        return
    train_years = [win[0], win[1]]
    print(f"  使用 window {win}，train_years={train_years}（複刻 model.py 切法）")

    full = load_window_stockmajor(make_dir, list(win), target)
    yr = full["年月日"].dt.year
    train_df = full[yr.isin(train_years)].copy()  # ← 保留 stock-major 順序
    if train_df.empty:
        _bad("train_df 為空，無法測試。")
        return

    print(f"  train_df 形狀: {train_df.shape}  "
          f"（前 5 列證券代碼: {train_df['證券代碼'].head().tolist()}）")

    # ① 現況：stock-major（model.py 原樣）
    sep_now = _analyze_folds(
        train_df.reset_index(drop=True),
        train_df["年月日"].reset_index(drop=True),
        train_df["證券代碼"].reset_index(drop=True),
        n_splits, tag="現況 model.py（sort by 證券代碼,年月日）",
    )

    # ② 修正：先 sort by 年月日
    fixed = train_df.sort_values("年月日").reset_index(drop=True)
    sep_fix = _analyze_folds(
        fixed, fixed["年月日"], fixed["證券代碼"],
        n_splits, tag="修正後（sort by 年月日）",
    )

    print()
    if not sep_now and sep_fix:
        _bad("確認 bug：現況 CV 折『時間不可分離』(folds 跨整段日期) → "
             "TimeSeriesSplit 實際在按股票分塊，非按時間切。")
        _ok("修正後 CV 折時間可分離，符合 walk-forward 預期。")
    elif sep_now:
        _ok("現況 CV 折已時間可分離（你的資料可能本來就近似時間排序，"
            "但仍建議顯式 sort by 年月日 以保險）。")
    else:
        _warn("現況與修正後都不完全可分離（panel 同日多股票會在邊界輕微重疊，屬正常）。"
              "重點看『valid新股票%』：現況若偏高即為股票分塊。")


# ─────────────────────────────────────────────────────────────
#  [T3] 邊界 label NaN
# ─────────────────────────────────────────────────────────────

def t3_boundary_label(make_dir: Path, years: list[int], target: str):
    _hr("[T3] 每年最後交易日 label NaN 檢查")
    print(f"  {'年份':<6}{'最後交易日':<14}{'當日列數':<10}"
          f"{'label NaN 列數':<14}{'NaN%':<8}")
    any_issue = False
    for y in years:
        df = load_year(make_dir, y)
        if target not in df.columns:
            _warn(f"{y}: 無 {target} 欄位，跳過")
            continue
        last_day = df["年月日"].max()
        last_rows = df[df["年月日"] == last_day]
        nan_n = last_rows[target].isna().sum()
        nan_pct = nan_n / max(len(last_rows), 1)
        if nan_pct > 0.5:
            any_issue = True
        print(f"  {y:<6}{str(last_day.date()):<14}{len(last_rows):<10}"
              f"{nan_n:<14}{nan_pct:>6.1%}")
    print()
    if any_issue:
        _warn("每年最後交易日的 label 大量 NaN（forward return 的 T+1 落在隔年檔案）。"
              "這些列會被 dropna 丟掉，屬已知邊界損耗，確認可接受即可。")
    else:
        _ok("邊界 label NaN 比例低，無明顯問題。")


# ─────────────────────────────────────────────────────────────
#  [T4] walk-forward window 重疊
# ─────────────────────────────────────────────────────────────

def t4_window_overlap(years: list[int]):
    _hr("[T4] Walk-Forward window train/test 重疊檢查")
    start, end = min(years), max(years)
    windows = [(s, s + 2) for s in range(start, end - 1)]  # 複刻 model.py.windows()
    print(f"  {'window':<14}{'train_years':<16}{'test_year':<10}{'重疊?':<6}")
    bad = False
    for (s, e) in windows:
        tr = [s, s + 1]
        te = e
        overlap = te in tr
        bad |= overlap
        print(f"  {f'{s}_{e}':<14}{str(tr):<16}{te:<10}{'是' if overlap else '否':<6}")
    print()
    _bad("有 train/test 年份重疊！") if bad else _ok("所有 window 的 train/test 年份不重疊。")


# ─────────────────────────────────────────────────────────────
#  [T5] look-ahead 欄位掃描
# ─────────────────────────────────────────────────────────────

def t5_lookahead(make_dir: Path, years: list[int], target: str):
    _hr("[T5] Look-ahead 前視欄位掃描")
    df = load_year(make_dir, years[-1])
    cols = list(df.columns)
    suspects = [c for c in cols
                if any(s.lower() in c.lower() for s in LOOKAHEAD_SUSPECTS)
                and c not in LABEL_COLS]
    print(f"  欄位總數: {len(cols)}")
    if suspects:
        _bad(f"發現疑似前視特徵（應為 label 才可前視）: {suspects}")
    else:
        _ok("未發現 *_fwd / future / next_ 等疑似前視特徵。")
    # 額外：label 是否真的存在
    present = [c for c in LABEL_COLS if c in cols]
    print(f"  偵測到的 label 欄位: {present}")
    if target not in cols:
        _bad(f"指定 target '{target}' 不在欄位中！可用 label: {present}")


# ─────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="database_make")
    ap.add_argument("--target", default="excess_return_tick")
    ap.add_argument("--n_splits", type=int, default=3)
    args = ap.parse_args()

    make_dir = Path(args.dir)
    print(f"診斷目錄: {make_dir.resolve()}")
    print(f"target  : {args.target}")

    ys = t1_env(make_dir)
    if not ys:
        return
    t5_lookahead(make_dir, ys, args.target)
    t2_timeseries_split(make_dir, ys, args.target, n_splits=args.n_splits)
    t3_boundary_label(make_dir, ys, args.target)
    t4_window_overlap(ys)

    _hr("完成")
    print("  把以上完整輸出貼回來即可。")


if __name__ == "__main__":
    main()