"""
add_ohlcv.py
============
將 OHLCV 五欄（開盤價元、最高價元、最低價元、收盤價元、成交量千股）
合併至 database/experiment/*/predictions.csv。

流程：
    1. 掃描 database/experiment/ 下所有子資料夾
    2. 從資料夾名稱（YYYY_YYYY）解析 test year（末尾年份）
    3. 讀取該年度的原始 TEJ 季度 CSV，解析並標準化 OHLCV 欄位
    4. Left-join 至 predictions.csv（key: 證券代碼 + 年月日）
    5. 原地覆寫存檔

作者：Daniel Huang
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
#  設定
# ============================================================

EXPERIMENT_DIR = Path("database/experiment")
RAW_DATA_DIR   = Path("database/raw_data")

# 標準化後的 OHLCV 欄位名稱（與 preprocess.py 一致）
OHLCV_COLS = ["開盤價元", "最高價元", "最低價元", "收盤價元", "成交量千股"]

# TEJ 收盤價與成交量的備用欄位對照（同 preprocess._resolve_ohlcv_columns）
OHLCV_CANDIDATES: dict[str, list[str]] = {
    "收盤價元":   ["未調整收盤價元", "收盤價元", "當日收盤"],
    "成交量千股": ["成交量千股1",    "成交量千股"],
}


# ============================================================
#  工具函式
# ============================================================

def _read_csv(path: Path) -> pd.DataFrame:
    """嘗試多種編碼讀取 TEJ Tab 分隔 CSV。"""
    for enc in ["utf-8", "utf-8-sig", "utf-16", "utf-16-sig", "big5", "gbk", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc, sep="\t", low_memory=False)
        except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError):
            continue
    raise ValueError(f"無法讀取 {path}")


def _normalize_col(name: str) -> str:
    """移除欄位名稱中的特殊字元（與 preprocess._normalize_column_names 一致）。"""
    return re.sub(r"[^\w\u4e00-\u9fff]", "", name)


def load_raw_ohlcv(year: int) -> pd.DataFrame:
    """
    讀取指定年度的四季 raw CSV，合併並解析出標準 OHLCV 五欄。

    Returns
    -------
    pd.DataFrame
        columns: 證券代碼, 年月日, 開盤價元, 最高價元, 最低價元, 收盤價元, 成交量千股
    """
    dfs = []
    for q in range(1, 5):
        path = RAW_DATA_DIR / f"{year}{q}.csv"
        if path.exists():
            df = _read_csv(path)
            # 正規化欄位名稱
            df.columns = [_normalize_col(c) for c in df.columns]
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"找不到 {year} 年任何季度 CSV（{RAW_DATA_DIR}）")

    raw = pd.concat(dfs, ignore_index=True)
    raw = raw.drop_duplicates(subset=["年月日", "證券代碼"], keep="last")

    # ── 解析備用欄位（收盤價元、成交量千股）───────────────────
    for target, candidates in OHLCV_CANDIDATES.items():
        if target in raw.columns:
            raw[target] = pd.to_numeric(raw[target], errors="coerce")
        else:
            best_col, best_rate = None, 0.0
            for c in candidates:
                if c in raw.columns:
                    rate = pd.to_numeric(raw[c], errors="coerce").notna().mean()
                    if rate > best_rate:
                        best_rate, best_col = rate, c
            if best_col:
                raw[target] = pd.to_numeric(raw[best_col], errors="coerce")
            else:
                raw[target] = float("nan")

    # ── 開盤/最高/最低強制轉 numeric ──────────────────────────
    for col in ["開盤價元", "最高價元", "最低價元"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        else:
            raw[col] = float("nan")

    # ── 清理 key 欄位 ─────────────────────────────────────────
    raw["證券代碼"] = raw["證券代碼"].astype(str).str.extract(r"(\d{4})", expand=False)
    raw["年月日"]   = pd.to_datetime(raw["年月日"], format="%Y%m%d", errors="coerce")
    raw = raw.dropna(subset=["年月日", "證券代碼"])

    keep = ["證券代碼", "年月日"] + OHLCV_COLS
    available = [c for c in keep if c in raw.columns]
    return raw[available].copy()


# ============================================================
#  主流程
# ============================================================

def add_ohlcv_to_predictions():
    if not EXPERIMENT_DIR.exists():
        print(f"✗ 找不到資料夾：{EXPERIMENT_DIR}")
        return

    folders = sorted(
        p for p in EXPERIMENT_DIR.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{4}_\d{4}", p.name)
    )

    if not folders:
        print(f"✗ {EXPERIMENT_DIR} 下找不到 YYYY_YYYY 格式的子資料夾")
        return

    print(f"找到 {len(folders)} 個 window 資料夾\n{'─'*50}")

    for folder in folders:
        pred_path = folder / "predictions.csv"
        if not pred_path.exists():
            print(f"  [{folder.name}] ⚠ 找不到 predictions.csv，跳過")
            continue

        # 解析 test year（資料夾末尾年份，如 2016_2018 → 2018）
        test_year = int(folder.name.split("_")[1])

        print(f"  [{folder.name}] test_year={test_year}", end=" … ")

        # 讀取 predictions
        pred = pd.read_csv(pred_path, low_memory=False)
        pred["年月日"] = pd.to_datetime(pred["年月日"], errors="coerce")
        pred["證券代碼"] = pred["證券代碼"].astype(str)

        # 先移除已存在的 OHLCV 欄位（避免重複執行產生 _x/_y）
        pred = pred.drop(columns=[c for c in OHLCV_COLS if c in pred.columns])

        # 讀取 raw OHLCV
        try:
            ohlcv = load_raw_ohlcv(test_year)
        except FileNotFoundError as e:
            print(f"✗ {e}")
            continue

        ohlcv["證券代碼"] = ohlcv["證券代碼"].astype(str)

        # Left-join
        merged = pred.merge(ohlcv, on=["證券代碼", "年月日"], how="left")

        # 統計補齊率
        fill_rate = merged["收盤價元"].notna().mean() if "收盤價元" in merged.columns else 0.0

        # 覆寫存檔
        merged.to_csv(pred_path, index=False)
        print(f"✓  {len(merged):,} 筆，收盤價補齊率 {fill_rate:.1%}")

    print(f"\n{'─'*50}\n完成。")


if __name__ == "__main__":
    add_ohlcv_to_predictions()