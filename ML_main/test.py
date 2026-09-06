"""
test.py
=======
Debug：確認 _remove_categorical_features 移除了哪些欄位
以及 TEJ CSV 是否有描述列問題

使用方式：
    python test.py
"""

from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT / "database/raw_data"

# ── 讀一個 CSV 看頭兩列 ─────────────────────────────────────────
def read_csv(file_path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "utf-16", "utf-16-sig", "big5", "gbk", "latin-1"]:
        try:
            return pd.read_csv(file_path, encoding=enc, sep="\t", low_memory=False)
        except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError):
            continue
    raise ValueError(f"無法讀取 {file_path}")


def test_raw_csv():
    """確認 TEJ CSV 頭兩列是否有描述列"""
    test_file = RAW_DATA_DIR / "20181.csv"
    if not test_file.exists():
        print(f"[SKIP] {test_file} 不存在")
        return

    df = read_csv(test_file)
    print("=" * 60)
    print("  [1] TEJ CSV 前 3 列原始資料")
    print("=" * 60)
    print(df.head(3).to_string())
    print(f"\n  欄位總數: {len(df.columns)}")
    print(f"  欄位（前 10）: {df.columns[:10].tolist()}")


def test_categorical_removal():
    """模擬 _remove_categorical_features，印出會被移除的欄位"""
    test_file = RAW_DATA_DIR / "20181.csv"
    if not test_file.exists():
        print(f"[SKIP] {test_file} 不存在")
        return

    df = read_csv(test_file)

    # 正規化欄位名稱（同 preprocess.py）
    df.columns = [re.sub(r"[^\w\u4e00-\u9fff]", "", c) for c in df.columns]
    KEY_COLS = ["證券代碼", "年月日"]

    numeric_cols = set(df.select_dtypes(include=["number"]).columns)
    keep = set(KEY_COLS) | numeric_cols | {"證券代碼"}
    remove = [c for c in df.columns if c not in keep]

    print("\n" + "=" * 60)
    print("  [2] _remove_categorical_features 會移除的欄位")
    print("=" * 60)
    for col in remove:
        sample = df[col].dropna().head(3).tolist()
        print(f"  - {col!r:30s} sample: {sample}")

    print(f"\n  共 {len(remove)} 個欄位會被移除")


def test_key_cols_dtype():
    """確認 年月日、證券代碼 的型別是否正常"""
    test_file = RAW_DATA_DIR / "20181.csv"
    if not test_file.exists():
        print(f"[SKIP] {test_file} 不存在")
        return

    df = read_csv(test_file)
    df.columns = [re.sub(r"[^\w\u4e00-\u9fff]", "", c) for c in df.columns]

    print("\n" + "=" * 60)
    print("  [3] 關鍵欄位型別與樣本值")
    print("=" * 60)
    for col in ["證券代碼", "年月日"]:
        if col in df.columns:
            print(f"  {col}: dtype={df[col].dtype}, sample={df[col].head(3).tolist()}")
        else:
            print(f"  ✗ 欄位 '{col}' 不存在")


if __name__ == "__main__":
    test_raw_csv()
    test_categorical_removal()
    test_key_cols_dtype()