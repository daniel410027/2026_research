import os
import zipfile
import tarfile
import gzip
import bz2
import lzma
import shutil
from pathlib import Path
import pandas as pd

try:
    import rarfile
except ImportError:
    rarfile = None

# unpack.py 在 project_root，輸入與輸出路徑
ROOT_DIR = Path(__file__).parent
INPUT_DIR = ROOT_DIR / "database" / "daily_data"
OUTPUT_DIR = ROOT_DIR / "database" / "daily_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_EXTENSIONS = ['.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz', '.gz', '.bz2', '.xz', '.rar']


# ── 編碼工具 ─────────────────────────────────────────────────────────────────

def _detect_encoding(file_path: Path) -> str:
    """偵測檔案編碼（BOM 優先），回傳可用的 encoding"""
    with open(file_path, 'rb') as f:
        raw = f.read(4)
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return 'utf-16'
    if raw[:3] == b'\xef\xbb\xbf':
        return 'utf-8-sig'
    return 'utf-8'  # 預設（GBK 已在解壓時轉換）


def _write_decoded(content: bytes, output_file: Path):
    """GBK → UTF-16 → UTF-8 fallback → binary 寫入，統一換行符為 LF"""
    for encoding in ('gbk', 'utf-16', 'utf-8'):
        try:
            text = content.decode(encoding).replace('\r\n', '\n').replace('\r', '\n')
            with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)
            return
        except UnicodeDecodeError:
            continue
    # 最後手段：直接寫 binary
    with open(output_file, 'wb') as f:
        f.write(content)
    print(f"⚠️ 以 binary 寫入: {output_file.name}，可能需要手動處理")


# ── 解壓縮 ───────────────────────────────────────────────────────────────────

def extract_single_file(file_path: Path):
    try:
        suffix = file_path.suffix.lower()
        output_file = OUTPUT_DIR / (file_path.stem + ".csv")

        # ✅ 已存在則跳過，不覆蓋
        if output_file.exists():
            print(f"⏭ 已存在，跳過: {output_file.name}")
            return

        print(f"🗂 解壓縮中: {file_path.name}")

        if suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                if len(members) != 1:
                    print(f"⚠️ 多於一個檔案，跳過: {file_path.name}")
                    return
                with zip_ref.open(members[0]) as source:
                    _write_decoded(source.read(), output_file)

        elif suffix in ['.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz']:
            with tarfile.open(file_path, 'r:*') as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
                if len(members) != 1:
                    print(f"⚠️ 多於一個檔案，跳過: {file_path.name}")
                    return
                _write_decoded(tar.extractfile(members[0]).read(), output_file)

        elif suffix == '.gz' and not file_path.name.endswith('.tar.gz'):
            with gzip.open(file_path, 'rb') as f_in:
                _write_decoded(f_in.read(), output_file)

        elif suffix == '.bz2' and not file_path.name.endswith('.tar.bz2'):
            with bz2.open(file_path, 'rb') as f_in:
                _write_decoded(f_in.read(), output_file)

        elif suffix == '.xz' and not file_path.name.endswith('.tar.xz'):
            with lzma.open(file_path, 'rb') as f_in:
                _write_decoded(f_in.read(), output_file)

        elif suffix == '.rar' and rarfile:
            with rarfile.RarFile(file_path) as rf:
                members = rf.namelist()
                if len(members) != 1:
                    print(f"⚠️ 多於一個檔案，跳過: {file_path.name}")
                    return
                with rf.open(members[0]) as source:
                    _write_decoded(source.read(), output_file)

        else:
            print(f"❌ 不支援格式: {file_path.name}")

    except Exception as e:
        print(f"❌ 解壓失敗: {file_path.name}，錯誤: {e}")


def extract_all_in_temp():
    for file in INPUT_DIR.iterdir():
        if file.is_file() and any(file.name.lower().endswith(ext) for ext in ARCHIVE_EXTENSIONS):
            extract_single_file(file)


# ── Header 修復 ──────────────────────────────────────────────────────────────

def detect_has_header(file_path: Path) -> bool:
    """
    TEJ 資料第二欄為日期 (YYYYMMDD，8位純數字) → 無 header
    有 header 時第二欄為中文欄位名稱 (e.g. '年月日')
    """
    enc = _detect_encoding(file_path)
    try:
        with open(file_path, encoding=enc, errors='replace') as f:
            first_line = f.readline().strip().split('\t')
        if len(first_line) < 2:
            return True
        second_col = first_line[1].strip()
        return not (second_col.isdigit() and len(second_col) == 8)
    except Exception:
        return True  # 讀取失敗，保守假設有 header


def fix_headers(output_dir: Path):
    csv_files = list(output_dir.glob("*.csv"))

    # 找第一份有 header 的 CSV 作為參考
    reference_header = None
    for f in csv_files:
        if detect_has_header(f):
            try:
                enc = _detect_encoding(f)
                reference_header = pd.read_csv(f, nrows=0, sep='\t', encoding=enc).columns.tolist()
                print(f"✅ 參考 header 來源: {f.name}，共 {len(reference_header)} 欄")
                break
            except Exception as e:
                print(f"⚠️ 無法讀取參考檔案 {f.name}：{e}")
                continue

    if reference_header is None:
        print("❌ 找不到任何帶 header 的 CSV，請手動指定")
        return

    for f in csv_files:
        if detect_has_header(f):
            continue  # 已有 header，跳過
        try:
            enc = _detect_encoding(f)
            df = pd.read_csv(f, header=None, sep='\t', encoding=enc)
            if df.shape[1] != len(reference_header):
                print(f"⚠️ 欄位數不符，跳過: {f.name}（{df.shape[1]} vs {len(reference_header)}）")
                continue
            df.columns = reference_header
            df.to_csv(f, index=False, sep='\t', encoding='utf-8', lineterminator='\n')
            print(f"🔧 補上 header: {f.name}")
        except Exception as e:
            print(f"❌ 處理失敗: {f.name}，錯誤: {e}")


# ── 主程式 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 開始解壓縮")
    print("=" * 50)
    extract_all_in_temp()

    print()
    print("=" * 50)
    print("🔍 開始修復 Header")
    print("=" * 50)
    fix_headers(OUTPUT_DIR)

    print()
    print("✅ 完成")