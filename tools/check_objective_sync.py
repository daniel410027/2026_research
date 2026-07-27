"""
check_objective_sync.py
=======================
對拍 `2026_research/objective/` 與 `2026_daily/objective/`，只報**真落差**。

★ 2026-07-27 新增。兩份 summary 都寫了同一句話：兩邊 objective/ 是「人工同步的
兩份副本，沒有任何自動檢查，而且已經證明會漏」——`min_data_in_leaf` 上界 100→600
就漏了三天（research §7.1）。這支就是補這個洞。

比對方式是 **AST 等價**，不是 byte 等價：
  - 註解與 docstring 一律忽略。兩邊的 provenance 註解措辭本來就不同
    （「2026-07-24（2026_research 移植）」vs「2026-07-22」），那不是落差，
    byte diff 會被這種噪音淹掉。
  - 逐個 top-level 符號（函式 / class / 賦值）比對，差異報到符號層級，
    直接看得出是哪一個常數或哪一個函式不同步。

刻意分歧走 `INTENTIONAL` 白名單，每一條都要寫理由——白名單是「已經想過並決定
維持不同」的紀錄，不是「懶得處理」的垃圾桶。白名單命中會印出來（不是靜默略過），
所以理由過期時看得到。

用法：
    .venv/bin/python tools/check_objective_sync.py
    .venv/bin/python tools/check_objective_sync.py --daily /path/to/2026_daily

exit code：0 = 無真落差（可能有白名單命中）；1 = 有真落差。
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DAILY = RESEARCH_ROOT.parent / "2026_daily"

# ─── 只在單邊存在的檔案：不是落差 ──────────────────────────────────
RESEARCH_ONLY = {
    "research_config.py": "research 專屬的 walk-forward 設定（daily 的對應物是 "
                          "daily_model/model_config.py，不在 objective/ 底下）",
}
DAILY_ONLY: dict[str, str] = {}

# ─── 刻意分歧：{檔名: {符號: 理由}} ────────────────────────────────
INTENTIONAL = {
    "model.py": {
        "_EXCLUDE_PREFIXES":
            "research 用 ('clu_id',) 前綴，daily 刻意留空。research 吃跨期資料"
            "（clu_id_local 與 clu_id_daily 同時存在過），daily 的 database_make/ "
            "每年只有 clu_id_daily，精確欄名已足夠。見 daily summary §8。",
    },
    "preprocess.py": {
        "DataPreprocessor":
            "delay1 微結構因子的接線位置不同：research 掛在 preprocess.py 的管線"
            "步驟，daily 在 make_new.py 內接線（兩邊入口不同）。因子本身"
            "（features/price_return.py）byte 一致。見 daily summary §8。",
    },
}


def _strip_docstrings(node: ast.AST) -> ast.AST:
    """就地移除 Module / ClassDef / FunctionDef 的 docstring（比對時視為註解）。"""
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Module, ast.ClassDef,
                            ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(sub, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                sub.body = body[1:]
    return node


def _symbol_name(stmt: ast.stmt) -> str:
    """top-level 陳述式的代表名稱。"""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return stmt.name
    if isinstance(stmt, ast.Assign):
        names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        return names[0] if names else "<assign>"
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return "<imports>"
    return f"<{type(stmt).__name__}>"


def symbol_map(path: Path) -> dict[str, str]:
    """{top-level 符號名: AST dump}，docstring 已剝除。import 合併成一個符號。"""
    tree = _strip_docstrings(ast.parse(path.read_text(encoding="utf-8")))
    out: dict[str, str] = {}
    imports: list[str] = []
    for stmt in tree.body:
        name = _symbol_name(stmt)
        dump = ast.dump(stmt, annotate_fields=True)
        if name == "<imports>":
            imports.append(dump)
        else:
            # 同名重複定義（少見）串起來，避免後者蓋掉前者
            out[name] = out.get(name, "") + dump
    if imports:
        out["<imports>"] = "\n".join(imports)
    return out


def compare(research_dir: Path, daily_dir: Path) -> int:
    r_files = {p.name for p in research_dir.glob("*.py")}
    d_files = {p.name for p in daily_dir.glob("*.py")}

    real_gaps: list[str] = []
    notes:     list[str] = []

    for name in sorted(r_files - d_files):
        if name in RESEARCH_ONLY:
            notes.append(f"[僅 research] {name} — {RESEARCH_ONLY[name]}")
        else:
            real_gaps.append(f"[僅 research] {name} — daily 沒有這個檔，未列入白名單")
    for name in sorted(d_files - r_files):
        if name in DAILY_ONLY:
            notes.append(f"[僅 daily] {name} — {DAILY_ONLY[name]}")
        else:
            real_gaps.append(f"[僅 daily] {name} — research 沒有這個檔，未列入白名單")

    for name in sorted(r_files & d_files):
        r_path, d_path = research_dir / name, daily_dir / name
        if r_path.read_bytes() == d_path.read_bytes():
            continue
        try:
            r_syms, d_syms = symbol_map(r_path), symbol_map(d_path)
        except SyntaxError as e:
            real_gaps.append(f"[{name}] 無法解析：{e}")
            continue

        allow = INTENTIONAL.get(name, {})
        diffs = []
        for sym in sorted(set(r_syms) | set(d_syms)):
            if r_syms.get(sym) == d_syms.get(sym):
                continue
            if sym not in r_syms:
                diffs.append((sym, "僅 daily 有"))
            elif sym not in d_syms:
                diffs.append((sym, "僅 research 有"))
            else:
                diffs.append((sym, "兩邊內容不同"))

        if not diffs:
            notes.append(f"[{name}] 僅註解／docstring 措辭不同，程式碼等價")
            continue

        for sym, kind in diffs:
            if sym in allow:
                notes.append(f"[{name}] {sym}（{kind}）— 刻意分歧：{allow[sym]}")
            else:
                real_gaps.append(f"[{name}] {sym}（{kind}）")

    print("=" * 70)
    print(f"  objective/ 對拍")
    print(f"  research: {research_dir}")
    print(f"  daily   : {daily_dir}")
    print("=" * 70)

    if notes:
        print("\n▸ 已知/可接受（白名單命中，理由過期時請回頭改白名單）：")
        for n in notes:
            print(f"    · {n}")

    if real_gaps:
        print(f"\n✗ 真落差 {len(real_gaps)} 項——需要決定往哪邊同步，"
              f"或補進 INTENTIONAL 白名單並寫下理由：")
        for g in real_gaps:
            print(f"    ! {g}")
        print("\n  ⚠ 移植時做定點搬移，不要整包覆蓋（research §7.1）。")
        return 1

    print("\n✓ 無真落差。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--research", type=Path, default=RESEARCH_ROOT / "objective")
    ap.add_argument("--daily", type=Path, default=DEFAULT_DAILY / "objective")
    args = ap.parse_args()

    for label, d in (("research", args.research), ("daily", args.daily)):
        if not d.is_dir():
            print(f"✗ 找不到 {label} 的 objective/：{d}", file=sys.stderr)
            return 2
    return compare(args.research.resolve(), args.daily.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
