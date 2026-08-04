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

★ 2026-08-03 新增 `--data`：**對拍程式碼只做完一半，另一半是資料。**
2026-07-31 這支報「✓ 無真落差」的同時，research 的 `database_make/` 比 daily 少了
6 個因子欄、多了一個已被取代的 `報酬率1`，落後九天（§7.1 的 ⚠）。特徵工程改動要
落地到 research，除了同步 `objective/` 還得把 daily 重生後的 `database_make/*.csv`
搬過來。`--data` 就是把 §7.1 那段 `head -1 | sort | diff` 表頭比對收進工具裡——
三秒的事，但它會在你不記得的時候救你一次。

★ 2026-08-04 新增 `--config`：**第三個洞是設定。**
2026-08-03 daily 把每日排程的訓練視窗從 2 年改成 4 年（commit 823be5a），母體
這邊還是 2 年，而 `--code` 與 `--data` **兩盞燈都是綠的**——因為 `train_years`
住在 `daily_model/model_config.py` 與 `objective/research_config.py`，這兩個檔
本來就刻意在 objective/ 對拍範圍外（見 `RESEARCH_ONLY`）。同一個教訓的第三次：
**綠燈只覆蓋它檢查的東西**（§9-#14）。

`--config` 比的是**語意鍵**不是檔案：兩邊的設定長在不同檔案、不同命名風格
（`TRAIN_YEARS_N` vs `train_years`）、甚至不同層（daily 的 dataclass 欄位是
`None` 時要回退到模組層常數），所以逐鍵指定來源，比出來的差異報成
「這個鍵兩邊不同」而不是「這兩個檔不一樣」。

抽值一律走 AST **不 import**：import daily 的 `daily_model/` 會連帶拉進
FinLab / 憑證那一串，對拍工具不該有那種相依。代價是只認得字面值，
`field(default_factory=...)` 之類會標成 `<unresolved>` 並當成真落差報出來
（寧可誤報也不要靜默略過）。

用法：
    .venv/bin/python tools/check_objective_sync.py
    .venv/bin/python tools/check_objective_sync.py --daily /path/to/2026_daily
    .venv/bin/python tools/check_objective_sync.py --data            # 只比資料表頭
    .venv/bin/python tools/check_objective_sync.py --config          # 只比訓練設定
    .venv/bin/python tools/check_objective_sync.py --code --data --config   # 三者都比

不給任何旗標時預設只比程式碼（維持既有行為與 exit code 語意）。

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


# ─── 設定對拍：刻意分歧的語意鍵 ──────────────────────────────────
INTENTIONAL_CONFIG: dict[str, str] = {
    "train_year_offset":
        "research 專屬的控制組旋鈕（把訓練視窗整體往前推 N 年，樣本數不變、"
        "只讓資料變舊）。daily 的視窗永遠緊貼最新交易日，沒有對應概念，"
        "所以 daily 側固定視為 0——只要 research 也是 0 就不算落差。",
    "decay_half_life_months":
        "指數時間衰減樣本權重，research 專屬。1107_window_length 實測比不加更差，"
        "留著只為可複現該負面結論（§9-#18），daily 沒有也不該有。",
}

# ─── 值相同、語意不同：比得出「一致」但不代表真的一樣 ───────────
# 這些**一律印出來**（不管有沒有差異），因為它們是這支工具能力的邊界。
CONFIG_CAVEATS: dict[str, str] = {
    "train_years":
        "兩邊的「N 年」切法不同，長度已對齊、對齊點沒有。\n"
        "      research：完整日曆年，train=[Y-N .. Y-1]、test=Y，長度恆為 N 年，\n"
        "                每年跳一次（walk-forward 的折是年）。\n"
        "      daily   ：以預測日往回推足月 N 年當下界，長度同樣恆為 N 年，但**每天**\n"
        "                平滑滾動（✅ 2026-08-04 修正；在那之前下界是所讀 CSV 起始年的\n"
        "                1/1，實際長度在 N.0 ~ N+1.0 年鋸齒擺盪，例如 train_years=4、\n"
        "                預測日 2026-08-04 實際跨約 4.6 年）。\n"
        "      → 現在「4 年」兩邊是同一個長度；殘留差異只在切點（年 vs 日），\n"
        "        比較 IC 時仍要記得 research 的 test 是整年、daily 是逐日。",
}

# 排除欄集合的逐欄白名單（key 為欄名）
INTENTIONAL_EXCLUDE_COLS: dict[str, str] = {
    "market_return":
        "§7.2 的刻意分歧：research 排除（per-day constant，疑為日期指紋），"
        "daily 判斷無洩漏疑慮故保留。**分歧存在期間兩邊訓練特徵集不同，"
        "IC/RMSE 不可互相比較**——這條白名單不是說它沒影響，是說已經想過。",
    "market_index":
        "純防禦性條目：2026-08-04 查過 database_make/2025.csv 表頭，這個欄位"
        "**根本不存在**，兩邊實際訓練特徵集不受影響。research 留著是為了"
        "「之後若資料重生帶回這欄」的雙重防護。",
    "clu_id_local":
        "同上，欄位已不在資料裡（2024 年起改名 clu_id_daily）。research 吃跨期"
        "資料所以兩個名字都列，daily 每年只有 clu_id_daily。實際防護是"
        "research 的 _EXCLUDE_PREFIXES=('clu_id',) 前綴比對。",
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


def _header(path: Path) -> list[str]:
    """讀 CSV 第一行的欄名（等同 `head -1 | tr ',' '\\n'`）。"""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        line = f.readline()
    return [c.strip() for c in line.rstrip("\r\n").split(",") if c.strip()]


def compare_data(research_dir: Path, daily_dir: Path) -> int:
    """對拍兩邊 database_make/*.csv 的**表頭**（§7.1 的 head -1 | sort | diff）。

    只比欄名集合，不比欄序、不比列數：欄序對訓練沒有意義（trainer 是照欄名取），
    列數兩邊本來就不同（research 吃跨期、daily 只到最新交易日）。真正會讓兩邊
    IC/RMSE 不可比的是**特徵集不同**，那就是表頭差異。
    """
    r_files = {p.name for p in research_dir.glob("*.csv")}
    d_files = {p.name for p in daily_dir.glob("*.csv")}

    real_gaps: list[str] = []

    for name in sorted(r_files - d_files):
        real_gaps.append(f"[僅 research] {name}")
    for name in sorted(d_files - r_files):
        real_gaps.append(f"[僅 daily] {name}")

    checked = 0
    for name in sorted(r_files & d_files):
        try:
            r_cols, d_cols = _header(research_dir / name), _header(daily_dir / name)
        except OSError as e:
            real_gaps.append(f"[{name}] 無法讀取：{e}")
            continue
        checked += 1
        only_r = sorted(set(r_cols) - set(d_cols))
        only_d = sorted(set(d_cols) - set(r_cols))
        if not only_r and not only_d:
            continue
        if only_d:
            real_gaps.append(
                f"[{name}] research 少了 {len(only_d)} 欄（daily 有、research 沒有）："
                f"{', '.join(only_d)}"
            )
        if only_r:
            real_gaps.append(
                f"[{name}] research 多了 {len(only_r)} 欄（可能是已被取代的舊欄）："
                f"{', '.join(only_r)}"
            )

    print("=" * 70)
    print(f"  database_make/ 表頭對拍")
    print(f"  research: {research_dir}")
    print(f"  daily   : {daily_dir}")
    print("=" * 70)

    if real_gaps:
        print(f"\n✗ 資料落差 {len(real_gaps)} 項——特徵集不同，"
              f"兩邊的 IC/RMSE **不能互相比較**：")
        for g in real_gaps:
            print(f"    ! {g}")
        print("\n  ⚠ 修法：daily 端跑 make_new.py 重生 database_make/，"
              "再把 *.csv 搬到 research（research §7.1）。")
        return 1

    print(f"\n✓ {checked} 個檔案表頭一致。")
    return 0


# ─────────────────────────────────────────────────────────────
#  設定對拍（--config）
# ─────────────────────────────────────────────────────────────

UNRESOLVED = "<unresolved>"


def _literal(node: ast.expr):
    """字面值 → Python 值；非字面（呼叫、f-string、field(...)）→ UNRESOLVED。"""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return UNRESOLVED


def _assign_pairs(body: list[ast.stmt]) -> dict:
    """一段 body 裡的 `NAME = 字面值` / `NAME: T = 字面值`。"""
    out = {}
    for s in body:
        if (isinstance(s, ast.Assign) and len(s.targets) == 1
                and isinstance(s.targets[0], ast.Name)):
            out[s.targets[0].id] = _literal(s.value)
        elif (isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
                and s.value is not None):
            out[s.target.id] = _literal(s.value)
    return out


def module_consts(path: Path) -> dict:
    """模組層常數。"""
    return _assign_pairs(ast.parse(path.read_text(encoding="utf-8")).body)


def class_attrs(path: Path, class_name: str) -> dict:
    """指定 class 的類別層屬性（dataclass 欄位預設值也走這裡）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return _assign_pairs(node.body)
    return {}


def _config_pairs(research_root: Path, daily_root: Path) -> tuple[list, list[str]]:
    """
    回傳 (逐鍵比對清單, 讀檔錯誤清單)。
    每一項是 (語意鍵, research 值, daily 值)。
    """
    errors: list[str] = []
    paths = {
        "r_cfg":   research_root / "objective" / "research_config.py",
        "r_model": research_root / "objective" / "model.py",
        "d_cfg":   daily_root / "daily_model" / "model_config.py",
    }
    for label, p in paths.items():
        if not p.exists():
            errors.append(f"找不到設定來源：{p}")
    if errors:
        return [], errors

    R  = class_attrs(paths["r_cfg"], "RunConfig")
    Rm = module_consts(paths["r_model"])
    Rw = class_attrs(paths["r_model"], "WalkForwardConfig")
    D  = class_attrs(paths["d_cfg"], "DailyConfig")
    Dm = module_consts(paths["d_cfg"])

    def Rm_default(name: str):
        """research 端沒有經 RunConfig 覆寫的項目 → WalkForwardConfig 的預設值。"""
        return Rw.get(name)

    def d_field(name: str, module_fallback: str):
        """daily dataclass 欄位；None = 「用模組層預設」，故回退。"""
        v = D.get(name)
        return Dm.get(module_fallback) if v is None else v

    r_exclude = set(Rm.get("_EXCLUDE_COLS") or ()) | set(R.get("EXTRA_EXCLUDE_COLS") or ())
    d_exclude = set(Dm.get("_EXCLUDE_COLS") or ())

    pairs = [
        # 決定「訓練矩陣長什麼樣」的那幾項——兩邊不同 = IC/RMSE 不可比
        ("train_years",            R.get("TRAIN_YEARS_N"),          D.get("train_years")),
        ("train_year_offset",      R.get("TRAIN_YEAR_OFFSET"),      0),
        ("decay_half_life_months", R.get("DECAY_HALF_LIFE_MONTHS"), None),
        ("target_col",             R.get("LABEL"),                  D.get("target_col")),
        ("use_return_weight",      R.get("USE_RETURN_WEIGHT"),      D.get("use_return_weight")),
        ("use_vol_weight",         R.get("USE_VOL_WEIGHT"),         D.get("use_vol_weight")),
        ("vol_weight_col",         Rm_default("vol_weight_col"),    D.get("vol_weight_col")),
        ("liq_filter_enabled",     R.get("LIQ_FILTER_ENABLED"),     d_field("liq_filter_enabled", "LIQ_FILTER_ENABLED")),
        ("liq_w1",                 R.get("LIQ_W1"),                 d_field("liq_w1", "LIQ_W1")),
        ("liq_w2",                 R.get("LIQ_W2"),                 d_field("liq_w2", "LIQ_W2")),
        ("liq_keep_ratio",         R.get("LIQ_KEEP_RATIO"),         d_field("liq_keep_ratio", "LIQ_KEEP_RATIO")),
        ("keep_nan_prefixes",      Rm.get("_KEEP_NAN_PREFIXES"),    Dm.get("_KEEP_NAN_PREFIXES")),
        ("exclude_cols",           r_exclude,                       d_exclude),
    ]
    return pairs, errors


def compare_config(research_root: Path, daily_root: Path) -> int:
    """
    對拍**訓練設定**：兩邊各自的設定檔不在 objective/ 對拍範圍內，
    train_years 這種決定線上行為的東西正是從這個縫隙漏掉的（2026-08-03）。
    """
    print("=" * 70)
    print("  訓練設定對拍")
    print(f"  research: {research_root / 'objective' / 'research_config.py'}")
    print(f"  daily   : {daily_root / 'daily_model' / 'model_config.py'}")
    print("=" * 70)

    pairs, errors = _config_pairs(research_root, daily_root)
    if errors:
        for e in errors:
            print(f"    ! {e}")
        return 1

    real_gaps: list[str] = []
    notes:     list[str] = []
    checked = 0

    for key, r_val, d_val in pairs:
        checked += 1
        if key == "exclude_cols":
            only_r = sorted(r_val - d_val)
            only_d = sorted(d_val - r_val)
            for col in only_r + only_d:
                side = "僅 research 排除" if col in only_r else "僅 daily 排除"
                if col in INTENTIONAL_EXCLUDE_COLS:
                    notes.append(f"[exclude_cols] {col}（{side}）— 刻意分歧："
                                 f"{INTENTIONAL_EXCLUDE_COLS[col]}")
                else:
                    real_gaps.append(f"[exclude_cols] {col}（{side}）")
            continue

        if r_val == d_val and UNRESOLVED not in (r_val, d_val):
            continue
        if key in INTENTIONAL_CONFIG:
            notes.append(f"[{key}] research={r_val!r} / daily={d_val!r} — "
                         f"刻意分歧：{INTENTIONAL_CONFIG[key]}")
        else:
            real_gaps.append(f"[{key}] research={r_val!r}  ≠  daily={d_val!r}")

    if notes:
        print("\n▸ 已知/可接受（白名單命中，理由過期時請回頭改白名單）：")
        for n in notes:
            print(f"    · {n}")

    if CONFIG_CAVEATS:
        print("\n▸ ⚠ 值相同 ≠ 語意相同（本工具能力邊界，一律提醒）：")
        for key, text in CONFIG_CAVEATS.items():
            print(f"    · [{key}] {text}")

    if real_gaps:
        print(f"\n✗ 設定落差 {len(real_gaps)} 項——兩邊訓練矩陣不同，"
              f"**IC/RMSE 不能互相比較**：")
        for g in real_gaps:
            print(f"    ! {g}")
        print("\n  ⚠ 修法：決定哪一邊是對的再改，不要為了讓燈變綠而抄。"
              "\n    線上行為以 daily 為準；research 要當「線上模型的離線分身」，"
              "\n    這幾項就必須跟著 daily 走（見 summary_research.md §7.1）。")
        return 1

    print(f"\n✓ {checked} 個語意鍵一致。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--research", type=Path, default=RESEARCH_ROOT / "objective")
    ap.add_argument("--daily", type=Path, default=DEFAULT_DAILY / "objective")
    ap.add_argument("--research-data", type=Path,
                    default=RESEARCH_ROOT / "database_make")
    ap.add_argument("--daily-data", type=Path,
                    default=DEFAULT_DAILY / "database_make")
    ap.add_argument("--code", action="store_true",
                    help="對拍 objective/ 程式碼（未指定任何旗標時的預設）")
    ap.add_argument("--data", action="store_true",
                    help="對拍 database_make/ 的 CSV 表頭")
    ap.add_argument("--config", action="store_true",
                    help="對拍訓練設定（research_config.py vs daily_model/model_config.py）")
    ap.add_argument("--daily-root", type=Path, default=DEFAULT_DAILY,
                    help="2026_daily 根目錄（--config 用，預設 ../2026_daily）")
    args = ap.parse_args()

    # 全都沒給 → 維持既有預設行為（只比程式碼）
    do_code, do_data, do_config = args.code, args.data, args.config
    if not (do_code or do_data or do_config):
        do_code = True

    rc = 0
    if do_code:
        for label, d in (("research", args.research), ("daily", args.daily)):
            if not d.is_dir():
                print(f"✗ 找不到 {label} 的 objective/：{d}", file=sys.stderr)
                return 2
        rc |= compare(args.research.resolve(), args.daily.resolve())

    if do_data:
        if do_code:
            print()
        for label, d in (("research", args.research_data),
                         ("daily", args.daily_data)):
            if not d.is_dir():
                print(f"✗ 找不到 {label} 的 database_make/：{d}", file=sys.stderr)
                return 2
        rc |= compare_data(args.research_data.resolve(), args.daily_data.resolve())

    if do_config:
        if do_code or do_data:
            print()
        if not args.daily_root.is_dir():
            print(f"✗ 找不到 daily 根目錄：{args.daily_root}", file=sys.stderr)
            return 2
        rc |= compare_config(RESEARCH_ROOT.resolve(), args.daily_root.resolve())

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
