"""
regression_check_window.py
==========================
回歸驗證：確認「訓練視窗長度可設定」這次改版**沒有改變舊行為**。

★ 2026-08-04 移植自 `1107_window_length/tools/`。原版拿 `~/Desktop/1106_mdd`
的產出當參照；母體這邊改成對**本 repo 自己的** `database/experiment/`（§5.8
baseline，2 年視窗跑的），不依賴任何實驗副本存不存在。

做法：顯式設 `TRAIN_YEARS_N = 2, TRAIN_YEAR_OFFSET = 0,
DECAY_HALF_LIFE_MONTHS = None`（＝改版前寫死的行為）只跑一折，跟該折已存在的
`metrics.json` 對數字。

    .venv/bin/python tools/regression_check_window.py              # 預設對 2023_2025 折
    .venv/bin/python tools/regression_check_window.py --window 2022_2024

⚠ **這支要跑完整訓練（含 Optuna），不是三秒的事**——它是「改動視窗推導邏輯之後
才跑一次」的東西，不是每次改 code 都跑。跑之前確認 `database_make_liq_filtered/`
的快取還在，否則會先重算流動性篩選。

門檻：ic / ic_daily / rmse 三項相對誤差 < 1e-6（同 seed、同資料、同程式路徑，
應該要逐位元相同，留一點浮點餘裕）；`n_test` 要求完全相同。

⚠ 參照折是用**當時的 database_make/** 跑的。若之後資料重生過（見 §7.1 的
`--data` 對拍、§9-#14），對不上未必是這次改版的錯——先確認資料沒動過。

exit code：0 = 與參照一致；1 = 不一致；2 = 參照或產出檔不存在。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from objective.research_config import RunConfig          # noqa: E402
import main_fix_0709 as mf                               # noqa: E402

RESEARCH_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR       = RESEARCH_ROOT / "database" / "experiment_regression_check"
KEYS          = ("ic", "ic_daily", "rmse", "n_test")


def build_config(train_first: int, test_year: int) -> RunConfig:
    """只跑指定的那一折，且顯式鎖回改版前的視窗設定。"""

    class RegressionConfig(RunConfig):
        # 唯一可跑的 fold：train=[train_first .. test_year-1] / test=test_year
        ML_WINDOW_START = train_first
        ML_WINDOW_END   = test_year
        EXPERIMENT_DIR  = OUT_DIR
        # ★ 顯式鎖回舊行為——不要吃 RunConfig 現行的 4 年預設，
        #   這支的用途就是驗「(2, 0, None) 這條路徑沒被改壞」。
        TRAIN_YEARS_N          = 2
        TRAIN_YEAR_OFFSET      = 0
        DECAY_HALF_LIFE_MONTHS = None

    return RegressionConfig()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", default="2023_2025",
                    help="參照 window 目錄名，格式 {訓練首年}_{測試年}（預設 2023_2025）")
    ap.add_argument("--reference-root", type=Path,
                    default=RESEARCH_ROOT / "database" / "experiment",
                    help="參照產出的根目錄（預設本 repo 的 database/experiment）")
    args = ap.parse_args()

    try:
        train_first, test_year = (int(x) for x in args.window.split("_"))
    except ValueError:
        print(f"✗ --window 格式應為 YYYY_YYYY，收到 {args.window!r}", file=sys.stderr)
        return 2

    reference = args.reference_root / args.window / "metrics.json"
    if not reference.exists():
        print(f"✗ 找不到參照 metrics：{reference}", file=sys.stderr)
        return 2

    ref = json.loads(reference.read_text(encoding="utf-8"))
    ref_n = ref.get("train_years_n")
    if ref_n not in (None, 2):
        print(f"✗ 參照折是 train_years_n={ref_n} 跑的，不是舊行為（2），"
              f"拿它當回歸基準沒有意義", file=sys.stderr)
        return 2

    mf.run_ml(build_config(train_first, test_year))

    got_path = OUT_DIR / args.window / "metrics.json"
    if not got_path.exists():
        print(f"✗ 本次沒有產出 {got_path}——確認 window 參數與資料年份是否對得上",
              file=sys.stderr)
        return 2
    got = json.loads(got_path.read_text(encoding="utf-8"))

    print(f"\n{'='*66}\n  回歸比對  {got_path}\n          vs  {reference}\n{'='*66}")
    ok = True
    for k in KEYS:
        a, b = got.get(k), ref.get(k)
        if a is None or b is None:
            ok = False
            print(f"  {k:10s}  new={a}  ref={b}   ✗ 缺值")
            continue
        same = (a == b) if isinstance(a, int) else abs(a - b) <= 1e-6 * max(1.0, abs(b))
        ok &= same
        print(f"  {k:10s}  new={a}  ref={b}   {'✓' if same else '✗ 不一致'}")

    print(f"  train_years  new={got.get('train_years')}  ref={ref.get('train_years')}")
    print("\n  ✓ 舊行為未被改變（視窗推導改寫是安全的）" if ok else
          "\n  ✗ 對不上——先查改動（或先確認 database_make/ 沒有重生過）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
