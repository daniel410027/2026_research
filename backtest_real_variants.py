#!/usr/bin/env python3
"""backtest_real_variants.py — 以覆寫常數的方式跑 backtest_real 的參數變體

  用途：把 `backtest_real.py` 原封不動跑一次，只覆寫指定的幾個常數，輸出另存一份，
       好跟基準版（output/backtest_real）並排比較。**不修改 backtest_real.py**——
       基準版隨時可重跑，所有變體的輸出同時存在。

  做法沿用 ~/Desktop/1201_monthfeatrue/backtest_real_thr25.py：
  `backtest_real.main()` 讀的是模組層全域，呼叫前覆寫即等效於改常數。

  ─────────────────────────────────────────────
    為什麼要單因子拆解
  ─────────────────────────────────────────────
  線上（2026_daily）與本 repo 的基準有兩處分歧，2026-08-10 一起量：

      PROB_STD_MULT   2.0 → 2.5     （daily commit 668b456，2026-08-10）
      MIN_PRED_SCORE  0.0 → 0.0040  （daily recorder_config.py，2026-08-05）

  合併跑一次只能得到兩者的**加總**效果，無法歸因。故拆成三個變體：

      thr25    只動 PROB_STD_MULT  → 2.5
      score40  只動 MIN_PRED_SCORE → 0.0040
      online   兩個都動（＝線上實際在跑的設定）

  ⚠⚠ **淨報酬的差不要當成效果。** 依 backtest_real.py 的 MAX_CHG_TO_BUY 註解所記
     （2026-08-06 配對複驗）：只換 LightGBM seed 重訓，`backtest_real` 年化本身就
     晃 23.6pp、Sharpe 晃 1.106~1.513。本腳本所有變體共用同一份 predictions，
     所以彼此的差**沒有** seed 噪音——但那個差要跟「換一個 seed 會晃多少」比，
     才知道值不值得一提。幾個 pp 的年化差、0.0X 的 sharpe 差，都在噪音裡。
     ✓ 可信的是**機械量**：換手率、買賣筆數、成本拖累、平均持倉檔數。
       那些是直接數出來的，不是估計量。

  ⚠ 1201 的門檻掃描顯示**門檻曲線是鋸齒的**（2.0→52.7% / 2.5→53.1% / 3.0→35.4%
    / 3.5→41.3%），作者結論是「2.5 之後全是噪音，不要再往上調追峰值」。
    要加新的門檻變體前先看那份檔頭。

  用法：
    python backtest_real_variants.py                # 列出所有變體
    python backtest_real_variants.py thr25          # 跑單一變體
    python backtest_real_variants.py all            # 依序跑全部
    python backtest_real_variants.py online score40 # 跑指定的幾個

  輸出：output/backtest_real_<變體名>/ —— 檔案清單與 backtest_real 完全相同，
        另加 compare_vs_base.csv：與基準版逐項對照（含成本與換手的差）。

作者：Daniel Huang
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import backtest_real as B

_ROOT = Path(__file__).resolve().parent
BASELINE_DIR = _ROOT / "output" / "backtest_real"   # 基準：PROB_STD_MULT=2.0, MIN_PRED_SCORE=0.0

# 變體名 → {backtest_real 的常數名: 新值}
VARIANTS: dict[str, dict[str, float]] = {
    "thr25":   {"PROB_STD_MULT": 2.5},
    "score40": {"MIN_PRED_SCORE": 0.0040},
    "online":  {"PROB_STD_MULT": 2.5, "MIN_PRED_SCORE": 0.0040},
}

# 與基準對照時要並排的欄位
_METRIC_ROWS = [("net", "實單淨值"), ("gross", "實單毛值"), ("D1_cost0", "D1 紙上")]
_METRIC_COLS = ("ann_return", "sharpe", "mdd", "IR")
_SUMMARY_KEYS = ("年化成本拖累(算術)", "日均單邊換手率", "總買入筆數", "總賣出筆數",
                 "平均持倉檔數", "持倉未滿日數佔比")


def compare_with_baseline(baseline_dir: Path, target_dir: Path,
                          name: str) -> pd.DataFrame | None:
    """與基準版並排比較。基準版還沒跑過就跳過（不是錯誤）。"""
    b_metrics = baseline_dir / "metrics.csv"
    if not b_metrics.exists():
        print(f"\n  ⚠ 找不到基準版 {b_metrics}（先跑 python backtest_real.py）→ 跳過比較")
        return None

    bm = pd.read_csv(b_metrics, index_col="key", encoding="utf-8-sig")
    tm = pd.read_csv(target_dir / "metrics.csv", index_col="key", encoding="utf-8-sig")
    bc = json.loads((baseline_dir / "run_config.json").read_text(encoding="utf-8"))
    tc = json.loads((target_dir / "run_config.json").read_text(encoding="utf-8"))

    col_b, col_t = "基準", name
    rows = []
    for key, label in _METRIC_ROWS:
        if key not in bm.index or key not in tm.index:
            continue
        for col in _METRIC_COLS:
            rows.append({"項目": f"{label} {col}",
                         col_b: bm.loc[key, col], col_t: tm.loc[key, col]})
    for k in _SUMMARY_KEYS:
        rows.append({"項目": k, col_b: bc["summary"][k], col_t: tc["summary"][k]})

    cmp = pd.DataFrame(rows).set_index("項目")
    cmp.to_csv(target_dir / "compare_vs_base.csv", encoding="utf-8-sig")

    changed = {k: (bc.get(k), tc.get(k)) for k in VARIANTS[name]}
    print("\n" + "=" * 70)
    print(f"  基準 vs {name}　"
          + "，".join(f"{k} {o} → {n}" for k, (o, n) in changed.items()))
    print("=" * 70)
    print(cmp.to_string())
    print("\n  ⚠ 淨報酬與 sharpe 的差要跟 seed 噪音（年化 ±23.6pp）比；"
          "可信的是成本與換手，那是機械量。")
    print(f"  ✓ compare_vs_base.csv")
    return cmp


def run_variant(name: str) -> Path:
    """覆寫常數 → 跑 backtest_real.main() → 還原。回傳輸出目錄。"""
    overrides = VARIANTS[name]
    target_dir = _ROOT / "output" / f"backtest_real_{name}"

    # 先記下原值才能還原——`all` 模式下同一個 process 會連跑多個變體，
    # 不還原的話第二個變體會疊在第一個之上，靜默跑出錯的東西。
    saved = {k: getattr(B, k) for k in overrides}
    saved_out = B.OUTPUT_DIR
    try:
        for k, v in overrides.items():
            setattr(B, k, v)
        B.OUTPUT_DIR = target_dir

        print("\n" + "#" * 70)
        print(f"#  變體 {name}："
              + "，".join(f"{k} {saved[k]} → {v}" for k, v in overrides.items()))
        print(f"#  輸出 → {target_dir}")
        print("#" * 70)
        B.main()
    finally:
        for k, v in saved.items():
            setattr(B, k, v)
        B.OUTPUT_DIR = saved_out

    compare_with_baseline(BASELINE_DIR, target_dir, name)
    return target_dir


def summarize(names: list[str]):
    """全部變體跑完後，把基準與各變體的關鍵數字併成一張表。"""
    if not (BASELINE_DIR / "run_config.json").exists():
        return
    cols = {}

    def _col(d: Path, label: str):
        if not (d / "metrics.csv").exists():
            return
        m = pd.read_csv(d / "metrics.csv", index_col="key", encoding="utf-8-sig")
        c = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        s = c["summary"]
        cols[label] = {
            "PROB_STD_MULT":   c["PROB_STD_MULT"],
            "MIN_PRED_SCORE":  c["MIN_PRED_SCORE"],
            "淨年化":           m.loc["net", "ann_return"],
            "毛年化":           m.loc["gross", "ann_return"],
            "sharpe":          m.loc["net", "sharpe"],
            "MDD":             m.loc["net", "mdd"],
            "年化成本拖累":      s["年化成本拖累(算術)"],
            "日均換手":         s["日均單邊換手率"],
            "總買入筆數":        s["總買入筆數"],
            "平均持倉檔數":      s["平均持倉檔數"],
            "持倉未滿佔比":      s["持倉未滿日數佔比"],
        }

    _col(BASELINE_DIR, "基準")
    for n in names:
        _col(_ROOT / "output" / f"backtest_real_{n}", n)
    if len(cols) < 2:
        return

    table = pd.DataFrame(cols)
    out = _ROOT / "output" / "variants_summary.csv"
    table.to_csv(out, encoding="utf-8-sig")
    print("\n" + "=" * 70)
    print("  變體總表（同一份 predictions，彼此的差不含 seed 噪音）")
    print("=" * 70)
    print(table.to_string())
    print(f"\n  ✓ {out}")
    print("  ⚠ 報酬類的差在 seed 噪音內時不可宣稱效果；成本與換手是機械量，可信。")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print(__doc__.split("用法：")[0])
        print("可用變體：")
        for n, ov in VARIANTS.items():
            print(f"  {n:10s} " + "，".join(f"{k} → {v}" for k, v in ov.items()))
        print(f"\n  基準（backtest_real.py 本體）："
              f"PROB_STD_MULT {B.PROB_STD_MULT}，MIN_PRED_SCORE {B.MIN_PRED_SCORE}")
        print("\n用法：python backtest_real_variants.py {all|" + "|".join(VARIANTS) + "}")
        return

    names = list(VARIANTS) if args == ["all"] else args
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"未知的變體：{unknown}；可用：{list(VARIANTS)}")

    B.setup_cjk_font(verbose=False)
    for n in names:
        run_variant(n)
    summarize(names)


if __name__ == "__main__":
    main()
