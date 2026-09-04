"""run_otc_experiment.py — 訓練 + 回測「上市+上櫃」候選池擴張版本。

2026-08-17 更新：database_make_otc/ 已補齊完整 2014-2026 歷史（不再只有
2022-2026 pilot），改成直接用 RunConfig 預設的完整 walk-forward
（TRAIN_YEARS_N=4、ML_WINDOW_START=2014、ML_WINDOW_END=2026、9 折），
只覆寫 PRECOMPUTED_DIR / EXPERIMENT_DIR，其餘設定與 database/experiment/
的純上市 baseline 完全對齊，方便 apples-to-apples 比較。

（舊的 2 年視窗 pilot 結果已搬到 database/experiment_otc_pilot_L2_2024_2026/
與 output/backtest_real_otc_pilot_L2_2024_2026/，不受這次影響。）

輸出另存到 database/experiment_otc/ 與 output/backtest_real_otc/，
不動既有的 database/experiment/（純上市 baseline）。

用法：
    python run_otc_experiment.py train      # 只跑訓練（9 折，完整 walk-forward）
    python run_otc_experiment.py backtest   # 只跑回測（需先有訓練產出）
    python run_otc_experiment.py all        # 訓練 + 回測（預設）
"""
from __future__ import annotations

import sys
from pathlib import Path

from objective.research_config import RunConfig
from main_fix_0709 import run_ml


class OtcConfig(RunConfig):
    PRECOMPUTED_DIR = Path("database_make_otc")
    EXPERIMENT_DIR  = Path("database/experiment_otc")


def do_train():
    run_ml(OtcConfig())


def do_backtest():
    import backtest_real as B

    B.EXPERIMENT_DIR = Path("database/experiment_otc").resolve()
    B.DATABASE_MAKE_DIR = Path("database_make_otc").resolve()
    B.OUTPUT_DIR = Path("output/backtest_real_otc").resolve()
    B.main()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("train", "all"):
        do_train()
    if mode in ("backtest", "all"):
        do_backtest()
