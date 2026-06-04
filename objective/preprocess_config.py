"""
preprocess_config.py
====================
PreprocessConfig dataclass（新專案版）。

變更（vs 舊版）：
    - 移除 raw_data_dir / market_data_path（資料改由 FinLab API 提供）
    - 新增 start_date / end_date property（由 start_year/end_year 轉換）
    - skip_year_check 保留供 DailyPreprocessor 使用

作者：Daniel Huang
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PreprocessConfig:
    """前處理設定檔，所有超參數集中於此。"""

    # ── 年份設定 ────────────────────────────────────────────
    start_year: int = 2014
    end_year:   int = 2016

    # ── 資料庫路徑（download.py 存出的年度 CSV 目錄）───────
    db_dir: Path = field(default_factory=lambda: Path("database"))

    # ── 預測標籤設定 ────────────────────────────────────────
    return_clip:      float = 0.15
    tick_threshold:   float = 0.01
    tick0_threshold:  float = 0.00

    # ── Beta 設定 ───────────────────────────────────────────
    beta_window: int = 60

    # ── 借券 / 融券特徵設定 ──────────────────────────────────
    short_windows:        list = field(default_factory=lambda: [5, 10, 20])
    short_log_transform:  bool = True
    short_drop_aggregate: bool = True

    # ── DailyPreprocessor 用：跳過年份範圍檢查 ──────────────
    skip_year_check: bool = False

    def __post_init__(self):
        self.db_dir = Path(self.db_dir)
        if not self.skip_year_check:
            assert self.end_year - self.start_year >= 2, (
                f"年份範圍需 >= 2 年，目前為 {self.end_year - self.start_year} 年"
            )
