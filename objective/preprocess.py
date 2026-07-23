"""
preprocess.py
=============
台灣股市資料前處理模組（FinLab 新專案版）。

主要變更（vs 舊版 TEJ CSV 版）：
    - 資料來源：TEJ 季度 CSV → 本地 database/YYYY.csv（via DBLoader）
    - 移除步驟：_normalize_column_names, _resolve_ohlcv_columns,
                _encode_flag_features, _load_market_data（CSV 相關）
    - 新增步驟：_add_price_derived_features, _add_return_momentum_extended,
                _add_capm_beta_extended, _add_institutional_flow_features,
                _add_fundamental_features
    - 欄位名稱標準化：adj_close / high / low / open / volume（替換中文 TEJ 名）
    - _fix_discontinuous_data 以 adj_close 為基底（替換 收盤價元）
    - 大盤特徵（market_return）由 DBLoader 直接提供

對外介面不變：
    from objective.preprocess import PreprocessConfig, DataPreprocessor

作者：Daniel Huang
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess_config import PreprocessConfig  # noqa: F401
from .feature_mixin     import FeatureMixin
from .db_loader         import DBLoader

warnings.filterwarnings("ignore")

__all__ = ["PreprocessConfig", "DataPreprocessor"]


# ─────────────────────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────────────────────

class DataPreprocessor(FeatureMixin):
    """
    台灣股市資料前處理器（FinLab 版）。

    Parameters
    ----------
    config : PreprocessConfig

    Examples
    --------
    >>> from objective.preprocess import PreprocessConfig, DataPreprocessor
    >>> config = PreprocessConfig(start_year=2018, end_year=2022)
    >>> df = DataPreprocessor(config).run()
    """

    KEY_COLS = ["證券代碼", "年月日"]

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.df: pd.DataFrame | None = None

    # ──────────────────────────────────────────────────────────
    #  公開介面
    # ──────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """執行完整流程：FinLab 載入 → 清洗 → 特徵工程 → 回傳 DataFrame。"""
        print(f"\n{'='*60}")
        print(f"  DataPreprocessor  {self.config.start_year} ~ {self.config.end_year}")
        print(f"{'='*60}")

        # ── 1. 從本地資料庫讀取資料 ──────────────────────────
        print("\n► 載入本地資料庫")
        loader   = DBLoader(
            start_year = self.config.start_year,
            end_year   = self.config.end_year,
            db_dir     = self.config.db_dir,
        )
        self.df = loader.load()

        # ── 2. 特徵工程流程 ────────────────────────────────────
        steps = [
            # 基礎清洗
            ("清理欄位",                      self._clean_columns),
            ("移除類別型特徵",                 self._remove_categorical_features),
            ("填補不連續資料",                 self._fix_discontinuous_data),
            # 大盤特徵（market_return 已由 DBLoader 提供，此步驟做驗證）
            ("大盤特徵驗證",                   self._verify_market_features),
            # 技術指標
            ("RSI",                            self._add_rsi_features),
            ("移動平均線",                     self._add_moving_average_features),
            ("MACD",                           self._add_macd_features),
            ("KD",                             self._add_kd_features),
            ("Williams %R",                    self._add_williams_r_features),
            ("ROC / Momentum",                 self._add_roc_momentum_features),
            ("ATR / Bollinger",                self._add_volatility_features),
            ("OBV / VWAP",                     self._add_volume_features),
            ("CCI",                            self._add_cci_features),
            # 衍生特徵（FinLab 新增）
            ("Price Derived",                  self._add_price_derived_features),
            ("Return / Momentum Extended",     self._add_return_momentum_extended),
            ("Institutional Flow",             self._add_institutional_flow_features),
            ("Fundamental",                    self._add_fundamental_features),
            # 借券 / 融券
            ("Short Selling",                  self._add_short_selling_features),
            # 風險特徵（依賴 daily_return；beta 需在 CAPM beta 前完成）
            ("Beta / Vol20",                   self._add_beta_vol_features),
            # delay1 反轉/微結構/流動性因子（依賴 vol20 + amount，排在 Beta/Vol20 後）
            ("Delay1 Microstructure",          self._add_delay1_microstructure_features),
            ("CAPM Beta Extended",             self._add_capm_beta_extended),
            # Return Labels（最後執行；drop market_return_fwd）
            ("Return Labels",                  self._add_return_features),
        ]

        for name, step in steps:
            print(f"\n► {name}")
            if step() is False:
                raise RuntimeError(f"處理失敗於步驟：{name}")

        self._print_summary()
        return self.df

    # ──────────────────────────────────────────────────────────
    #  資料清洗（簡化版：FinLab 資料已為 numeric，無 CSV 雜訊）
    # ──────────────────────────────────────────────────────────

    def _clean_columns(self):
        """
        1. 確保關鍵欄位型別正確（年月日 datetime, 證券代碼 str）
        2. 去重（日期 × 股票）
        3. 移除 null_rate >= 99% 的欄位
        """
        if not {"證券代碼", "年月日"}.issubset(self.df.columns):
            print("  ✗ 缺少關鍵欄位")
            return False

        self.df["年月日"]   = pd.to_datetime(self.df["年月日"])
        self.df["證券代碼"] = self.df["證券代碼"].astype(str)

        # dedup
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=self.KEY_COLS, keep="last")
        dup = before - len(self.df)
        if dup:
            print(f"  ⚠ 移除重複: {dup:,} 筆")
        else:
            print("  ✓ 無重複 (日期, 股票)")

        # 移除幾乎全空的欄位
        total = len(self.df)
        drop  = [
            c for c in self.df.columns
            if c not in self.KEY_COLS and self.df[c].isnull().mean() >= 0.99
        ]
        if drop:
            self.df = self.df.drop(columns=drop)
            print(f"  移除高缺值欄位: {len(drop)} 個")

        self.df = self.df.sort_values(self.KEY_COLS).reset_index(drop=True)
        print(f"  清洗後: {self.df.shape}")

    def _remove_categorical_features(self):
        """移除無法轉為 numeric 的類別型欄位。"""
        obj_cols        = self.df.select_dtypes(include=["object"]).columns.tolist()
        converted_count = 0
        for col in obj_cols:
            if col in self.KEY_COLS:
                continue
            orig_nonnull = self.df[col].notna().mean()
            converted    = pd.to_numeric(self.df[col], errors="coerce")
            conv_nonnull = converted.notna().mean()
            if orig_nonnull > 0 and (conv_nonnull / orig_nonnull) >= 0.8:
                self.df[col] = converted
                converted_count += 1

        numeric_cols = set(self.df.select_dtypes(include=[np.number]).columns)
        keep   = set(self.KEY_COLS) | numeric_cols | {"證券代碼"}
        remove = [c for c in self.df.columns if c not in keep]
        self.df = self.df.drop(columns=remove, errors="ignore")

        if converted_count:
            print(f"  object→numeric 轉換: {converted_count} 欄")
        if remove:
            print(f"  移除類別型特徵: {len(remove)} 個")

    def _fix_discontinuous_data(self):
        """
        以全局交易日曆為模板，補齊不連續交易日（停牌等）。
        基底欄位改為 adj_close（替換舊版的 收盤價元）。
        批次 50 檔控制記憶體。
        """
        price_col = "adj_close"
        if price_col not in self.df.columns:
            print("  ✗ 找不到 adj_close 欄位")
            return False

        trading_days = sorted(self.df["年月日"].unique())
        truncated    = 0
        fixed_list   = []
        batch_size   = 50
        stocks       = self.df["證券代碼"].unique()

        for batch_start in range(0, len(stocks), batch_size):
            batch_stocks = stocks[batch_start:batch_start + batch_size]
            batch_dfs    = []

            for stock in batch_stocks:
                group    = self.df[self.df["證券代碼"] == stock]
                template = pd.DataFrame({"年月日": trading_days, "證券代碼": stock})
                merged   = template.merge(group, on=["年月日", "證券代碼"], how="left")
                merged[price_col] = merged[price_col].ffill()

                if merged[price_col].isnull().any():
                    truncated += 1
                    first = merged[price_col].first_valid_index()
                    if first is not None:
                        merged = merged.loc[first:].copy()

                for col in merged.select_dtypes(include=[np.number]).columns:
                    if col != price_col:
                        merged[col] = merged[col].ffill().fillna(0)

                batch_dfs.append(merged)

            if batch_dfs:
                fixed_list.append(
                    pd.concat(batch_dfs, ignore_index=True)
                    .sort_values(self.KEY_COLS)
                    .reset_index(drop=True)
                )

        if fixed_list:
            self.df = (
                pd.concat(fixed_list, ignore_index=True)
                .sort_values(self.KEY_COLS)
                .reset_index(drop=True)
            )

        print(f"  填補完成，{truncated} 檔股票因早期缺值被截斷")

    def _verify_market_features(self):
        """
        大盤特徵（market_return / market_return_fwd）
        由 DBLoader 直接提供，此步驟僅做欄位驗證與 ffill 補值。
        """
        for col in ["market_return", "market_return_fwd", "market_index"]:
            if col in self.df.columns:
                self.df[col] = (
                    self.df.groupby("證券代碼")[col]
                    .transform(lambda x: x.ffill().fillna(0))
                )
        n_ok = self.df.get("market_return", pd.Series()).notna().sum()
        print(f"  market_return 有效筆數: {n_ok:,}")

    # ──────────────────────────────────────────────────────────
    #  摘要
    # ──────────────────────────────────────────────────────────

    def _print_summary(self):
        df = self.df
        indicator_cols = [
            c for c in df.columns if any(
                k in c for k in [
                    "RSI", "SMA", "EMA", "MACD", "KD", "Williams",
                    "ROC", "Momentum", "ATR", "BB_", "OBV", "VWAP", "CCI",
                    "beta", "vol20", "market", "daily_return", "CAPM_Beta",
                ]
            )
        ]
        short_cols = [
            c for c in df.columns
            if c.startswith("short_")
            or c in ("days_to_cover", "days_to_cover_rank",
                     "inst_retail_ratio", "short_margin_ratio")
        ]
        look_ahead_ok = "market_return_fwd" not in df.columns

        print(f"\n{'='*60}")
        print(f"  ✓ 完成")
        print(f"  形狀            : {df.shape}")
        print(f"  股票數量        : {df['證券代碼'].nunique()}")
        print(f"  日期範圍        : {df['年月日'].min().date()} ~ {df['年月日'].max().date()}")
        print(f"  技術指標欄位數  : {len(indicator_cols)}")
        print(f"  借券/融券欄位數 : {len(short_cols)}")
        print(
            f"  Look-ahead 防護: "
            f"{'✓ market_return_fwd 已 drop' if look_ahead_ok else '✗ 警告: market_return_fwd 仍存在！'}"
        )
        print(f"{'='*60}\n")
