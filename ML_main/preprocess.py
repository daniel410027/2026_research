"""
preprocess.py
=============
台灣股市資料前處理與特徵工程模組。

功能：
    - 自動讀取指定年份範圍的季度 CSV（raw_data/YYYYQ.csv）
    - 欄位清理、類別型特徵移除、不連續交易日填補
    - 技術指標計算：RSI, SMA, EMA, MACD, KD, Williams %R,
                   ROC, Momentum, ATR, Bollinger Bands, OBV, VWAP, CCI
    - 大盤特徵：market（加權指數收盤），market_return（日報酬）
    - 風險特徵：
        * beta      : rolling 60日 OLS 斜率（Cov/Var），再做 Vasicek shrinkage 縮向 1.0
        * vol20     : daily_return 的 20 日滾動標準差 × √252（年化日波動率）
        * daily_return 使用「收盤價 pct_change」（backward-looking），與 return label 嚴格切割
    - 預測標籤生成：
        * return / return_tick / return_tick_0
        * excess_return / excess_return_tick（個股 return 去除 beta × 大盤同期報酬後的超額報酬）
    - market_return_fwd 僅用於計算 excess_return，計算完畢後自動 drop，防止 look-ahead bias

使用方式（在其他 .py 呼叫）：
    from preprocess import PreprocessConfig, DataPreprocessor

    config = PreprocessConfig(start_year=2018, end_year=2022)
    preprocessor = DataPreprocessor(config)
    df = preprocessor.run()

CONFIG 說明：
    start_year        : 起始年份（資料需存在）
    end_year          : 結束年份，需 >= start_year + 2（至少涵蓋 2 年）
    raw_data_dir      : 季度 CSV 所在資料夾路徑
    market_data_path  : 大盤 CSV 路徑（預設 database/market/market.csv）
    return_clip       : return 的上下截斷絕對值（預設 ±15%）
    tick_threshold    : return_tick 的正報酬閾值（預設 1%，即 0.01）
    tick0_threshold   : return_tick_0 的正報酬閾值（預設 0，即 return > 0）
    beta_window       : rolling beta 視窗（預設 60 交易日）

作者：Daniel Huang
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
#  專案根目錄：2026_research/（ML_main/ 的上一層）
# ============================================================
_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
#  CONFIG
# ============================================================

@dataclass
class PreprocessConfig:
    """前處理設定檔，所有超參數集中於此。"""

    # ── 年份設定 ────────────────────────────────────────────
    start_year: int = 2014          # 起始年份
    end_year:   int = 2016          # 結束年份（需 >= start_year + 2）

    # ── 路徑設定 ────────────────────────────────────────────
    raw_data_dir:     Path = field(default_factory=lambda: _ROOT / "database/raw_data")
    market_data_path: Path = field(default_factory=lambda: _ROOT / "database/market/market.csv")

    # ── 預測標籤設定 ────────────────────────────────────────
    return_clip:      float = 0.15  # return 截斷絕對值上限（±15%）
    tick_threshold:   float = 0.01  # return_tick   閾值（預設 1%）
    tick0_threshold:  float = 0.00  # return_tick_0 閾值（預設 > 0）

    # ── Beta 設定 ───────────────────────────────────────────
    beta_window: int = 60           # rolling beta 視窗（交易日），業界常用 60–252

    def __post_init__(self):
        assert self.end_year - self.start_year >= 2, (
            f"年份範圍需 >= 2 年，目前為 {self.end_year - self.start_year} 年"
        )
        self.raw_data_dir     = Path(self.raw_data_dir)
        self.market_data_path = Path(self.market_data_path)


# ============================================================
#  MAIN CLASS
# ============================================================

class DataPreprocessor:
    """
    台灣股市資料前處理器。

    Parameters
    ----------
    config : PreprocessConfig
        前處理設定檔。

    Examples
    --------
    >>> from preprocess import PreprocessConfig, DataPreprocessor
    >>> config = PreprocessConfig(start_year=2018, end_year=2022)
    >>> preprocessor = DataPreprocessor(config)
    >>> df = preprocessor.run()
    """

    KEY_COLS = ["證券代碼", "年月日"]

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.df: pd.DataFrame | None = None
        self._market_df: pd.DataFrame | None = None  # 大盤日資料（date-indexed）

    # ────────────────────────────────────────────────────────
    #  公開介面
    # ────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """執行完整流程：讀檔 → 前處理 → 特徵工程 → 回傳 DataFrame。"""
        print(f"\n{'='*60}")
        print(f"  DataPreprocessor  {self.config.start_year} ~ {self.config.end_year}")
        print(f"{'='*60}")

        self.df = self._load_and_merge()

        steps = [
            ("正規化欄位名稱",    self._normalize_column_names),
            ("清理欄位",          self._clean_columns),
            ("移除類別型特徵",    self._remove_categorical_features),
            ("填補不連續資料",    self._fix_discontinuous_data),
            # ── 大盤特徵（需在 beta/excess_return 前完成）──
            ("大盤特徵",          self._add_market_features),
            # ── 技術指標 ──
            ("RSI",               self._add_rsi_features),
            ("移動平均線",        self._add_moving_average_features),
            ("MACD",              self._add_macd_features),
            ("KD",                self._add_kd_features),
            ("Williams %R",       self._add_williams_r_features),
            ("ROC / Momentum",    self._add_roc_momentum_features),
            ("ATR / Bollinger",   self._add_volatility_features),
            ("OBV / VWAP",        self._add_volume_features),
            ("CCI",               self._add_cci_features),
            # ── 風險特徵（beta, vol20 依賴 daily_return，需在 return labels 前）──
            ("Beta / Vol20",      self._add_beta_vol_features),
            # ── Return Labels（含 excess_return，最後 drop market_return_fwd）──
            ("Return Labels",     self._add_return_features),
        ]

        for name, step in steps:
            print(f"\n► {name}")
            if step() is False:
                raise RuntimeError(f"處理失敗於步驟：{name}")

        self._print_summary()
        return self.df

    # ────────────────────────────────────────────────────────
    #  讀檔與合併
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _read_csv(file_path: Path) -> pd.DataFrame:
        """自動嘗試多種編碼讀取 CSV（Tab 分隔的 TEJ 格式）。"""
        for enc in ["utf-8", "utf-8-sig", "utf-16", "utf-16-sig", "big5", "gbk", "latin-1"]:
            try:
                return pd.read_csv(file_path, encoding=enc, sep="\t", low_memory=False)
            except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError):
                continue
        raise ValueError(f"無法讀取 {file_path}")

    @staticmethod
    def _read_csv_auto(file_path: Path) -> pd.DataFrame:
        """
        自動偵測分隔符（逗號 or Tab）並嘗試多種編碼讀取 CSV。
        適用於大盤等非 TEJ Tab 格式的 CSV。
        """
        encodings = ["utf-8-sig", "utf-8", "utf-16", "utf-16-sig", "big5", "gbk", "latin-1"]
        separators = [",", "\t"]
        for enc in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, encoding=enc, sep=sep, low_memory=False)
                    if df.shape[1] >= 2:
                        return df
                except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError):
                    continue
        raise ValueError(f"無法讀取 {file_path}")

    def _load_and_merge(self) -> pd.DataFrame:
        """讀取所有季度 CSV 並合併去重。"""
        dfs = []
        for year in range(self.config.start_year, self.config.end_year + 1):
            for q in range(1, 5):
                path = self.config.raw_data_dir / f"{year}{q}.csv"
                if path.exists():
                    df = self._read_csv(path)
                    dfs.append(df)
                    print(f"  讀取: {path.name}  ({len(df):,} 筆)")
                else:
                    print(f"  警告: 找不到 {path.name}")

        if not dfs:
            raise FileNotFoundError(
                f"未找到任何檔案，請確認路徑: {self.config.raw_data_dir}"
            )

        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.drop_duplicates(subset=["年月日", "證券代碼"], keep="last")
        print(f"\n  合併完成: {len(merged):,} 筆，{merged['證券代碼'].nunique()} 檔股票")
        return merged

    # ────────────────────────────────────────────────────────
    #  前處理步驟
    # ────────────────────────────────────────────────────────

    def _normalize_column_names(self):
        old_cols = self.df.columns.tolist()
        new_cols = [re.sub(r"[^\w\u4e00-\u9fff]", "", c) for c in old_cols]

        duplicates = {c for c, n in Counter(new_cols).items() if n > 1}
        if duplicates:
            counter, final = {}, []
            for old, new in zip(old_cols, new_cols):
                if new in duplicates:
                    idx = counter.get(new, 0)
                    counter[new] = idx + 1
                    final.append(new if idx == 0 else f"{new}_{idx}")
                else:
                    final.append(new)
            new_cols = final

        self.df.columns = new_cols
        self.KEY_COLS = [re.sub(r"[^\w\u4e00-\u9fff]", "", c) for c in ["證券代碼", "年月日"]]

    def _clean_columns(self):
        if not {"證券代碼", "年月日"}.issubset(self.df.columns):
            print("  ✗ 缺少關鍵欄位")
            return False

        self.df["證券代碼"] = self.df["證券代碼"].astype(str).str.extract(r"(\d{4})", expand=False)
        self.df["年月日"] = pd.to_datetime(self.df["年月日"], format="%Y%m%d", errors="coerce")

        before = len(self.df)
        self.df = self.df.dropna(subset=["年月日"])
        dropped = before - len(self.df)
        if dropped:
            print(f"  移除年月日缺失: {dropped} 筆")

        total = len(self.df)
        keep, drop = self.KEY_COLS.copy(), []

        for col in self.df.columns:
            if col in self.KEY_COLS:
                continue
            null_r = self.df[col].isnull().sum() / total
            zero_r = (
                (self.df[col] == 0).sum() / total
                if pd.api.types.is_numeric_dtype(self.df[col]) else 0
            )
            if null_r >= 0.99:
                drop.append(col)
            elif zero_r >= 0.99 and pd.api.types.is_numeric_dtype(self.df[col]):
                drop.append(col)
            else:
                keep.append(col)

        print(f"  保留: {len(keep)} 欄，移除: {len(drop)} 欄")
        self.df = self.df[keep].sort_values(self.KEY_COLS).reset_index(drop=True)

    def _remove_categorical_features(self):
        numeric_cols = set(self.df.select_dtypes(include=[np.number]).columns)
        keep = set(self.KEY_COLS) | numeric_cols | {"證券代碼"}
        remove = [c for c in self.df.columns if c not in keep]
        self.df = self.df.drop(columns=remove, errors="ignore")
        self.df = self.df.sort_values(self.KEY_COLS).reset_index(drop=True)
        print(f"  移除 {len(remove)} 個類別型特徵")

    def _fix_discontinuous_data(self):
        price_col = "收盤價元"
        if price_col not in self.df.columns:
            print("  ✗ 找不到收盤價欄位")
            return False

        trading_days = sorted(self.df["年月日"].unique())
        truncated = 0
        fixed_list = []
        batch_size = 50  # 每 50 檔股票批次處理，減少記憶體峰值

        # 取得所有股票列表
        stocks = self.df["證券代碼"].unique()
        total_stocks = len(stocks)
        
        for batch_start in range(0, total_stocks, batch_size):
            batch_end = min(batch_start + batch_size, total_stocks)
            batch_stocks = stocks[batch_start:batch_end]
            batch_dfs = []
            
            for stock in batch_stocks:
                group = self.df[self.df["證券代碼"] == stock]
                template = pd.DataFrame({"年月日": trading_days, "證券代碼": stock})
                merged = template.merge(group, on=["年月日", "證券代碼"], how="left")
                merged[price_col] = merged[price_col].fillna(method="ffill")

                if merged[price_col].isnull().any():
                    truncated += 1
                    first = merged[price_col].first_valid_index()
                    if first is not None:
                        merged = merged.loc[first:].copy()

                for col in merged.select_dtypes(include=[np.number]).columns:
                    if col != price_col:
                        merged[col] = merged[col].fillna(method="ffill").fillna(0)

                batch_dfs.append(merged)
            
            # 批次內 concat 並排序
            if batch_dfs:
                batch_df = pd.concat(batch_dfs, ignore_index=True)
                batch_df = batch_df.sort_values(self.KEY_COLS).reset_index(drop=True)
                fixed_list.append(batch_df)
        
        # 最後合併所有批次
        if fixed_list:
            self.df = pd.concat(fixed_list, ignore_index=True)
            # 確保最終排序（批次之間可能無序）
            self.df = self.df.sort_values(self.KEY_COLS).reset_index(drop=True)
        
        print(f"  填補完成，{truncated} 檔股票因早期缺值被截斷")

    # ────────────────────────────────────────────────────────
    #  大盤特徵
    # ────────────────────────────────────────────────────────

    def _load_market_data(self) -> pd.DataFrame:
        """
        讀取大盤 CSV，回傳以 年月日 為 index 的 DataFrame，包含：
            market            : 加權指數收盤（原始水準值，可作宏觀環境特徵）
            market_return     : 大盤日報酬（backward-looking，pct_change，無 look-ahead bias）
            market_return_fwd : 大盤 T+1 日報酬（forward-looking）
                                → 對應 return 的持有期間（T+1 開盤至 T+2 開盤）
                                → 僅供 excess_return 計算，最終由 _add_return_features drop
        """
        path = self.config.market_data_path
        if not path.exists():
            raise FileNotFoundError(f"找不到大盤資料: {path}")

        raw = self._read_csv_auto(path)
        # 標準化欄位名稱（移除括號等特殊字元）
        raw.columns = [re.sub(r"[^\w\u4e00-\u9fff]", "", c) for c in raw.columns]
        close_col = next(c for c in raw.columns if "收盤價" in c)

        raw["年月日"] = pd.to_datetime(raw["年月日"].astype(str), format="%Y%m%d", errors="coerce")
        raw = raw.dropna(subset=["年月日"]).sort_values("年月日").reset_index(drop=True)
        raw[close_col] = pd.to_numeric(raw[close_col], errors="coerce")

        raw["market"]            = raw[close_col]
        raw["market_return"]     = raw[close_col].pct_change()       # backward-looking
        raw["market_return_fwd"] = raw["market_return"].shift(-1)    # T+1 大盤報酬（forward）

        return raw.set_index("年月日")[["market", "market_return", "market_return_fwd"]]

    def _add_market_features(self):
        """
        將大盤特徵（market, market_return, market_return_fwd）以 年月日 merge 進主 DataFrame。
        - market            : 大盤水準，直接作為宏觀環境特徵
        - market_return     : 大盤日報酬，保留供模型使用（backward-looking，無 bias）
        - market_return_fwd : 暫存，用於 excess_return 計算後由 _add_return_features drop
        """
        self._market_df = self._load_market_data()

        self.df = self.df.merge(
            self._market_df.reset_index(),
            on="年月日",
            how="left"
        )

        # 若某交易日無大盤資料（如補假日），forward-fill
        for col in ["market", "market_return", "market_return_fwd"]:
            self.df[col] = self.df[col].fillna(method="ffill")
        # 第一筆 pct_change 為 NaN，填 0
        self.df["market_return"]     = self.df["market_return"].fillna(0)
        self.df["market_return_fwd"] = self.df["market_return_fwd"].fillna(0)

        n_ok = self.df["market"].notna().sum()
        print(f"  合併大盤資料: {n_ok:,} 筆成功對齊")
        print(f"  大盤指數範圍: {self.df['market'].min():.0f} ~ {self.df['market'].max():.0f}")

    # ────────────────────────────────────────────────────────
    #  技術指標計算（靜態方法）
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _rsi(series: pd.Series, window: int) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce")
        delta = series.diff()
        alpha = 1.0 / window
        gain = delta.clip(lower=0).ewm(alpha=alpha, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=alpha, adjust=False).mean()
        rsi = 100 - 100 / (1 + gain / (loss + 1e-9))
        rsi.iloc[:window] = np.nan
        return rsi

    @staticmethod
    def _sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=window).mean()

    @staticmethod
    def _ema(series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def _macd(series: pd.Series, fast=12, slow=26, signal=9):
        line = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
        sig  = line.ewm(span=signal, adjust=False).mean()
        return line, sig, line - sig

    @staticmethod
    def _kd(high: pd.Series, low: pd.Series, close: pd.Series, k_window=9):
        ll = low.rolling(k_window, min_periods=k_window).min()
        hh = high.rolling(k_window, min_periods=k_window).max()
        rsv = 100 * (close - ll) / (hh - ll + 1e-9)
        k_vals, d_vals, kp, dp = [], [], 50, 50
        for v in rsv:
            if pd.isna(v):
                k_vals.append(np.nan); d_vals.append(np.nan)
            else:
                kp = 2/3 * kp + 1/3 * v
                dp = 2/3 * dp + 1/3 * kp
                k_vals.append(kp); d_vals.append(dp)
        return pd.Series(k_vals, index=close.index), pd.Series(d_vals, index=close.index)

    @staticmethod
    def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window=14) -> pd.Series:
        hh = high.rolling(window, min_periods=window).max()
        ll = low.rolling(window, min_periods=window).min()
        return -100 * (hh - close) / (hh - ll + 1e-9)

    @staticmethod
    def _roc(series: pd.Series, window: int) -> pd.Series:
        return 100 * (series - series.shift(window)) / (series.shift(window) + 1e-9)

    @staticmethod
    def _momentum(series: pd.Series, window: int) -> pd.Series:
        return series - series.shift(window)

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window=14) -> pd.Series:
        pc = close.shift(1)
        tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
        return tr.ewm(span=window, adjust=False).mean()

    @staticmethod
    def _bollinger(series: pd.Series, window=20, num_std=2):
        mid = series.rolling(window, min_periods=window).mean()
        std = series.rolling(window, min_periods=window).std()
        upper, lower = mid + num_std * std, mid - num_std * std
        pct_b = (series - lower) / (upper - lower + 1e-9)
        return upper, mid, lower, pct_b

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        vals, prev = [0], 0
        for i in range(1, len(close)):
            if   close.iloc[i] > close.iloc[i-1]: prev += volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]: prev -= volume.iloc[i]
            vals.append(prev)
        return pd.Series(vals, index=close.index)

    @staticmethod
    def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        tp = (high + low + close) / 3
        return (tp * volume).cumsum() / (volume.cumsum() + 1e-9)

    @staticmethod
    def _cci(high: pd.Series, low: pd.Series, close: pd.Series, window=20) -> pd.Series:
        tp  = (high + low + close) / 3
        sma = tp.rolling(window, min_periods=window).mean()
        mad = tp.rolling(window, min_periods=window).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        return (tp - sma) / (0.015 * mad + 1e-9)

    # ────────────────────────────────────────────────────────
    #  特徵新增步驟
    # ────────────────────────────────────────────────────────

    def _require_cols(self, *cols: str) -> bool:
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            print(f"  ✗ 缺少欄位: {missing}")
            return False
        return True

    def _add_rsi_features(self):
        if not self._require_cols("收盤價元"): return False
        for w in [5, 10, 20, 60, 120, 240]:
            self.df[f"RSI_{w}"] = (
                self.df.groupby("證券代碼")["收盤價元"]
                .transform(lambda x: self._rsi(x, w))
                .fillna(-1)
            )

    def _add_moving_average_features(self):
        if not self._require_cols("收盤價元"): return False
        c = "收盤價元"
        for w in [5, 10, 20, 60, 120, 240]:
            sma = self.df.groupby("證券代碼")[c].transform(lambda x: self._sma(x, w))
            ema = self.df.groupby("證券代碼")[c].transform(lambda x: self._ema(x, w))
            self.df[f"SMA_{w}"] = sma.fillna(0)
            self.df[f"EMA_{w}"] = ema.fillna(0)
            self.df[f"Price_SMA{w}_Ratio"] = ((self.df[c] / (sma + 1e-9)) - 1) * 100
            self.df[f"Price_SMA{w}_Ratio"] = self.df[f"Price_SMA{w}_Ratio"].fillna(0)

    def _add_macd_features(self):
        if not self._require_cols("收盤價元"): return False

        def _calc(g):
            line, sig, hist = self._macd(g)
            return pd.DataFrame({"MACD": line, "MACD_Signal": sig, "MACD_Hist": hist}, index=g.index)

        result = self.df.groupby("證券代碼")["收盤價元"].apply(_calc).reset_index(level=0, drop=True)
        for col in ["MACD", "MACD_Signal", "MACD_Hist"]:
            self.df[col] = result[col].fillna(0)

    def _add_kd_features(self):
        h, l, c = "最高價元", "最低價元", "收盤價元"
        if not self._require_cols(h, l, c): return False

        def _calc(g):
            k, d = self._kd(g[h], g[l], g[c])
            return pd.DataFrame({"K": k, "D": d}, index=g.index)

        result = self.df.groupby("證券代碼").apply(_calc).reset_index(level=0, drop=True)
        self.df["K"]       = result["K"].fillna(50)
        self.df["D"]       = result["D"].fillna(50)
        self.df["KD_Diff"] = self.df["K"] - self.df["D"]

    def _add_williams_r_features(self):
        h, l, c = "最高價元", "最低價元", "收盤價元"
        if not self._require_cols(h, l, c): return False
        for w in [14, 28]:
            self.df[f"Williams_R_{w}"] = (
                self.df.groupby("證券代碼")
                .apply(lambda g: self._williams_r(g[h], g[l], g[c], w))
                .reset_index(level=0, drop=True)
                .fillna(-50)
            )

    def _add_roc_momentum_features(self):
        if not self._require_cols("收盤價元"): return False
        c = "收盤價元"
        for w in [5, 10, 20]:
            self.df[f"ROC_{w}"]      = self.df.groupby("證券代碼")[c].transform(lambda x: self._roc(x, w)).fillna(0)
            self.df[f"Momentum_{w}"] = self.df.groupby("證券代碼")[c].transform(lambda x: self._momentum(x, w)).fillna(0)

    def _add_volatility_features(self):
        h, l, c = "最高價元", "最低價元", "收盤價元"
        if not self._require_cols(h, l, c): return False

        for w in [14, 28]:
            atr = self.df.groupby("證券代碼").apply(
                lambda g: self._atr(g[h], g[l], g[c], w)
            ).reset_index(level=0, drop=True)
            self.df[f"ATR_{w}"]     = atr.fillna(0)
            self.df[f"ATR_{w}_Pct"] = (atr / (self.df[c] + 1e-9) * 100).fillna(0)

        def _calc_bb(g):
            upper, mid, lower, pct_b = self._bollinger(g, 20)
            return pd.DataFrame({
                "BB_Upper_20": upper, "BB_Middle_20": mid,
                "BB_Lower_20": lower, "BB_PctB_20":   pct_b,
            }, index=g.index)

        bb = self.df.groupby("證券代碼")[c].apply(_calc_bb).reset_index(level=0, drop=True)
        for col in bb.columns:
            self.df[col] = bb[col].fillna(0)

    def _add_volume_features(self):
        h, l, c = "最高價元", "最低價元", "收盤價元"
        vol_col = "成交量千股"
        if vol_col not in self.df.columns:
            candidates = [col for col in self.df.columns if "成交量" in col or "volume" in col.lower()]
            if candidates:
                vol_col = candidates[0]
            else:
                print("  ⚠ 找不到成交量欄位，跳過")
                return
        if not self._require_cols(h, l, c, vol_col): return

        self.df["OBV"] = (
            self.df.groupby("證券代碼")
            .apply(lambda g: self._obv(g[c], g[vol_col]))
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        self.df["OBV_ROC"] = (
            self.df.groupby("證券代碼")["OBV"]
            .transform(lambda x: x.pct_change(5) * 100)
            .fillna(0)
        )
        self.df["VWAP"] = (
            self.df.groupby("證券代碼")
            .apply(lambda g: self._vwap(g[h], g[l], g[c], g[vol_col]))
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        self.df["Price_VWAP_Ratio"] = (self.df[c] / (self.df["VWAP"] + 1e-9) - 1) * 100

        for w in [5, 20]:
            sma_v = self.df.groupby("證券代碼")[vol_col].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            self.df[f"Volume_SMA_{w}"]   = sma_v.fillna(0)
            self.df[f"Volume_Ratio_{w}"] = (self.df[vol_col] / (sma_v + 1e-9)).fillna(0)

    def _add_cci_features(self):
        h, l, c = "最高價元", "最低價元", "收盤價元"
        if not self._require_cols(h, l, c): return False
        for w in [14, 20]:
            self.df[f"CCI_{w}"] = (
                self.df.groupby("證券代碼")
                .apply(lambda g: self._cci(g[h], g[l], g[c], w))
                .reset_index(level=0, drop=True)
                .fillna(0)
            )

    # ────────────────────────────────────────────────────────
    #  風險特徵：Beta（Vasicek）& vol20（年化）
    # ────────────────────────────────────────────────────────

    def _add_beta_vol_features(self):
        """
        daily_return（backward-looking，收盤價 pct_change）
        → 作為 beta 與 vol20 的共同基礎，嚴格與 forward-looking return label 切割

        vol20 = daily_return 的 20 日滾動標準差 × √252（年化日波動率）

        beta（Vasicek shrinkage，5步驟）：
          A. rolling_beta_raw = Cov(r_stock, r_market) / Var(r_market)，視窗 = beta_window
          B. 估計誤差代理 = 各股在當前視窗內 beta_raw 的時序滾動 var
          C. 截面方差 var_cross = 同一交易日所有股票 beta_raw 的截面 var
          D. Vasicek weight w = var_cross / (var_cross + var_estimation)
             → w → 1：相信個股 beta；w → 0：噪音大，縮向市場 beta = 1
          E. beta = clip(w × beta_raw + (1 − w) × 1.0, −5, 5)
             缺值填 1.0（market beta，語意中性）
        """
        if not self._require_cols("收盤價元", "market_return"):
            return False

        window = self.config.beta_window
        min_p  = max(window // 2, 10)

        # ── daily_return（backward-looking）────────────────────────────
        self.df["daily_return"] = (
            self.df.groupby("證券代碼")["收盤價元"]
            .transform(lambda x: x.pct_change())
            .fillna(0)
        )

        # ── vol20：20 日滾動標準差 × √252（年化）──────────────────────
        self.df["vol20"] = (
            self.df.groupby("證券代碼")["daily_return"]
            .transform(lambda x: x.rolling(20, min_periods=10).std() * np.sqrt(252))
            .fillna(0)
        )

        # ── Step A：rolling beta_raw ─────────────────────────────────
        def _rolling_beta_raw(group: pd.DataFrame) -> pd.Series:
            r   = group["daily_return"]
            mkt = group["market_return"]
            cov = r.rolling(window, min_periods=min_p).cov(mkt)
            var = mkt.rolling(window, min_periods=min_p).var()
            return (cov / (var + 1e-9)).clip(-5, 5)

        beta_raw = (
            self.df.groupby("證券代碼", group_keys=False)
            .apply(_rolling_beta_raw)
        )
        self.df["_beta_raw"] = beta_raw.values

        # ── Step B：per-stock 估計誤差代理（beta_raw 時序滾動 var）────
        var_estimation = (
            self.df.groupby("證券代碼")["_beta_raw"]
            .transform(lambda x: x.rolling(window, min_periods=min_p).std() ** 2)
            .fillna(method="bfill")
            .fillna(0.25)   # fallback：假設 beta 估計標準誤 ≈ 0.5，var ≈ 0.25
        )

        # ── Step C：截面方差（同一交易日所有股票的 beta_raw var）───────
        var_cross = (
            self.df.groupby("年月日")["_beta_raw"]
            .transform("var")
            .fillna(0)
        )

        # ── Step D & E：Vasicek shrinkage ──────────────────────────
        w = (var_cross / (var_cross + var_estimation + 1e-9)).clip(0, 1)
        beta_shrunk = (w * self.df["_beta_raw"] + (1 - w) * 1.0).clip(-5, 5)
        self.df["beta"] = beta_shrunk.fillna(1.0)

        # 清理暫存欄位
        self.df = self.df.drop(columns=["_beta_raw"], errors="ignore")

        print(f"  beta  視窗: {window}日 | Vasicek 後均值: {self.df['beta'].mean():.3f} | std: {self.df['beta'].std():.3f}")
        print(f"  vol20 年化 | 均值: {self.df['vol20'].mean():.4f} | std: {self.df['vol20'].std():.4f}")

    # ────────────────────────────────────────────────────────
    #  Return Labels（含 excess_return）
    # ────────────────────────────────────────────────────────

    def _add_return_features(self):
        """
        return = (open[T+2] − open[T+1]) / open[T+1]，clip ±return_clip

        Labels:
            return_tick      : return > tick_threshold   → 1（預設 1%）
            return_tick_0    : return > tick0_threshold  → 1（預設 > 0）

        Excess Return（去除系統性大盤影響）:
            excess_return = return − beta × market_return_fwd
                            clip ±return_clip
            → market_return_fwd 為 T+1 日大盤報酬，對應 return 的持有期間
            → 數學上等同 CAPM 殘差：α + ε（idiosyncratic component）

            excess_return_tick : excess_return > tick_threshold → 1

        ⚠ market_return_fwd 在本步驟結束時強制 drop，防止 look-ahead bias 外洩至特徵空間。
        """
        if not self._require_cols("開盤價元", "market_return_fwd", "beta"):
            return False

        cfg = self.config

        # ── return ──────────────────────────────────────────────────────
        def _calc(g):
            t1 = g["開盤價元"].shift(-1)
            t2 = g["開盤價元"].shift(-2)
            return (t2 - t1) / (t1 + 1e-9)

        ret = self.df.groupby("證券代碼").apply(_calc).reset_index(level=0, drop=True)
        self.df["return"] = ret.clip(-cfg.return_clip, cfg.return_clip)

        # ── return_tick / return_tick_0 ─────────────────────────────────
        valid = self.df["return"].notna()
        self.df["return_tick"]   = np.where(valid, (self.df["return"] > cfg.tick_threshold).astype(int),  np.nan)
        self.df["return_tick_0"] = np.where(valid, (self.df["return"] > cfg.tick0_threshold).astype(int), np.nan)

        # ── excess_return = return − beta × market_return_fwd ───────────
        # market_return_fwd：T+1 大盤報酬，與 return 的持有期 [T+1, T+2] 對應
        self.df["excess_return"] = (
            self.df["return"] - self.df["beta"] * self.df["market_return_fwd"]
        ).clip(-cfg.return_clip, cfg.return_clip)

        excess_valid = self.df["excess_return"].notna()
        self.df["excess_return_tick"] = np.where(
            excess_valid,
            (self.df["excess_return"] > cfg.tick_threshold).astype(int),
            np.nan
        )

        # ── ⚠ drop market_return_fwd（look-ahead bias 防護）────────────
        self.df = self.df.drop(columns=["market_return_fwd"], errors="ignore")
        print(f"  market_return_fwd dropped（look-ahead bias 防護）")

        # ── 統計輸出 ────────────────────────────────────────────────────
        pos_rate        = (self.df["return_tick"]        == 1).sum() / valid.sum()
        pos_rate0       = (self.df["return_tick_0"]      == 1).sum() / valid.sum()
        pos_excess_rate = (self.df["excess_return_tick"] == 1).sum() / excess_valid.sum()

        print(f"  return_tick        (>{cfg.tick_threshold:.1%}) 正樣本率: {pos_rate:.2%}")
        print(f"  return_tick_0      (>{cfg.tick0_threshold:.1%}) 正樣本率: {pos_rate0:.2%}")
        print(f"  excess_return_tick (>{cfg.tick_threshold:.1%}) 正樣本率: {pos_excess_rate:.2%}")

    # ────────────────────────────────────────────────────────
    #  摘要輸出
    # ────────────────────────────────────────────────────────

    def _print_summary(self):
        df = self.df
        indicator_cols = [
            c for c in df.columns if any(
                k in c for k in ["RSI", "SMA", "EMA", "MACD", "KD", "Williams",
                                  "ROC", "Momentum", "ATR", "BB_", "OBV", "VWAP", "CCI",
                                  "beta", "vol20", "market", "daily_return"]
            )
        ]
        look_ahead_ok = "market_return_fwd" not in df.columns
        print(f"\n{'='*60}")
        print(f"  ✓ 完成")
        print(f"  形狀          : {df.shape}")
        print(f"  股票數量      : {df['證券代碼'].nunique()}")
        print(f"  日期範圍      : {df['年月日'].min().date()} ~ {df['年月日'].max().date()}")
        print(f"  技術指標欄位數: {len(indicator_cols)}")
        print(f"  Look-ahead 防護: {'✓ market_return_fwd 已 drop' if look_ahead_ok else '✗ 警告: market_return_fwd 仍存在！'}")
        print(f"{'='*60}\n")


# ============================================================
#  直接執行（快速測試用）
# ============================================================

# if __name__ == "__main__":
#     config = PreprocessConfig(
#         start_year=2014,
#         end_year=2016,
#         return_clip=0.15,
#         tick_threshold=0.01,
#         tick0_threshold=0.00,
#         beta_window=60,
#     )
#     preprocessor = DataPreprocessor(config)
#     df = preprocessor.run()
#     print(df.head())