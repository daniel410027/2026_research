"""
db_loader.py
============
本地資料庫載入器（替換 finlab_loader.py）。

讀取 download.py 存出的 database/YYYY.csv，
重命名欄位為 FeatureMixin 相容的內部短名，
計算 market_return / market_return_fwd，
回傳 long-format DataFrame。

database/ 目錄結構（download.py 輸出）：
    database/2014.csv
    database/2015.csv
    ...

每個 CSV 欄位：
    code, date,
    etl_adj_close, etl_adj_open, price_收盤價, price_最高價, price_最低價,
    price_成交股數, price_成交金額, etl_market_value,
    benchmark_return,              ← macro（所有 code 同日共用）
    intraday_stat_sell_pct,        ← macro
    institutional_investors_trading_summary_外陸資買賣超股數_不含外資自營商,
    ...（依 finlab_datasets.csv 自動命名）

欄位命名規則（download.py auto-gen）：
    dataset_name 中的特殊字元（: ( ) %）→ _，再 strip("_")

作者：Daniel Huang
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
#  CSV 欄位 → 內部短名對照表
#  key   = download.py auto-gen col_name（CSV 欄位）
#  value = FeatureMixin 使用的內部名稱
# ─────────────────────────────────────────────────────────────

COL_MAP: dict[str, str] = {
    # ── key columns ──────────────────────────────────────────
    "code"  : "證券代碼",
    "date"  : "年月日",

    # ── 還原後價格（etl:*）───────────────────────────────────
    "etl_adj_close"    : "adj_close",
    "etl_adj_open"     : "adj_open",
    "etl_market_value" : "market_value",

    # ── 未還原 OHLCV（price:*）──────────────────────────────
    "price_收盤價"   : "close",
    "price_最高價"   : "high",
    "price_最低價"   : "low",
    "price_成交股數" : "volume",
    "price_成交金額" : "amount",

    # ── Macro（全市場廣播）──────────────────────────────────
    "benchmark_return"       : "market_index",
    "intraday_stat_sell_pct" : "intraday_sell_pct",

    # ── 三大法人 ────────────────────────────────────────────
    "institutional_investors_trading_summary_外陸資買賣超股數_不含外資自營商": "fii_net",
    "institutional_investors_trading_summary_自營商買賣超股數_自行買賣":       "dealer_net",
    "institutional_investors_trading_summary_投信買賣超股數":                  "trust_net",

    # ── 融資融券 ────────────────────────────────────────────
    "margin_transactions_融資買進"     : "margin_buy",
    "margin_transactions_融資賣出"     : "margin_sell",
    "margin_transactions_融資今日餘額" : "margin_balance",
    "margin_transactions_融資使用率"   : "margin_util",
    "margin_transactions_融券今日餘額" : "ms_balance",
    "margin_transactions_融券限額"     : "ms_limit",

    # ── 籌碼 ────────────────────────────────────────────────
    "foreign_investors_shareholding_發行股數"             : "fii_shares",
    "foreign_investors_shareholding_外資及陸資尚可投資比率": "fii_investable",
    "internal_equity_changes_董監持有股數"                 : "insider_shares",

    # ── 借券 ────────────────────────────────────────────────
    "security_lending_sell_借券賣出餘額": "sl_sell_balance",
    "security_lending_借券餘額"         : "sl_balance",

    # ── 當沖 ────────────────────────────────────────────────
    "intraday_trading_當日沖銷交易成交股數": "intraday_shares",

    # ── 月營收 ──────────────────────────────────────────────
    "monthly_revenue_當月營收"    : "monthly_revenue",
    "monthly_revenue_當月累計營收": "monthly_revenue_cum",
    "monthly_revenue_去年累計營收": "monthly_revenue_ly_cum",
    "monthly_revenue_上月比較增減": "monthly_revenue_mom",   # 原 上月比較增減(%) → strip → 上月比較增減

    # ── 財報 ────────────────────────────────────────────────
    "financial_statement_每股盈餘"    : "eps",
    "financial_statement_股東權益總額": "equity",
    "financial_statement_稅前淨利"    : "pretax_income",

    # ── 估值 ────────────────────────────────────────────────
    "price_earning_ratio_股價淨值比": "pbr",

    # ── 停券旗標（0/1）──────────────────────────────────────
    "margin_short_sale_suspension": "margin_suspension",
}


# ─────────────────────────────────────────────────────────────
#  DBLoader
# ─────────────────────────────────────────────────────────────

class DBLoader:
    """
    讀取 download.py 存出的 database/YYYY.csv。

    Parameters
    ----------
    start_year : int
    end_year   : int
    db_dir     : str | Path   database/ 目錄路徑（預設 "database"）
    """

    def __init__(
        self,
        start_year: int,
        end_year:   int,
        db_dir:     str | Path = "database",
    ):
        self.start_year = start_year
        self.end_year   = end_year
        self.db_dir     = Path(db_dir)

    # ──────────────────────────────────────────────────────────
    #  公開介面
    # ──────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """
        載入 [start_year, end_year] 的所有年度 CSV，
        重命名欄位，計算 market_return / market_return_fwd，
        回傳 long-format DataFrame。

        Returns
        -------
        pd.DataFrame
            欄位：年月日, 證券代碼, adj_close, adj_open, close, high, low,
                  volume, amount, market_index, market_return,
                  market_return_fwd, fii_net, dealer_net, ...
        """
        dfs = []
        for year in range(self.start_year, self.end_year + 1):
            path = self.db_dir / f"{year}.csv"
            if not path.exists():
                print(f"  ⚠ 找不到 {path.name}，跳過")
                continue
            df = pd.read_csv(path, low_memory=False)
            dfs.append(df)
            print(f"  ✓ 讀取 {path.name}  {df.shape}")

        if not dfs:
            raise FileNotFoundError(
                f"未找到任何年度 CSV，請確認路徑: {self.db_dir}\n"
                f"預期檔案: {self.db_dir}/{self.start_year}.csv ~ {self.end_year}.csv\n"
                f"請先執行 download.py 下載資料"
            )

        df = pd.concat(dfs, ignore_index=True)
        print(f"\n  合併: {len(df):,} 筆，欄位: {df.shape[1]}")

        df = self._rename_columns(df)
        df = self._fix_types(df)
        df = self._add_market_returns(df)

        print(
            f"  載入完成: {len(df):,} 筆，"
            f"{df['證券代碼'].nunique()} 檔股票，"
            f"欄位數: {df.shape[1]}"
        )
        return df.sort_values(["年月日", "證券代碼"]).reset_index(drop=True)

    def latest_date(self) -> pd.Timestamp:
        """
        回傳資料庫中最新的交易日（供 DailyTrainer 查詢 predict_date）。
        掃描最近一個存在的年度 CSV，取其最後一個 date。
        """
        for year in range(self.end_year, self.start_year - 1, -1):
            path = self.db_dir / f"{year}.csv"
            if path.exists():
                df = pd.read_csv(path, usecols=["date"], low_memory=False)
                return pd.to_datetime(df["date"]).max()
        raise FileNotFoundError(f"找不到任何年度 CSV in {self.db_dir}")

    # ──────────────────────────────────────────────────────────
    #  內部工具
    # ──────────────────────────────────────────────────────────

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """依 COL_MAP 重命名欄位，不在 map 中的欄位保持原名（如 inventory_* 等）。"""
        rename = {k: v for k, v in COL_MAP.items() if k in df.columns}
        df = df.rename(columns=rename)

        missing = [v for k, v in COL_MAP.items()
                   if k not in df.columns and k not in ("code", "date")]
        if missing:
            print(f"  ⚠ CSV 缺少以下欄位（可能未在 download.py 中下載）：")
            for m in missing:
                print(f"      {m}")
        return df

    def _fix_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """確保關鍵欄位型別正確。"""
        df["年月日"]   = pd.to_datetime(df["年月日"])
        df["證券代碼"] = df["證券代碼"].astype(str)

        # 所有非 key 欄位強制轉 numeric（CSV 讀取可能帶 object dtype）
        for col in df.columns:
            if col in ("年月日", "證券代碼"):
                continue
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _add_market_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        由 market_index（發行量加權報酬指數）計算：
            market_return     = pct_change（當日大盤報酬）
            market_return_fwd = market_return.shift(-1)（T+1 大盤報酬）
        兩者廣播至所有股票。
        """
        if "market_index" not in df.columns:
            print("  ⚠ 未找到 market_index，market_return 無法計算")
            df["market_return"]     = 0.0
            df["market_return_fwd"] = 0.0
            return df

        mkt = (
            df.groupby("年月日")["market_index"]
            .first()
            .sort_index()
        )
        mkt_ret     = mkt.pct_change().rename("market_return")
        mkt_ret_fwd = mkt_ret.shift(-1).rename("market_return_fwd")

        mkt_df = pd.DataFrame({
            "market_return":     mkt_ret,
            "market_return_fwd": mkt_ret_fwd,
        }).reset_index()

        df = df.merge(mkt_df, on="年月日", how="left")
        df["market_return"]     = df["market_return"].fillna(0)
        df["market_return_fwd"] = df["market_return_fwd"].fillna(0)
        return df
