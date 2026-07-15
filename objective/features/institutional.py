"""
institutional.py
=================
三大法人 / 融資融券 / 借券衍生特徵。
"""

from __future__ import annotations

import numpy as np

from .. import ta_indicators as ta
from ._base import _FeatureHelperMixin, _RCLOSE, _VOL


class InstitutionalFeaturesMixin(_FeatureHelperMixin):
    """法人籌碼 / 信用交易相關 _add_* 方法。繼承類別需提供：self.df, self.config"""

    # ──────────────────────────────────────────────────────────
    #  三大法人 / 融資融券衍生特徵
    # ──────────────────────────────────────────────────────────

    def _add_institutional_flow_features(self):
        """
        合計買賣超金額千元      = (fii_net + dealer_net + trust_net) × close / 1000
        投信買賣超市值百萬       = trust_net × close / 1,000,000
        融資買賣成交量           = margin_buy + margin_sell
        融資買進千元             = margin_buy × close × 1000 / 1000
        整戶維持率（近似）       = (融資擔保市值 + 融券擔保) / (原融資金額 + 融券市值) × 100
                                   融資成數 60%；融券保證金+擔保價款 190%（以現價近似成本）
        融資增減千元             = (margin_balance - margin_balance.shift(1)) × close / 1000
        尚可投資比率TSE          = fii_investable
        董監持股數               = insider_shares
        外資本年以來買賣超千股   = YTD cumulative fii_net / 1000
        自營本週以來買賣超千股   = WTD cumulative dealer_net / 1000
        自營本月以來買賣超千股   = MTD cumulative dealer_net / 1000
        自營連續累計買賣超千     = cumulative dealer_net / 1000
        信用交易比重             = (margin_buy + margin_sell + ms_balance) / volume
        """
        df = self.df

        # ① 合計買賣超金額千元（三大法人）
        if all(c in df.columns for c in ["fii_net", "dealer_net", "trust_net", _RCLOSE]):
            total_net = df["fii_net"] + df["dealer_net"] + df["trust_net"]
            df["合計買賣超金額千元"] = (total_net * df[_RCLOSE] / 1000).fillna(0)
            print("    合計買賣超金額千元 ✓")

        # ② 投信買賣超市值百萬
        if all(c in df.columns for c in ["trust_net", _RCLOSE]):
            df["投信買賣超市值百萬"] = (df["trust_net"] * df[_RCLOSE] / 1e6).fillna(0)
            print("    投信買賣超市值百萬 ✓")

        # ③ 融資買賣成交量（張）
        if all(c in df.columns for c in ["margin_buy", "margin_sell"]):
            df["融資買賣成交量"] = (df["margin_buy"] + df["margin_sell"]).fillna(0)
            print("    融資買賣成交量 ✓")

        # ④ 融資買進千元
        if all(c in df.columns for c in ["margin_buy", _RCLOSE]):
            df["融資買進千元"] = (df["margin_buy"] * df[_RCLOSE]).fillna(0)
            print("    融資買進千元 ✓")

        # ⑤ 融資使用率（直接取用）
        if "margin_util" in df.columns:
            df["融資使用率"] = df["margin_util"].fillna(0)
            print("    融資使用率 ✓")

        # ⑥ 融資增減千元（餘額日變動）
        if all(c in df.columns for c in ["margin_balance", _RCLOSE]):
            margin_chg = (
                df.groupby("證券代碼")["margin_balance"]
                .transform(lambda x: x.diff())
            )
            df["融資增減千元"] = (margin_chg * df[_RCLOSE] / 1000).fillna(0)
            print("    融資增減千元 ✓")

        # ⑦ 整戶維持率（近似公式）
        # = (融資擔保證券市值 + 原融券擔保價款及保證金) / (原融資金額 + 融券證券市值) × 100
        # 近似假設：融資成數 60%；融券保證金+擔保價款 = 190%（以現價代替成本價）
        if all(c in df.columns for c in ["margin_balance", "ms_balance", _RCLOSE]):
            close     = df[_RCLOSE]
            mkt_long  = df["margin_balance"].clip(lower=0) * close   # 融資擔保證券市值
            mkt_short = df["ms_balance"].clip(lower=0) * close       # 融券證券市值
            原融資    = mkt_long * 0.6                                # 融資成數 60%
            融券擔保  = mkt_short * 1.9                               # 保證金 90% + 擔保價款 100%
            df["整戶維持率"] = (
                (mkt_long + 融券擔保) / (原融資 + mkt_short + 1e-9) * 100
            ).clip(0, 500).fillna(130.0)
            print("    整戶維持率（近似：TWSE 公式）✓")

        # ⑧ 信用交易比重
        if all(c in df.columns for c in ["margin_buy", "margin_sell", "ms_balance", _VOL]):
            credit_vol = df["margin_buy"] + df["margin_sell"] + df["ms_balance"]
            df["信用交易比重"] = (credit_vol / (df[_VOL] + 1e-9)).clip(0, 1).fillna(0)
            print("    信用交易比重 ✓")

        # ⑨ 尚可投資比率TSE
        if "fii_investable" in df.columns:
            df["尚可投資比率TSE"] = df["fii_investable"].fillna(0)
            print("    尚可投資比率TSE ✓")

        # ⑩ 董監持股數（千股單位標準化）
        if "insider_shares" in df.columns:
            df["董監持股數"] = (df["insider_shares"] / 1000).fillna(0)
            print("    董監持股數 ✓")

        # ⑪ 自營累計買賣超（連續 / 本月 / 本週）
        if "dealer_net" in df.columns:
            # 連續累計（符號翻轉重置）
            df["自營連續累計買賣超千"] = (
                df.groupby("證券代碼")["dealer_net"]
                .transform(lambda x: ta.consecutive_cumsum(x) / 1000)
                .fillna(0)
            )
            # 本週以來
            # ★ 2026-07-13 清理：移除 dead 欄位 dealer_net_wtd
            #   （純複製 dealer_net、下游從未引用，KEEP_FEATURES 白名單亦會丟棄）。
            week_key = df["年月日"].dt.to_period("W")
            df["自營本週以來買賣超千股"] = (
                df.groupby(["證券代碼", week_key])["dealer_net"]
                .transform("cumsum") / 1000
            ).fillna(0)
            # 本月以來
            month_key = df["年月日"].dt.to_period("M")
            df["自營本月以來買賣超千股"] = (
                df.groupby(["證券代碼", month_key])["dealer_net"]
                .transform("cumsum") / 1000
            ).fillna(0)
            print("    自營累計買賣超（連續sign-reset/週/月）✓")

        # ⑫ 外資買賣超累計（連續 sign-reset + YTD 兩版本）
        if "fii_net" in df.columns:
            # ── 連續累計（符號翻轉重置）→ 對應 finlab「外資連續累計買賣超張1」
            df["外資連續累計買賣超張1"] = (
                df.groupby("證券代碼")["fii_net"]
                .transform(lambda x: ta.consecutive_cumsum(x) / 1000)
                .fillna(0)
            )
            # ── YTD 累計（每年元旦重置）→ 另存備用
            year_key = df["年月日"].dt.year
            df["外資本年以來買賣超千股"] = (
                df.groupby(["證券代碼", year_key])["fii_net"]
                .transform("cumsum") / 1000
            ).fillna(0)
            print("    外資連續累計買賣超張1（sign-reset）& 外資本年以來買賣超千股（YTD）✓")

        # ⑬ 三大法人買超張 / 賣超張
        if all(c in df.columns for c in ["fii_net", "dealer_net", "trust_net"]):
            total = df["fii_net"] + df["dealer_net"] + df["trust_net"]
            df["三大法人買超張"]  = total.clip(lower=0).fillna(0)
            df["三大法人賣超張"]  = (-total).clip(lower=0).fillna(0)

        # ⑭ 股價淨值比
        if "pbr" in df.columns:
            df["股價淨值比TEJ"] = df["pbr"].fillna(0)

        # ⑮ 融資買進張（margin_buy 直接，單位：千股）
        if "margin_buy" in df.columns:
            df["融資買進張"] = df["margin_buy"].fillna(0)

        # ⑯ 外資賣出張數（fii_net 負值部分取絕對值）
        if "fii_net" in df.columns:
            df["外資賣出張數"] = (-df["fii_net"]).clip(lower=0).fillna(0)

        # ⑰ 投信成交比重 = |trust_net| / volume
        if all(c in df.columns for c in ["trust_net", _VOL]):
            df["投信成交比重"] = (
                df["trust_net"].abs() / (df[_VOL] + 1e-9)
            ).clip(0, 1).fillna(0)

        # ⑱ 法人買賣超日數1：近 20 日三大法人合計買超天數
        if all(c in df.columns for c in ["fii_net", "dealer_net", "trust_net"]):
            is_buy = (
                (df["fii_net"] + df["dealer_net"] + df["trust_net"]) > 0
            ).astype(int)
            df["法人買賣超日數1"] = (
                is_buy.groupby(df["證券代碼"])
                .transform(lambda x: x.rolling(20, min_periods=1).sum())
                .fillna(0)
            )

        # ⑲ 自營自行買賣超_年千股（YTD 累計，與 外資本年以來 對應）
        if "dealer_net" in df.columns:
            year_key = df["年月日"].dt.year
            df["自營自行買賣超_年千股"] = (
                df.groupby(["證券代碼", year_key])["dealer_net"]
                .transform("cumsum") / 1000
            ).fillna(0)

        # ⑳ 自營自行買賣超_月千股（MTD 累計，同 自營本月以來買賣超千股，補別名）
        if "自營本月以來買賣超千股" in df.columns:
            df["自營自行買賣超_月千股"] = df["自營本月以來買賣超千股"]
        elif "dealer_net" in df.columns:
            month_key = df["年月日"].dt.to_period("M")
            df["自營自行買賣超_月千股"] = (
                df.groupby(["證券代碼", month_key])["dealer_net"]
                .transform("cumsum") / 1000
            ).fillna(0)

        # ㉑ 合計本月買賣超金額千元（MTD 累計三大法人 × 收盤）
        if all(c in df.columns for c in ["fii_net", "dealer_net", "trust_net", _RCLOSE]):
            total_net_val = (df["fii_net"] + df["dealer_net"] + df["trust_net"]) * df[_RCLOSE]
            month_key = df["年月日"].dt.to_period("M")
            df["合計本月買賣超金額千元"] = (
                total_net_val.groupby([df["證券代碼"], month_key])
                .transform("cumsum") / 1000
            ).fillna(0)

        # ㉒ 外資本週以來買賣超千股（WTD 累計）
        if "fii_net" in df.columns:
            week_key = df["年月日"].dt.to_period("W")
            df["外資本週以來買賣超千股"] = (
                df.groupby(["證券代碼", week_key])["fii_net"]
                .transform("cumsum") / 1000
            ).fillna(0)
            print("    外資本週以來買賣超千股（WTD）✓")

        # ㉓ 自營本週買賣超金額千元（WTD dealer_net 張數 × 收盤價，累計後 /1000 → 千元）
        if all(c in df.columns for c in ["dealer_net", _RCLOSE]):
            dealer_amt = df["dealer_net"] * df[_RCLOSE]
            week_key = df["年月日"].dt.to_period("W")
            df["自營本週買賣超金額千元"] = (
                dealer_amt.groupby([df["證券代碼"], week_key])
                .transform("cumsum") / 1000
            ).fillna(0)
            print("    自營本週買賣超金額千元（WTD）✓")

        # ㉔ 合計本週買賣超金額千元（WTD 三大法人合計張數 × 收盤價，累計後 /1000 → 千元）
        if all(c in df.columns for c in ["fii_net", "dealer_net", "trust_net", _RCLOSE]):
            total_net_val_wtd = (df["fii_net"] + df["dealer_net"] + df["trust_net"]) * df[_RCLOSE]
            week_key = df["年月日"].dt.to_period("W")
            df["合計本週買賣超金額千元"] = (
                total_net_val_wtd.groupby([df["證券代碼"], week_key])
                .transform("cumsum") / 1000
            ).fillna(0)
            print("    合計本週買賣超金額千元（WTD）✓")

        self.df = df.copy()   # 整合碎片化欄位

    # ──────────────────────────────────────────────────────────
    #  借券 / 融券特徵（FinLab 版）
    # ──────────────────────────────────────────────────────────

    def _add_short_selling_features(self):
        """
        FinLab 版本的借券 / 融券特徵。

        使用欄位（來自 DBLoader / database/YYYY.csv）：
            ms_balance      融券今日餘額（張）
            ms_limit        融券限額（張）
            sl_sell_balance 借券賣出餘額
            sl_balance      借券餘額（總）
            volume          成交股數

        計算：
            short_sell_volume_ratio = ms_balance / volume
            short_headroom          = 1 - ms_balance / ms_limit
            days_to_cover           = sl_sell_balance / avg_volume_20
            days_to_cover_rank      = 截面 percentile rank
            inst_retail_ratio       = sl_sell_balance / (sl_balance + 1e-9)
            short_net_flow_sec      = sl_sell_balance - sl_sell_balance.shift(1)
            借券賣出可使用額度_log  = log1p(ms_limit - ms_balance)
        """
        cfg = self.config
        df  = self.df

        # short_sell_volume_ratio
        if all(c in df.columns for c in ["ms_balance", _VOL]):
            df["short_sell_volume_ratio"] = (
                df["ms_balance"] / (df[_VOL] + 1e-9)
            ).clip(0, 1).fillna(0)

        # short_headroom：1 - 融券餘額 / 融券限額
        if all(c in df.columns for c in ["ms_balance", "ms_limit"]):
            df["short_headroom"] = (
                1 - df["ms_balance"] / (df["ms_limit"] + 1e-9)
            ).clip(0, 1).fillna(1.0)
            # 可使用額度 log
            headroom_lots = (df["ms_limit"] - df["ms_balance"]).clip(lower=0)
            df["借券賣出可使用額度_log"] = np.log1p(headroom_lots).fillna(0)

        # days_to_cover（借券賣出餘額 / 20日均量）
        if all(c in df.columns for c in ["sl_sell_balance", _VOL]):
            avg_vol_20 = (
                df.groupby("證券代碼")[_VOL]
                .transform(lambda x: x.rolling(20, min_periods=5).mean())
                .fillna(df[_VOL])
            )
            df["days_to_cover"] = (
                df["sl_sell_balance"] / (avg_vol_20 + 1e-9)
            ).clip(0, 250).fillna(0)
            df["days_to_cover_rank"] = (
                df.groupby("年月日")["days_to_cover"]
                .rank(method="average", pct=True)
            ).fillna(0.5)

        # inst_retail_ratio：借券賣出 / 借券總餘額
        if all(c in df.columns for c in ["sl_sell_balance", "sl_balance"]):
            df["inst_retail_ratio"] = (
                df["sl_sell_balance"] / (df["sl_balance"] + 1e-9)
            ).clip(0, 1).fillna(0)

        # short_net_flow_sec：借券賣出餘額日變動
        if "sl_sell_balance" in df.columns:
            df["short_net_flow_sec"] = (
                df.groupby("證券代碼")["sl_sell_balance"]
                .transform(lambda x: x.diff())
                .fillna(0)
            )

        # Rolling 特徵
        if "sl_sell_balance" in df.columns:
            for w in cfg.short_windows:
                min_p = max(w // 2, 3)
                df[f"short_balance_chg_{w}d"] = (
                    df.groupby("證券代碼")["sl_sell_balance"]
                    .transform(lambda x, _w=w: x.pct_change(_w, fill_method=None).clip(-2, 2))
                    .fillna(0)
                )
            roll_count = len(cfg.short_windows)
            print(f"    short_balance_chg rolling: {roll_count} 欄  windows={cfg.short_windows}")

        if cfg.short_log_transform:
            for col in ["sl_sell_balance", "sl_balance", "ms_balance"]:
                if col in df.columns:
                    df[f"{col}_log"] = np.log1p(df[col].clip(lower=0)).fillna(0)

        self.df = df

        exist = [c for c in [
            "short_sell_volume_ratio", "short_headroom", "days_to_cover",
            "days_to_cover_rank", "inst_retail_ratio", "short_net_flow_sec",
            "借券賣出可使用額度_log",
        ] if c in self.df.columns]
        print(f"    借券/融券衍生特徵: {len(exist)} 欄 ✓")
