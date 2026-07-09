"""
backtest.py
===========
Walk-Forward 回測模組。

─────────────────────────────────────────────
  資料流向（Data Flow）
─────────────────────────────────────────────
  [輸入 Input]
    database/experiment/YYYY_YYYY/predictions.csv
        欄位：證券代碼, 年月日, return, y_true, y_prob, y_pred
        return = 明日開盤 → 後日開盤（已排除 look-ahead bias）

    database/experiment/liquidity_filter_meta.json（選用）
        若 main_fix.py 有套用 805_group2 流動性篩選(formula3)，
        會在此留下 w1/w2/keep_ratio/tag 等溯源資訊；
        backtest.py 會自動讀取並印在報表開頭、同步存一份到
        output/backtest/，避免事後看報表誤判為全市場訓練結果。
        找不到時視為未篩選（舊版 pipeline 產出），僅印警告，不中斷。

    database_make/market_return_series.csv  （由 test.py 產生，需重新產生含 market_return_fwd 欄）
        或直接從 database_make/*.csv 讀取 market_return_fwd 欄位
        ★ 2026-06-25：改用 market_return_fwd（open[T+1]→open[T+2]),
          與 return 同窗對齊,取代舊版時間錯位的 market_return（close-to-close, lag-0）

  [處理 Process]
    1. 讀取所有 window 的 predictions.csv 並合併
    2. 去除重複（同股票同日期保留最早 window）
    3. 每個交易日依 y_prob 排序，分成 10 個 Decile 組
       Decile 1 = y_prob 最高（最看漲）
    4. 計算各組每日等權報酬，乘以資金使用率（CAPITAL_USAGE）
    5. 載入 market_return_fwd，對齊日期
    6. 計算 excess_return = D1_scaled - market_return_fwd
    7. 輸出指標（含 alpha / IR）與圖表

  [資金使用率 Capital Usage]
    CAPITAL_USAGE = 0.5 表示 50% 資金投入股票，50% 閒置（報酬=0）
    所有 Decile return 均乘以 CAPITAL_USAGE
    market_return_fwd 保持原始（代表 100% 持有大盤）
    → 比較時需注意 benchmark 與策略的槓桿差異

  [輸出 Output]
    output/backtest/
        decile_metrics.csv          10 組的 return/std/sharpe/mdd/alpha/IR 等指標
        cumulative_returns.png      10 組累積報酬曲線（含大盤基準線）
        annual_decile1.png          Decile 1 各年度表現長條圖（含大盤對照）
        decile_summary_bar.png      各組年化報酬（ln 形式）& Sharpe 橫向比較
        excess_return.png           D1 excess return 累積曲線
        drawdown_decile1.png        D1 水下曲線
        excess_drawdown_decile1.png D1 相對大盤回撤水下曲線（多少 / 多久）
        excess_drawdown_episodes.csv 每段水下期間明細（深度、谷底、追平、天數）
        daily_returns.csv           每日各組報酬（含 market、excess_D1 欄）

─────────────────────────────────────────────
  設定（Config）
─────────────────────────────────────────────
    EXPERIMENT_DIR  : predictions.csv 所在的上層目錄
    DATABASE_MAKE_DIR: market_return CSV 所在目錄
    OUTPUT_DIR      : 圖表與指標輸出目錄
    N_DECILES       : 分組數（預設 10）
    COST_BPS        : 單邊交易成本 bps（預設 0）
    ANNUAL_DAYS     : 年化交易日數（預設 252）
    CAPITAL_USAGE   : 資金使用率（預設 0.5，即 50%）

作者：Daniel Huang
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd


# ============================================================
#  中文字型（CJK font）設定
# ============================================================

def setup_cjk_font(verbose: bool = True) -> str | None:
    """自動偵測並設定支援中文的 matplotlib 字型，避免圖中中文變方框（tofu）。

    依序嘗試常見跨平台 CJK 字型；找到第一個「已安裝」者即套用。
    同時關閉 unicode_minus，避免負號顯示成方框。

    回傳套用的字型名稱；全部找不到回傳 None（中文仍會是方框）。
    可手動指定：把想要的字型名稱插到 candidates 最前面即可。
    """
    candidates = [
        # Windows
        "Microsoft JhengHei",   # 微軟正黑體（繁中）
        "Microsoft YaHei",      # 微軟雅黑（簡中）
        # macOS
        "PingFang TC", "PingFang SC", "Heiti TC", "Heiti SC",
        # Linux / 通用（Noto / 思源 / 文泉驛 / 台北黑體）
        "Taipei Sans TC Beta",
        "Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP",
        "Source Han Sans TC", "Source Han Sans CN", "Source Han Sans",
        "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
        "Droid Sans Fallback",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}

    chosen = next((name for name in candidates if name in installed), None)
    if chosen is not None:
        # 把選定字型放最前面，後面保留其餘候選作為缺字 fallback
        plt.rcParams["font.sans-serif"] = (
            [chosen] + [n for n in candidates if n in installed and n != chosen]
        )
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["axes.unicode_minus"] = False   # 負號正常顯示
        if verbose:
            print(f"  [font] 套用 CJK 字型：{chosen}")
        return chosen

    if verbose:
        print("  [font] [WARN] 找不到任何 CJK 字型，圖中中文可能顯示為方框。")
        print("         請安裝其一：Noto Sans CJK TC / 思源黑體 / 微軟正黑體，"
              "或自行把字型名稱加進 setup_cjk_font() 的 candidates。")
    return None


# ============================================================
#  設定
# ============================================================

EXPERIMENT_DIR    = Path("database/experiment")
DATABASE_MAKE_DIR = Path("database_make")
OUTPUT_DIR        = Path("output/backtest")
N_DECILES         = 10
COST_BPS          = 0       # 單邊交易成本（bps），每日換倉 × 2
ANNUAL_DAYS       = 252
CAPITAL_USAGE     = 1.0      # 資金使用率：1.0 = 滿倉投入（與大盤 100% 對等，便於解讀 alpha）

# ── β-neutral 超額報酬設定 ──────────────────────────────
#   excess_D1 = D1 − β·market（取代原本的算術差 D1 − market）
#   BETA_MODE:
#     "rolling"   滾動窗 β（ex-ante，無未來函數）← 預設，適合可交易避險解讀
#     "expanding" 擴張窗 β（用 t 之前所有資料，ex-ante）
#     "full"      全樣本單一 β（ex-post 歸因，含 look-ahead，不可作交易比例）
BETA_MODE         = "rolling"
BETA_WINDOW       = 252      # rolling 視窗（交易日）
BETA_MIN_PERIODS  = 60       # 估計 β 所需最少樣本

DATE_COL = "年月日"
# ★ 2026-06-25：改用 market_return_fwd（open[T+1]→open[T+2]）。
#   舊版 market_return 是「第 T 天 close-to-close 的大盤報酬」,
#   但 predictions.csv 的 return 是「open[T+1]→open[T+2]」的策略持倉報酬,
#   兩者時間窗不對齊（lag 且 close/open 基準不同）。
#   market_return_fwd 公式與策略 return 完全同構（皆為 open-to-open,同一窗口),
#   對齊後才能算出真正乾淨的 excess_return / alpha。
MKT_COL  = "market_return_fwd"


# ============================================================
#  工具函式
# ============================================================

def load_all_predictions(exp_dir: Path) -> pd.DataFrame:
    """讀取所有 window 的 predictions.csv，合併並去重。"""
    frames = []
    for path in sorted(exp_dir.glob("*/predictions.csv")):
        df = pd.read_csv(path, low_memory=False)
        df["_window"] = path.parent.name
        frames.append(df)
        print(f"  讀取 {path}  {df.shape}")

    if not frames:
        raise FileNotFoundError(f"{exp_dir} 下找不到任何 predictions.csv")

    combined = pd.concat(frames, ignore_index=True)
    combined[DATE_COL] = pd.to_datetime(combined[DATE_COL])
    combined["證券代碼"] = combined["證券代碼"].astype(str)
    combined = combined.sort_values(["證券代碼", DATE_COL, "_window"])

    before = len(combined)
    combined = combined.drop_duplicates(subset=["證券代碼", DATE_COL], keep="first")
    after = len(combined)
    if before != after:
        print(f"  去重：移除 {before - after:,} 筆重複（保留最早 window）")

    combined = combined.sort_values([DATE_COL, "證券代碼"]).reset_index(drop=True)
    print(f"\n  合併後：{len(combined):,} 筆  日期範圍 {combined[DATE_COL].min().date()} ~ {combined[DATE_COL].max().date()}")
    return combined


def load_market_return(db_make_dir: Path) -> pd.Series:
    """
    載入大盤日報酬序列（market_return_fwd, open[T+1]→open[T+2],與策略 return 同窗對齊）。
    優先讀取 test.py 產生的 market_return_series.csv；
    若不存在則直接從 db_make_dir/*.csv 提取（取各檔第一次出現的值）。

    ⚠ 若舊版 market_return_series.csv 仍是舊欄位（無 market_return_fwd),
      需先重跑 test.py / make_new.py 重新產生,否則 fallback 會找不到欄位而報錯。
    """
    cached = db_make_dir / "market_return_series.csv"
    if cached.exists():
        df = pd.read_csv(cached)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        series = df.set_index(DATE_COL)[MKT_COL].sort_index()
        print(f"  載入 {cached}  ({len(series)} 天)")
        return series

    # fallback：掃描目錄
    print(f"  {cached} 不存在，直接掃描 {db_make_dir}/*.csv ...")
    frames = []
    for path in sorted(db_make_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in [DATE_COL, MKT_COL])
        except Exception:
            continue
        if DATE_COL not in df.columns or MKT_COL not in df.columns:
            continue
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df[MKT_COL]  = pd.to_numeric(df[MKT_COL], errors="coerce")
        frames.append(df.groupby(DATE_COL)[MKT_COL].first())

    if not frames:
        raise FileNotFoundError(f"{db_make_dir} 下找不到含 {MKT_COL} 的 CSV")

    combined = pd.concat(frames, axis=1).median(axis=1).sort_index()
    combined.name = MKT_COL
    print(f"  fallback 掃描完成：{len(combined)} 天")
    return combined


def assign_deciles(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """每個交易日依 y_prob 降序分 n 組，Decile 1 = 最高 prob。
    Vectorized 實作，避免 pandas 2.x groupby.apply 的 index 問題。
    """
    df = df.copy().reset_index(drop=True)
    df["_rank"] = df.groupby(DATE_COL)["y_prob"].rank(method="first", ascending=False)
    df["_size"] = df.groupby(DATE_COL)["y_prob"].transform("count")
    mask = df["_size"] >= n
    df["decile"] = np.nan
    df.loc[mask, "decile"] = np.ceil(
        df.loc[mask, "_rank"] / df.loc[mask, "_size"] * n
    ).clip(1, n).astype(int)
    df["decile"] = df["decile"].astype("Int64")
    df = df.drop(columns=["_rank", "_size"])
    return df


def estimate_beta(
    strat: pd.Series,
    mkt: pd.Series,
    mode: str = "rolling",
    window: int = 252,
    min_periods: int = 60,
) -> pd.Series:
    """估計策略對大盤的 β = Cov(strat, mkt) / Var(mkt)，回傳對齊 strat 的 β 序列。

    mode
    ----
    "full"      全樣本單一 β（事後歸因；含 look-ahead，不可作為「可交易」避險比例）
    "expanding" 擴張窗 β（用 t-1 以前所有資料）
    "rolling"   滾動窗 β（用 t-1 以前 window 天）← 預設

    rolling / expanding 皆對 β 做 shift(1)，確保 t 日的避險比例僅用 t-1 以前資訊
    （無未來函數，符合 walk-forward 精神）。warm-up 期以第一個有效 β 回填，
    仍缺則設 1.0（退化為原本的算術差）。
    """
    pair = pd.concat([strat.rename("s"), mkt.rename("m")], axis=1).dropna()

    if mode == "full":
        var_m = pair["m"].var()
        b = pair["s"].cov(pair["m"]) / var_m if var_m > 0 else 1.0
        return pd.Series(b, index=strat.index)

    if mode == "expanding":
        cov = pair["s"].expanding(min_periods=min_periods).cov(pair["m"])
        var = pair["m"].expanding(min_periods=min_periods).var()
    else:  # rolling
        cov = pair["s"].rolling(window, min_periods=min_periods).cov(pair["m"])
        var = pair["m"].rolling(window, min_periods=min_periods).var()

    beta = (cov / var).shift(1)               # 無未來函數
    beta = beta.reindex(strat.index).bfill().fillna(1.0)
    return beta


def daily_returns(
    df: pd.DataFrame,
    market_series: pd.Series,
    cost_bps: float = 0,
    capital_usage: float = 1.0,
) -> pd.DataFrame:
    """
    計算每日各 Decile 的等權報酬，乘以資金使用率後與大盤對齊。

    Parameters
    ----------
    capital_usage : float
        資金使用率，e.g. 0.5 代表 50% 資金投入、50% 現金閒置。
        策略報酬 = raw_decile_return × capital_usage
        market_return_fwd 保持原始（100% 持有大盤作為 benchmark）

    新增欄位
    --------
    market      : 大盤同窗報酬（market_return_fwd, open[T+1]→open[T+2], 原始，不乘 capital_usage）
    excess_D1   : D1（scaled） - market
    """
    cost = cost_bps / 10_000 * 2

    ret = (
        df.groupby([DATE_COL, "decile"])["return"]
        .mean()
        .unstack("decile")
        .sort_index()
    )
    ret.columns = [f"D{int(c)}" for c in ret.columns]

    # 扣交易成本 → 乘資金使用率
    ret = (ret - cost) * capital_usage

    # 對齊大盤
    ret["market"] = market_series.reindex(ret.index)

    n_missing = ret["market"].isna().sum()
    if n_missing > 0:
        print(f"  [WARN] market_return_fwd 缺少 {n_missing} 天（已設為 NaN，不影響 Decile 計算）")

    # excess return（D1 scaled - market）
    if "D1" in ret.columns:
        ret["excess_D1"] = ret["D1"] - ret["market"]

    return ret


def calc_metrics(
    ret_series: pd.Series,
    market_series: pd.Series | None = None,
    annual_days: int = 252,
) -> dict:
    """
    計算單一報酬序列的績效指標。
    若提供 market_series，額外計算 alpha 與 IR（Information Ratio）。

    alpha : 年化超額報酬（策略年化 - 大盤年化）
    IR    : alpha / tracking_error_annualised
    """
    r = ret_series.dropna()
    ann_ret = (1 + r).prod() ** (annual_days / len(r)) - 1
    ann_std = r.std() * np.sqrt(annual_days)
    sharpe  = ann_ret / ann_std if ann_std > 0 else np.nan
    cum     = (1 + r).cumprod()
    rolling_max = cum.cummax()
    drawdown    = (cum - rolling_max) / rolling_max
    mdd    = drawdown.min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan
    hit_rate = (r > 0).mean()

    result = {
        "ann_return": round(ann_ret,  4),
        "ann_std":    round(ann_std,  4),
        "sharpe":     round(sharpe,   4),
        "mdd":        round(mdd,      4),
        "calmar":     round(calmar,   4),
        "hit_rate":   round(hit_rate, 4),
        "n_days":     len(r),
        "alpha":      np.nan,
        "IR":         np.nan,
    }

    if market_series is not None:
        common = r.index.intersection(market_series.dropna().index)
        if len(common) > 10:
            mkt_r    = market_series.loc[common]
            strat_r  = r.loc[common]
            mkt_ann  = (1 + mkt_r).prod() ** (annual_days / len(mkt_r)) - 1
            alpha    = ann_ret - mkt_ann
            excess   = strat_r - mkt_r
            te       = excess.std() * np.sqrt(annual_days)
            ir       = (excess.mean() * annual_days) / (excess.std() * np.sqrt(annual_days)) if te > 0 else np.nan
            result["alpha"] = round(alpha, 4)
            result["IR"]    = round(ir,    4)

    return result


# ============================================================
#  圖表
# ============================================================

COLORS = plt.cm.RdYlGn(np.linspace(0.15, 0.85, N_DECILES))


def plot_cumulative(ret_df: pd.DataFrame, output_dir: Path):
    """10 組累積報酬 + 大盤基準線 + excess_D1（同一左軸 log10 NAV）。"""
    decile_cols = [c for c in ret_df.columns if c.startswith("D") and c not in ("excess_D1",)]
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, col in enumerate(decile_cols):
        cum     = (1 + ret_df[col].fillna(0)).cumprod()
        log_cum = np.log10(cum)
        lw      = 2.5 if col == "D1" else 1.0
        ax.plot(log_cum.index, log_cum.values, label=col, color=COLORS[i], linewidth=lw)

    if "market" in ret_df.columns:
        mkt_cum = (1 + ret_df["market"].fillna(0)).cumprod()
        ax.plot(mkt_cum.index, np.log10(mkt_cum.values),
                label="Market", color="black", linewidth=1.8, linestyle="--")

    if "excess_D1" in ret_df.columns:
        exc_cum = (1 + ret_df["excess_D1"].fillna(0)).cumprod()
        ax.plot(exc_cum.index, np.log10(exc_cum.values),
                label="Excess D1", color="#7b2d8b", linewidth=1.8, linestyle="-.")

    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_ylabel("log10(NAV)", fontsize=9)
    ax.set_xlabel("Date")
    cap_label = f"  (Capital {CAPITAL_USAGE:.0%})" if CAPITAL_USAGE < 1 else ""
    ax.set_title(f"Decile Cumulative Returns (log10 NAV){cap_label}", fontsize=13)
    ax.legend(ncol=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "cumulative_returns.png", dpi=150)
    plt.close(fig)
    print("  ✓ cumulative_returns.png")


def plot_excess_return(ret_df: pd.DataFrame, output_dir: Path):
    """D1 累積 excess return 曲線（D1_scaled - market）。"""
    if "excess_D1" not in ret_df.columns:
        return
    exc = ret_df["excess_D1"].fillna(0)
    cum_exc = (1 + exc).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cum_exc.index, cum_exc.values, color="#2c7bb6", linewidth=2)
    ax.fill_between(cum_exc.index, cum_exc.values, 1.0,
                    where=(cum_exc.values >= 1.0), color="#2c7bb6", alpha=0.15)
    ax.fill_between(cum_exc.index, cum_exc.values, 1.0,
                    where=(cum_exc.values < 1.0),  color="#d7191c", alpha=0.15)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))
    cap_label = f"  (Capital {CAPITAL_USAGE:.0%})" if CAPITAL_USAGE < 1 else ""
    ax.set_title(f"D1 Cumulative Excess Return vs Market{cap_label}", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Excess NAV")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "excess_return.png", dpi=150)
    plt.close(fig)
    print("  ✓ excess_return.png")


def plot_decile_summary(metrics_df: pd.DataFrame, output_dir: Path):
    decile_rows = [i for i in metrics_df.index if i.startswith("D") and i not in ("excess_D1",)]
    sub_df = metrics_df.loc[decile_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(decile_rows))

    # 年化報酬（ln 形式：ln(1+R)，連續複利年化，含大盤基準橫線）
    #   破表的 D1 年化（~499%）在算術尺度會壓扁其他組；改 ln(1+R) 後尺度與
    #   cumulative_returns.png 的 log NAV 一致，組間差異更易判讀。
    ax = axes[0]
    ln_ret = np.log1p(sub_df["ann_return"].clip(lower=-0.999))   # ln(1 + ann_return)
    bars = ax.bar(x, ln_ret, color=COLORS, edgecolor="none", alpha=0.85)
    if "market" in metrics_df.index:
        ax.axhline(np.log1p(max(metrics_df.loc["market", "ann_return"], -0.999)),
                   color="black", linewidth=1.2, linestyle="--", label="Market")
        ax.legend(fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(decile_rows, fontsize=9)
    ax.set_ylabel("ln(1 + annualised return)", fontsize=9)
    ax.set_title("Annualised Return by Decile (ln)", fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, ln_val, raw in zip(bars, ln_ret, sub_df["ann_return"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * np.sign(ln_val + 1e-9),
                f"{ln_val:.2f}\n({raw:.0%})", ha="center", va="bottom", fontsize=7)

    # Sharpe
    ax = axes[1]
    bars = ax.bar(x, sub_df["sharpe"], color=COLORS, edgecolor="none", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(decile_rows, fontsize=9)
    ax.set_title("Sharpe Ratio by Decile", fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, sub_df["sharpe"]):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02 * np.sign(val + 1e-9),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "decile_summary_bar.png", dpi=150)
    plt.close(fig)
    print("  ✓ decile_summary_bar.png")


def plot_annual_decile1(ret_df: pd.DataFrame, output_dir: Path):
    """Decile 1 各年度報酬長條圖，附大盤同期報酬對照點。"""
    d1 = ret_df["D1"].dropna()
    annual_d1 = d1.groupby(d1.index.year).apply(lambda r: (1 + r).prod() - 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d7191c" if v < 0 else "#2c7bb6" for v in annual_d1.values]
    bars = ax.bar(annual_d1.index.astype(str), annual_d1.values,
                  color=colors, edgecolor="none", alpha=0.85)

    # 大盤年度對照
    if "market" in ret_df.columns:
        mkt = ret_df["market"].dropna()
        annual_mkt = mkt.groupby(mkt.index.year).apply(lambda r: (1 + r).prod() - 1)
        common_years = annual_d1.index.intersection(annual_mkt.index)
        ax.plot(common_years.astype(str), annual_mkt.loc[common_years].values,
                "o--", color="black", linewidth=1.2, markersize=6, label="Market", zorder=5)
        ax.legend(fontsize=9)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    cap_label = f" (Capital {CAPITAL_USAGE:.0%})" if CAPITAL_USAGE < 1 else ""
    ax.set_title(f"Decile 1 – Annual Return{cap_label}", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, annual_d1.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * np.sign(val + 1e-9),
                f"{val:.1%}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "annual_decile1.png", dpi=150)
    plt.close(fig)
    print("  ✓ annual_decile1.png")


def plot_drawdown(ret_df: pd.DataFrame, output_dir: Path):
    """Decile 1 水下曲線。"""
    d1  = ret_df["D1"].fillna(0)
    cum = (1 + d1).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, color="#d7191c", alpha=0.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_title("Decile 1 – Drawdown", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "drawdown_decile1.png", dpi=150)
    plt.close(fig)
    print("  ✓ drawdown_decile1.png")


def relative_drawdown(strat: pd.Series, market: pd.Series):
    """策略相對大盤的 NAV 比值與回撤序列。

    rel  = (1+strat).cumprod() / (1+market).cumprod()  ← 相對財富曲線
    dd   = rel / rel.cummax() - 1                       ← 距「歷史最佳領先」的跌幅

    dd 觸底 = 策略相對大盤輸最多的時點；dd 回到 0 = 重新追回先前的領先高點。
    僅取 strat 與 market 同時有值的交易日。
    """
    common = strat.dropna().index.intersection(market.dropna().index)
    s = strat.loc[common].sort_index()
    m = market.loc[common].sort_index()
    rel = (1 + s).cumprod() / (1 + m).cumprod()
    dd  = rel / rel.cummax() - 1.0
    return rel, dd


def drawdown_episodes(dd: pd.Series) -> pd.DataFrame:
    """將回撤序列切成一段段「水下期間」。

    欄位：peak（領先高點日）/ trough（輸最多日）/ recovery（追平日，未追平為 NaT）
          depth（最大相對回撤）/ days_to_trough / days_underwater / ongoing
    依 depth 由深到淺排序。
    """
    rows = []
    in_ep = False
    peak_date = dd.index[0]
    trough_date, trough_val = peak_date, 0.0

    for date, d in dd.items():
        if d < -1e-12:                       # 水下中
            if not in_ep:
                in_ep, trough_val, trough_date = True, d, date
            elif d < trough_val:
                trough_val, trough_date = d, date
        else:                                # 回到（新）高點
            if in_ep:
                rows.append(dict(peak=peak_date, trough=trough_date,
                                 recovery=date, depth=trough_val, ongoing=False))
                in_ep = False
            peak_date = date

    if in_ep:                                # 序列結束時仍在水下
        rows.append(dict(peak=peak_date, trough=trough_date,
                         recovery=pd.NaT, depth=trough_val, ongoing=True))

    cols = ["peak", "trough", "recovery", "depth",
            "days_to_trough", "days_underwater", "ongoing"]
    if not rows:
        return pd.DataFrame(columns=cols)

    tab = pd.DataFrame(rows)
    for c in ("peak", "trough", "recovery"):
        tab[c] = pd.to_datetime(tab[c])
    last_date = dd.index[-1]
    rec = tab["recovery"].fillna(last_date)
    tab["days_to_trough"]  = (tab["trough"] - tab["peak"]).dt.days
    tab["days_underwater"] = (rec - tab["peak"]).dt.days
    return tab[cols].sort_values("depth").reset_index(drop=True)


def plot_excess_drawdown(ret_df: pd.DataFrame, output_dir: Path):
    """D1 相對大盤回撤（relative drawdown vs market）水下曲線 + 水下期間明細。

    回答兩個問題：
      多少 → 相對大盤最大回撤深度（最深那段，max relative drawdown）
      多久 → 水下期間天數（peak→recovery，最久那段；常與最深段不同）

    回傳 episodes 表供主流程列印；同時輸出 excess_drawdown_episodes.csv。
    """
    if "D1" not in ret_df.columns or "market" not in ret_df.columns:
        return None

    rel, dd = relative_drawdown(ret_df["D1"], ret_df["market"])
    if dd.empty:
        return None

    episodes = drawdown_episodes(dd)
    episodes.to_csv(output_dir / "excess_drawdown_episodes.csv",
                    index=False, encoding="utf-8-sig")

    deepest = episodes.iloc[0]                              # depth 最深
    longest = episodes.loc[episodes["days_underwater"].idxmax()]  # 水下最久

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, color="#7b2d8b", alpha=0.45)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_title("D1 Relative Drawdown vs Market（相對大盤回撤）", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # 最久那段：底色陰影
    end = dd.index[-1] if bool(longest["ongoing"]) else longest["recovery"]
    ax.axvspan(longest["peak"], end, color="#7b2d8b", alpha=0.10)

    # 最深那段：箭頭標註（深度 + 水下天數）
    ax.annotate(
        f"最深 {deepest['depth']:.1%} / {int(deepest['days_underwater'])}d",
        xy=(deepest["trough"], deepest["depth"]),
        xytext=(deepest["trough"], deepest["depth"] * 0.55),
        fontsize=8, color="#7b2d8b", ha="center",
        arrowprops=dict(arrowstyle="->", color="#7b2d8b"),
    )
    if longest["peak"] != deepest["peak"]:
        ax.annotate(
            f"最久 {int(longest['days_underwater'])}d",
            xy=(longest["trough"], longest["depth"]),
            fontsize=8, color="#555", ha="center", va="top",
        )

    plt.tight_layout()
    fig.savefig(output_dir / "excess_drawdown_decile1.png", dpi=150)
    plt.close(fig)
    print("  ✓ excess_drawdown_decile1.png")
    print("  ✓ excess_drawdown_episodes.csv")
    return episodes


def load_liquidity_filter_meta(exp_dir: Path) -> dict | None:
    """
    讀取 main_fix.py 寫入的 liquidity_filter_meta.json（若存在）。
    用途：純為回測報表標示「這批 predictions.csv 是用哪組流動性篩選
    參數訓練出來的」，避免事後看報表時誤判為全市場訓練結果。
    找不到時回傳 None（視為未套用篩選 / 舊版 pipeline 產出）。
    """
    path = exp_dir / "liquidity_filter_meta.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ⚠ 讀取 {path} 失敗（{e}），視為無篩選資訊")
        return None


# ============================================================
#  主流程
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_cjk_font()   # 設定中文字型，避免圖中中文變方框

    print(f"\n{'='*60}")
    print(f"  Backtest")
    print(f"  Experiment dir  : {EXPERIMENT_DIR}")
    print(f"  Deciles         : {N_DECILES}")
    print(f"  Cost (bps)      : {COST_BPS}")
    print(f"  Capital usage   : {CAPITAL_USAGE:.0%}")
    print(f"{'='*60}\n")

    # ── ★ 流動性篩選溯源（若 main_fix.py 有寫 meta 則顯示）───
    liq_meta = load_liquidity_filter_meta(EXPERIMENT_DIR)
    if liq_meta is not None:
        print(f"  ★ 本批訓練資料已套用流動性篩選："
              f"{liq_meta.get('formula')}  tag={liq_meta.get('tag')}\n"
              f"    w1={liq_meta.get('w1_amount')}  "
              f"w2={liq_meta.get('w2_market_value')}  "
              f"w3={liq_meta.get('w3_turnover')}  "
              f"keep_ratio={liq_meta.get('keep_ratio')}\n"
              f"    rows: {liq_meta.get('rows_before'):,} → {liq_meta.get('rows_after'):,}\n")
    else:
        print("  ⚠ 找不到 liquidity_filter_meta.json，"
              "本批 predictions 可能來自未套用流動性篩選的全市場訓練\n")

    # ── 1. 載入資料 ──────────────────────────────────────────
    print("► 載入 predictions...")
    df = load_all_predictions(EXPERIMENT_DIR)

    # ── 2. 載入大盤報酬 ──────────────────────────────────────
    print("\n► 載入 market_return...")
    market_series = load_market_return(DATABASE_MAKE_DIR)

    # ── 3. 分組 ──────────────────────────────────────────────
    print("\n► 分 Decile 組...")
    df = assign_deciles(df, n=N_DECILES)
    print(f"  Decile 分布：\n{df['decile'].value_counts().sort_index().to_string()}")

    # ── 4. 每日報酬（含資金使用率與大盤對齊） ────────────────
    print("\n► 計算每日組別報酬...")
    ret_df = daily_returns(
        df,
        market_series=market_series,
        cost_bps=COST_BPS,
        capital_usage=CAPITAL_USAGE,
    )
    ret_df.to_csv(OUTPUT_DIR / "daily_returns.csv", encoding="utf-8-sig")
    print(f"  ✓ daily_returns.csv  shape={ret_df.shape}  欄位：{list(ret_df.columns)}")

    # ── 5. 績效指標 ──────────────────────────────────────────
    print("\n► 計算績效指標...")
    mkt_series_aligned = ret_df["market"] if "market" in ret_df.columns else None

    metrics_list = []
    for col in ret_df.columns:
        if col in ("market", "excess_D1"):
            pass_mkt = None   # 大盤自己不算 alpha
        else:
            pass_mkt = mkt_series_aligned
        m = calc_metrics(ret_df[col], market_series=pass_mkt, annual_days=ANNUAL_DAYS)
        m["decile"] = col
        metrics_list.append(m)

    metrics_df = pd.DataFrame(metrics_list).set_index("decile")
    metrics_df.to_csv(OUTPUT_DIR / "decile_metrics.csv", encoding="utf-8-sig")

    # ★ 把本次篩選溯源資訊也存一份到輸出目錄，跟 decile_metrics.csv 放一起，
    #   避免只看 output/backtest/ 而不知道用了什麼篩選參數
    if liq_meta is not None:
        with open(OUTPUT_DIR / "liquidity_filter_meta.json", "w", encoding="utf-8") as f:
            json.dump(liq_meta, f, ensure_ascii=False, indent=2)

    display_cols = ["ann_return", "ann_std", "sharpe", "mdd", "calmar", "hit_rate", "alpha", "IR"]
    print(metrics_df[display_cols].to_string())

    # ── 6. 圖表 ──────────────────────────────────────────────
    print("\n► 產出圖表...")
    plot_cumulative(ret_df, OUTPUT_DIR)
    plot_excess_return(ret_df, OUTPUT_DIR)
    plot_decile_summary(metrics_df, OUTPUT_DIR)
    plot_annual_decile1(ret_df, OUTPUT_DIR)
    plot_drawdown(ret_df, OUTPUT_DIR)
    episodes = plot_excess_drawdown(ret_df, OUTPUT_DIR)

    if episodes is not None and not episodes.empty:
        deepest = episodes.iloc[0]
        longest = episodes.loc[episodes["days_underwater"].idxmax()]
        print("\n► 相對大盤回撤（D1 vs market）")
        print(f"    輸最多（最深）：{deepest['depth']:.2%}"
              f"  谷底 {deepest['trough'].date()}"
              f"  水下 {int(deepest['days_underwater'])} 天"
              f"{'（尚未追平）' if deepest['ongoing'] else ''}")
        print(f"    輸最久（最長水下）：{int(longest['days_underwater'])} 天"
              f"  {longest['peak'].date()} → "
              f"{'至今' if longest['ongoing'] else longest['recovery'].date()}"
              f"  深度 {longest['depth']:.2%}")
        print("\n  前 5 大水下期間：")
        show = episodes.head(5).copy()
        for c in ("peak", "trough", "recovery"):
            show[c] = show[c].dt.date.astype(str)
        show["depth"] = (show["depth"] * 100).round(2).astype(str) + "%"
        print(show[["peak", "trough", "recovery", "depth",
                    "days_underwater", "ongoing"]].to_string(index=False))

    print(f"\n{'='*60}")
    print(f"  完成  →  {OUTPUT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()