"""
backtest_with_sanity.py
=======================

【功能概述】
整合回測分析（原 5_result.py）與健全性檢查（原 6_check.py）。

【交易假設】
  訊號日 t  →  t+1 開盤買入  →  t+2 開盤賣出
  return = open_{t+2} / open_{t+1} - 1

  ⚠️  Limit-Up 執行風險說明
  ─────────────────────────────────────────────
  「在 t+1 開盤時股票已掛漲停」才是真正買不到的情境。
  判斷依據：open_{t+1} / close_t >= 1.095（台股 ±10% 限制）

  本腳本提供兩層檢查：
    A. 入場側（entry_limit_up）：若資料含 t 日收盤價欄位，
       可計算 open_{t+1} / close_t 是否已漲停，此情境買不到。
       → 需要欄位：prev_close 或 close（前一交易日收盤）

    B. 出場側（exit_limit_up_approx）：用持有期報酬 > 9.5% 作為代理，
       但這反而是「賺到」的情境（開盤賣出時對手方很多），
       台股機制中漲停開盤賣方不受影響，不算執行風險。

  若資料中無法計算 entry_limit_up，改用更嚴格代理：
    return < -0.095  →  跌停代理（做空 D1 時才有影響）

【分組定義】
  D01 ~ D10 : 每日依 y_prob 由低到高分成十等分（等權平均報酬）
  D11 (L/S) : Long D10 / Short D1，資本中性

【輸出】
  experiment_merge_prediction.csv
  daily_pnl.csv
  backtest_metrics.csv
  ic_series.csv
  sanity_report.txt           ← 新增：健全性檢查報告
  chart_cumret.png
  chart_drawdown.png
  chart_risk_adjusted.png
  chart_return_dd_winrate.png
  chart_ic.png
  chart_heatmap.png
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

ANNUAL_FACTOR = 252
RF_DAILY      = 0.0

# 台股漲跌停門檻（一般股票 ±10%，2020 後部分改 ±15%，此處保守用 9.5%）
LIMIT_THRESHOLD = 0.095


# ══════════════════════════════════════════════════════════════════
# Step 1  合併所有 predictions.csv
# ══════════════════════════════════════════════════════════════════

def merge_predictions(experiment_dir: Path, output_dir: Path) -> pd.DataFrame:
    pred_files = sorted(experiment_dir.glob('*/predictions.csv'))
    if not pred_files:
        raise FileNotFoundError(f"找不到任何 predictions.csv in {experiment_dir}")

    print(f"📂 找到 {len(pred_files)} 份 predictions.csv")

    dfs = []
    for f in pred_files:
        df = pd.read_csv(f, encoding='utf-8-sig')
        df['source'] = f.parent.name
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged['年月日'] = merged['年月日'].astype(str).str.zfill(8)

    before = len(merged)
    merged = merged.drop_duplicates(subset=['證券代碼', '年月日'])
    print(f"   去除重複: {before - len(merged)} 筆，剩餘 {len(merged)} 筆")

    out_path = output_dir / 'experiment_merge_prediction.csv'
    merged.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"✅ 合併完成 → {out_path}\n")
    return merged


# ══════════════════════════════════════════════════════════════════
# Step 2  每日依 y_prob 分十組，計算每日 PnL
# ══════════════════════════════════════════════════════════════════

def build_daily_pnl(merged: pd.DataFrame):
    records     = []
    stock_lists = {}

    for date, group in merged.groupby('年月日'):
        if len(group) < 10:
            continue

        group = group.copy()
        group['decile'] = pd.qcut(
            group['y_prob'],
            q=10,
            labels=range(1, 11),
            duplicates='drop'
        )

        for decile, dec_group in group.groupby('decile', observed=False):
            d = int(decile)
            records.append({
                'date':     date,
                'decile':   d,
                'n_stocks': len(dec_group),
                'pnl':      dec_group['return'].mean()
            })
            stock_lists[(pd.Timestamp(date).strftime('%Y-%m-%d'), d)] = set(dec_group['證券代碼'].tolist())

    daily_pnl = pd.DataFrame(records)
    daily_pnl['date'] = pd.to_datetime(daily_pnl['date'], infer_datetime_format=True)
    daily_pnl = daily_pnl.sort_values(['date', 'decile']).reset_index(drop=True)
    return daily_pnl, stock_lists


# ══════════════════════════════════════════════════════════════════
# Step 3  計算 checklist 指標
# ══════════════════════════════════════════════════════════════════

def calc_drawdown_series(cum_ret: pd.Series):
    rolling_max = cum_ret.cummax()
    drawdown    = (cum_ret - rolling_max) / rolling_max.abs().replace(0, np.nan)
    return drawdown


def drawdown_durations(drawdown: pd.Series):
    in_dd     = drawdown < 0
    durations = []
    count     = 0
    for v in in_dd:
        if v:
            count += 1
        else:
            if count > 0:
                durations.append(count)
            count = 0
    if count > 0:
        durations.append(count)
    return durations


def ic_series(merged: pd.DataFrame) -> pd.Series:
    ics = {}
    for date, g in merged.groupby('年月日'):
        if len(g) < 5:
            continue
        r, _ = stats.spearmanr(g['y_prob'], g['return'])
        if not np.isnan(r):
            ics[date] = r
    return pd.Series(ics)


def factor_turnover_series(merged: pd.DataFrame) -> float:
    merged = merged.copy()
    merged['date'] = pd.to_datetime(merged['年月日'], infer_datetime_format=True)
    merged = merged.sort_values(['證券代碼', 'date'])
    merged['rank']        = merged.groupby('date')['y_prob'].rank(pct=True)
    merged['rank_prev']   = merged.groupby('證券代碼')['rank'].shift(1)
    merged['rank_change'] = (merged['rank'] - merged['rank_prev']).abs()
    return float(merged['rank_change'].mean())


def calc_decile_turnover(stock_lists: dict, decile: int, sorted_dates: list) -> tuple:
    turnovers   = []
    prev_stocks = None

    for date in sorted_dates:
        key = (pd.Timestamp(date).strftime('%Y-%m-%d'), decile)
        if key not in stock_lists:
            prev_stocks = None
            continue
        curr_stocks = stock_lists[key]

        if prev_stocks is not None and len(prev_stocks) > 0:
            removed = prev_stocks - curr_stocks
            to      = len(removed) / len(prev_stocks)
            turnovers.append(to)

        prev_stocks = curr_stocks

    avg_to = float(np.mean(turnovers)) if turnovers else np.nan
    return turnovers, avg_to


def calc_transaction_cost(turnover_series: list,
                           n_stocks_avg: float,
                           commission: float = 0.001425,
                           tax: float = 0.003) -> float:
    if not turnover_series:
        return np.nan
    avg_to      = float(np.mean(turnover_series))
    cost_per_to = commission + (commission + tax)
    daily_cost  = avg_to * cost_per_to
    annual_cost = daily_cost * ANNUAL_FACTOR
    return round(annual_cost, 6)


def calc_metrics(pnl_series: pd.Series, benchmark_series: pd.Series = None) -> dict:
    r = pnl_series.dropna()
    n = len(r)

    if n == 0:
        return {}

    cum_ret   = (1 + r).cumprod()
    total_ret = float(cum_ret.iloc[-1] - 1)
    T_years   = n / ANNUAL_FACTOR
    cagr      = float((1 + total_ret) ** (1 / T_years) - 1) if T_years > 0 else np.nan
    avg_ret   = float(r.mean())

    if benchmark_series is not None:
        bm          = benchmark_series.reindex(r.index).fillna(0)
        alpha_daily = r - bm
    else:
        alpha_daily = r.copy()
        bm          = pd.Series(0, index=r.index)

    alpha_ann    = float(alpha_daily.mean() * ANNUAL_FACTOR)
    volatility   = float(r.std() * np.sqrt(ANNUAL_FACTOR))
    downside     = r[r < 0]
    downside_dev = float(downside.std() * np.sqrt(ANNUAL_FACTOR)) if len(downside) > 1 else np.nan

    dd_series = calc_drawdown_series(cum_ret)
    max_dd    = float(dd_series.min())
    avg_dd    = float(dd_series[dd_series < 0].mean()) if (dd_series < 0).any() else 0.0

    excess_r = r - RF_DAILY
    sharpe   = float(excess_r.mean() / r.std() * np.sqrt(ANNUAL_FACTOR)) if r.std() > 0 else np.nan
    sortino  = float(excess_r.mean() / (downside.std() + 1e-12) * np.sqrt(ANNUAL_FACTOR)) if len(downside) > 1 else np.nan
    calmar   = float(cagr / abs(max_dd)) if max_dd != 0 else np.nan

    tracking_error = float(alpha_daily.std() * np.sqrt(ANNUAL_FACTOR))
    ir = float(alpha_daily.mean() * ANNUAL_FACTOR / tracking_error) if tracking_error > 0 else np.nan

    win_rate_daily = float((r > 0).mean())
    profits        = r[r > 0].sum()
    losses         = abs(r[r < 0].sum())
    profit_factor  = float(profits / losses) if losses > 0 else np.nan
    skewness       = float(stats.skew(r))
    kurt           = float(stats.kurtosis(r))

    durations  = drawdown_durations(dd_series)
    max_dd_dur = int(max(durations)) if durations else 0
    avg_dd_dur = float(np.mean(durations)) if durations else 0.0

    if n > 2:
        t_stat, p_val = stats.ttest_1samp(alpha_daily, 0)
        t_stat = float(t_stat)
        p_val  = float(p_val)
    else:
        t_stat = p_val = np.nan

    return {
        'total_return':       round(total_ret, 6),
        'cagr':               round(cagr, 6),
        'avg_daily_return':   round(avg_ret, 6),
        'alpha_annualized':   round(alpha_ann, 6),
        'volatility':         round(volatility, 6),
        'max_drawdown':       round(max_dd, 6),
        'avg_drawdown':       round(avg_dd, 6),
        'downside_deviation': round(downside_dev, 6) if not np.isnan(downside_dev) else np.nan,
        'sharpe_ratio':       round(sharpe, 6) if not np.isnan(sharpe) else np.nan,
        'sortino_ratio':      round(sortino, 6) if not np.isnan(sortino) else np.nan,
        'calmar_ratio':       round(calmar, 6) if not np.isnan(calmar) else np.nan,
        'information_ratio':  round(ir, 6) if not np.isnan(ir) else np.nan,
        'win_rate_daily':     round(win_rate_daily, 6),
        'win_rate_stock':     np.nan,
        'profit_factor':      round(profit_factor, 6) if not np.isnan(profit_factor) else np.nan,
        'skewness':           round(skewness, 6),
        'kurtosis':           round(kurt, 6),
        'max_dd_duration':    max_dd_dur,
        'avg_dd_duration':    round(avg_dd_dur, 2),
        't_stat_alpha':       round(t_stat, 4) if not np.isnan(t_stat) else np.nan,
        'p_value_alpha':      round(p_val, 4) if not np.isnan(p_val) else np.nan,
        'n_days':             n,
    }


# ══════════════════════════════════════════════════════════════════
# Step 4  圖表輸出
# ══════════════════════════════════════════════════════════════════

def plot_all(daily_pnl: pd.DataFrame, metrics_df: pd.DataFrame,
             ic_ser: pd.Series, output_dir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.cm as mplcm

    DECILE_LABELS = {i: f'D{i:02d}' for i in range(1, 11)}
    DECILE_LABELS[11] = 'L/S'

    cmap   = plt.cm.RdYlGn
    colors = {i: cmap((i - 1) / 9) for i in range(1, 11)}
    colors[11] = '#FFD700'

    # ── 1. 累積報酬曲線 ──
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    for decile in list(range(1, 11)) + [11]:
        dec_df = daily_pnl[daily_pnl['decile'] == decile].set_index('date')['pnl'].sort_index()
        cum    = (1 + dec_df).cumprod()
        lw     = 2.5 if decile == 11 else 1.2
        alpha  = 1.0 if decile in [1, 10, 11] else 0.55
        ax.plot(cum.index, cum.values, color=colors[decile],
                linewidth=lw, alpha=alpha, label=DECILE_LABELS[decile])
    ax.set_title('Cumulative Return by Decile', color='white', fontsize=14, pad=12)
    ax.tick_params(colors='#aaaaaa')
    ax.spines[:].set_color('#333333')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.1f}x'))
    ax.legend(ncol=4, fontsize=8, framealpha=0.15, labelcolor='white', facecolor='#1a1a2e')
    ax.grid(axis='y', color='#333333', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / 'chart_cumret.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("   ✅ chart_cumret.png")

    # ── 2. Drawdown ──
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    for decile, label, color in [(1, 'D01', colors[1]), (10, 'D10', colors[10]), (11, 'L/S', colors[11])]:
        dec_df = daily_pnl[daily_pnl['decile'] == decile].set_index('date')['pnl'].sort_index()
        cum    = (1 + dec_df).cumprod()
        dd     = calc_drawdown_series(cum)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color=color, label=label)
        ax.plot(dd.index, dd.values, color=color, linewidth=1.0)
    ax.set_title('Drawdown', color='white', fontsize=14, pad=12)
    ax.tick_params(colors='#aaaaaa')
    ax.spines[:].set_color('#333333')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=9, framealpha=0.15, labelcolor='white', facecolor='#1a1a2e')
    ax.grid(axis='y', color='#333333', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / 'chart_drawdown.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("   ✅ chart_drawdown.png")

    # ── 3. Risk Adjusted 長條圖 ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0d1117')
    deciles  = metrics_df[metrics_df['decile'] <= 11]['decile'].tolist()
    xlabels  = [DECILE_LABELS[d] for d in deciles]
    bar_colors = [colors.get(d, '#FFD700') for d in deciles]
    for ax, col, title in zip(axes,
                               ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio'],
                               ['Sharpe', 'Sortino', 'Calmar']):
        ax.set_facecolor('#0d1117')
        vals = metrics_df[metrics_df['decile'] <= 11][col].fillna(0).tolist()
        ax.bar(xlabels, vals, color=bar_colors)
        ax.set_title(title, color='white', fontsize=12)
        ax.tick_params(colors='#aaaaaa', rotation=45)
        ax.spines[:].set_color('#333333')
        ax.axhline(0, color='#555555', linewidth=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / 'chart_risk_adjusted.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("   ✅ chart_risk_adjusted.png")

    # ── 4. Total Return / MaxDD / WinRate ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0d1117')
    for ax, col, title, fmt in zip(
        axes,
        ['total_return', 'max_drawdown', 'win_rate_daily'],
        ['Total Return', 'Max Drawdown', 'Win Rate (daily)'],
        ['{:.1%}', '{:.1%}', '{:.1%}']
    ):
        ax.set_facecolor('#0d1117')
        vals = metrics_df[metrics_df['decile'] <= 11][col].fillna(0).tolist()
        ax.bar(xlabels, vals, color=bar_colors)
        ax.set_title(title, color='white', fontsize=12)
        ax.tick_params(colors='#aaaaaa', rotation=45)
        ax.spines[:].set_color('#333333')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.axhline(0, color='#555555', linewidth=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / 'chart_return_dd_winrate.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("   ✅ chart_return_dd_winrate.png")

    # ── 5. IC 序列 ──
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ic_ser_plot = ic_ser.copy()
    ic_ser_plot.index = pd.to_datetime(ic_ser_plot.index, infer_datetime_format=True)
    ax.bar(ic_ser_plot.index, ic_ser_plot.values, color='#4488cc', alpha=0.5, width=1)
    roll = ic_ser_plot.rolling(20).mean()
    ax.plot(roll.index, roll.values, color='#FFD700', linewidth=1.5, label='20d MA')
    ax.axhline(0, color='#888888', linewidth=0.8)
    ax.set_title('IC Series (Spearman)', color='white', fontsize=14, pad=12)
    ax.tick_params(colors='#aaaaaa')
    ax.spines[:].set_color('#333333')
    ax.legend(fontsize=9, framealpha=0.15, labelcolor='white', facecolor='#1a1a2e')
    plt.tight_layout()
    fig.savefig(output_dir / 'chart_ic.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("   ✅ chart_ic.png")

    # ── 6. Heatmap ──
    heat_cols = ['sharpe_ratio', 'total_return', 'max_drawdown',
                 'win_rate_daily', 'win_rate_stock', 'calmar_ratio',
                 'alpha_annualized', 'avg_turnover', 'annual_tx_cost',
                 'ic_mean']
    heat_df   = metrics_df.set_index('decile')[heat_cols]
    heat_df.index = [DECILE_LABELS[d] for d in heat_df.index]
    n_rows, n_cols = heat_df.shape

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    norm = plt.Normalize(vmin=0, vmax=1)

    for j, col in enumerate(heat_cols):
        col_vals = heat_df[col].values.astype(float)
        finite   = col_vals[np.isfinite(col_vals)]
        if len(finite) == 0:
            continue
        vmin, vmax = finite.min(), finite.max()
        rng = vmax - vmin if vmax != vmin else 1.0
        if col in ['max_drawdown', 'annual_tx_cost', 'avg_turnover']:
            normed = 1 - (col_vals - vmin) / rng
        else:
            normed = (col_vals - vmin) / rng

        for i, val in enumerate(col_vals):
            color = mplcm.RdYlGn(normed[i]) if np.isfinite(val) else (0.2, 0.2, 0.2, 1)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
            txt = f'{val:.2f}' if np.isfinite(val) else 'N/A'
            brightness = mplcm.RdYlGn(normed[i])[0] if np.isfinite(val) else 0
            ax.text(j + 0.5, i + 0.5, txt, ha='center', va='center',
                    color='black' if brightness > 0.6 else 'white',
                    fontsize=7.5, fontweight='bold')

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(heat_cols, rotation=30, ha='right', color='#cccccc', fontsize=9)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(heat_df.index, color='#cccccc', fontsize=9)
    ax.tick_params(length=0)
    ax.set_title('Metrics Heatmap by Decile', color='white', fontsize=14, pad=12)
    ax.spines[:].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / 'chart_heatmap.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("   ✅ chart_heatmap.png")


# ══════════════════════════════════════════════════════════════════
# Step 5  健全性檢查（Sanity Check）
# ══════════════════════════════════════════════════════════════════
#
# 【與原 6_check.py 的差異】
#
# ① limit_up 定義修正
#    原版：return > 0.095（持有期報酬超過漲停）
#         → 這是「賺到漲停」，台股機制下賣方仍可成交，不是執行風險
#
#    修正後分三種情境：
#    A. entry_limit_up（真正的執行風險）
#       = open_{t+1} / close_t - 1 >= 0.095
#       = 買入當天開盤已漲停，根本買不到
#       → 需要欄位：prev_close（t 日收盤價，即 signal 日收盤）
#         若有此欄位將自動啟用，否則 fallback 到 B
#
#    B. exit_gain_proxy（持有期大漲代理，非執行風險）
#       = return > 0.095
#       → 這是舊版的定義，改名以避免誤解
#       → 仍有參考意義：D10 集中大量漲停收益是否過度依賴極端正報酬
#
#    C. short_limit_down（做空 D1 的執行風險）
#       = return < -0.095
#       → D1 做空時，跌停開盤買回不到，空頭無法回補
#
# ② FutureWarning 修正
#    所有 groupby 加上 observed=True
#
# ③ IC shuffle test 使用固定 seed，確保可重現

def run_sanity_check(df_raw: pd.DataFrame, output_dir: Path) -> str:
    """
    對 experiment_merge_prediction.csv 執行健全性檢查。
    df_raw 為已讀入的 DataFrame（含 年月日, 證券代碼, return, y_prob）。
    回傳報告字串，並存到 sanity_report.txt。
    """
    lines = []

    def log(s=''):
        lines.append(s)
        print(s)

    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['年月日'])
    df = df.sort_values(['date', '證券代碼'])

    # ── Decile 分組 ──
    result = []
    for d, g in df.groupby('date'):
        if len(g) < 10:
            continue
        g = g.copy()
        g['decile'] = pd.qcut(g['y_prob'], 10,
                               labels=range(1, 11), duplicates='drop')
        result.append(g)
    df = pd.concat(result)

    log("=" * 55)
    log("SANITY CHECK  (交易假設: t+1 開盤買 → t+2 開盤賣)")
    log("=" * 55)

    # ─────────────────────────────────
    # 1  基本資訊
    # ─────────────────────────────────
    log("\n[1] DATA BASIC INFO")
    log(f"    rows   : {len(df):,}")
    log(f"    stocks : {df['證券代碼'].nunique():,}")
    log(f"    dates  : {df['date'].nunique():,}")
    log(f"    return mean : {df['return'].mean():.6f}")
    log(f"    return std  : {df['return'].std():.6f}")

    # ─────────────────────────────────
    # 2  極端報酬
    # ─────────────────────────────────
    log("\n[2] EXTREME RETURN")
    for thr in [0.20, 0.50, 1.00]:
        ratio = (df['return'].abs() > thr).mean()
        log(f"    |return| > {thr:.0%} : {ratio:.4%}")

    # ─────────────────────────────────
    # 3  Limit-Up 偏差（分三類）
    # ─────────────────────────────────
    log("\n[3] LIMIT-UP / LIMIT-DOWN BIAS")
    log("    交易假設：t+1 開盤買，t+2 開盤賣")

    # A. Entry limit-up（真正買不到）
    has_prev_close = 'prev_close' in df.columns and 'open' in df.columns
    if has_prev_close:
        # open = t+1 開盤，prev_close = t 日收盤
        df['entry_limit_up'] = (df['open'] / df['prev_close'] - 1) >= LIMIT_THRESHOLD
        lu_entry = df.groupby('decile', observed=True)['entry_limit_up'].mean()
        log("\n    A. Entry Limit-Up（t+1 開盤已漲停，買不到）：")
        log("       " + lu_entry.to_string().replace('\n', '\n       '))
        lu_flag = 'entry_limit_up'
        lu_label = 'Entry Limit-Up'
    else:
        log("\n    A. Entry Limit-Up：⚠ 無 prev_close/open 欄位，無法直接計算")
        log("       → 請在 predictions.csv 中加入 prev_close（t 日收盤）")
        log("         與 open（t+1 開盤），以啟用此檢查")
        lu_flag = None

    # B. Exit gain proxy（持有期大漲，非執行風險）
    df['exit_gain_proxy'] = df['return'] > LIMIT_THRESHOLD
    lu_exit = df.groupby('decile', observed=True)['exit_gain_proxy'].mean()
    log("\n    B. Exit Gain Proxy（持有期 return > 9.5%）：")
    log("       ⚠ 此為「賺到漲停」，台股開盤賣出仍可成交，不是執行風險")
    log("       但若 D10 比例過高，代表 alpha 來源集中在少數極端股")
    log("       " + lu_exit.to_string().replace('\n', '\n       '))

    # C. Short limit-down（做空 D1 時跌停買不到）
    df['short_limit_down'] = df['return'] < -LIMIT_THRESHOLD
    lu_short = df[df['decile'] == 1].groupby('decile', observed=True)['short_limit_down'].mean()
    log("\n    C. Short Limit-Down（D1 做空時 return < -9.5%，跌停回補困難）：")
    log("       " + lu_short.to_string().replace('\n', '\n       '))

    # D10 vs D1 exit_gain_proxy 比值
    d10_lu = lu_exit.loc[10] if 10 in lu_exit.index else np.nan
    d1_lu  = lu_exit.loc[1]  if 1  in lu_exit.index else np.nan
    if d1_lu > 0:
        ratio = d10_lu / d1_lu
        log(f"\n    D10 / D1 exit_gain_proxy 比值 = {ratio:.1f}x  ", )
        if ratio > 10:
            log("    ⚠ 比值過高，D10 報酬嚴重集中在少數大漲股")
        elif ratio > 5:
            log("    ⚠ 比值偏高，建議進一步過濾 exit_gain_proxy 股後重測")
        else:
            log("    ✅ 比值尚可")

    # ─────────────────────────────────
    # 4  Decile 平均報酬 & 單調性
    # ─────────────────────────────────
    log("\n[4] DECILE MEAN RETURN & MONOTONICITY")
    decile_ret = df.groupby('decile', observed=True)['return'].mean()
    log("    " + decile_ret.to_string().replace('\n', '\n    '))
    diff = decile_ret.diff()
    violations = (diff.dropna() < 0).sum()
    log(f"\n    單調性違反次數（相鄰 decile 均值下降）: {violations} / {len(diff.dropna())}")
    if violations <= 2:
        log("    ✅ 單調性尚可")
    else:
        log("    ⚠ 單調性較差（中間段分層效果弱）")

    # ─────────────────────────────────
    # 5  IC 檢查
    # ─────────────────────────────────
    log("\n[5] IC CHECK")
    ics = []
    for d, g in df.groupby('date'):
        if len(g) < 5:
            continue
        ic, _ = spearmanr(g['y_prob'], g['return'])
        if not np.isnan(ic):
            ics.append(ic)
    ics_s = pd.Series(ics)
    ic_mean_val = ics_s.mean()
    ic_std_val  = ics_s.std()
    ic_ir_val   = ic_mean_val / ic_std_val if ic_std_val > 0 else np.nan
    log(f"    IC Mean : {ic_mean_val:.6f}")
    log(f"    IC Std  : {ic_std_val:.6f}")
    log(f"    IC IR   : {ic_ir_val:.6f}")

    # 說明 IC 負值與 D10 最佳的一致性
    d10_ret = decile_ret.loc[10] if 10 in decile_ret.index else np.nan
    d1_ret  = decile_ret.loc[1]  if 1  in decile_ret.index else np.nan
    if ic_mean_val < 0 and d10_ret > d1_ret:
        log("    ℹ IC 為負 + D10 報酬最高 → 因子排序為「低 y_prob = 高報酬」")
        log("      (D10 = 低分群)，此為方向性問題，IC 符號本身不矛盾")
        log("      建議確認：df.groupby('decile')['y_prob'].mean() 是否 D10 最低")

    # ─────────────────────────────────
    # 6  D10 股票集中度
    # ─────────────────────────────────
    log("\n[6] STOCK CONCENTRATION (D10)")
    d10 = df[df['decile'] == 10]
    top10 = d10['證券代碼'].value_counts().head(10)
    total_d10_obs = len(d10)
    log("    出現次數前 10 名：")
    for code, cnt in top10.items():
        pct = cnt / total_d10_obs * 100
        log(f"      {code}  {cnt:5d} 次  ({pct:.2f}% of D10)")

    # ─────────────────────────────────
    # 7  Decile 大小均衡性
    # ─────────────────────────────────
    log("\n[7] DECILE SIZE BALANCE")
    size = df.groupby(['date', 'decile'], observed=True).size().groupby('decile', observed=True).mean()
    size_cv = size.std() / size.mean()
    log("    每日各 Decile 平均持股數：")
    log("    " + size.to_string().replace('\n', '\n    '))
    log(f"\n    CV（變異係數） = {size_cv:.4f}  ", )
    if size_cv < 0.05:
        log("✅ 各組大小均衡")
    else:
        log("⚠ 各組大小不均，可能有 duplicates='drop' 導致某組過小")

    # ─────────────────────────────────
    # 8  Return 分布（含波動率分層警告）
    # ─────────────────────────────────
    log("\n[8] RETURN DISTRIBUTION BY DECILE")
    dist = df.groupby('decile', observed=True)['return'].describe()[['mean', 'std', 'min', 'max']]
    log("    " + dist.to_string().replace('\n', '\n    '))
    d10_std = dist.loc[10, 'std'] if 10 in dist.index else np.nan
    d1_std  = dist.loc[1,  'std'] if 1  in dist.index else np.nan
    if d10_std / d1_std > 2.0:
        log(f"\n    ⚠ D10 std / D1 std = {d10_std/d1_std:.2f}x")
        log("      D10 顯著集中高波動股（小型/投機股），alpha 來源可能不純")
        log("      建議：加入市值 / 流動性 neutralization")

    # ─────────────────────────────────
    # 9  Future Leak Proxy（固定 seed）
    # ─────────────────────────────────
    log("\n[9] FUTURE LEAK PROXY TEST  (seed=42)")
    rng = np.random.default_rng(42)
    shuffled = df.copy()
    shuffled['return'] = rng.permutation(shuffled['return'].values)
    ics_shuf = []
    for d, g in shuffled.groupby('date'):
        if len(g) < 5:
            continue
        ic, _ = spearmanr(g['y_prob'], g['return'])
        if not np.isnan(ic):
            ics_shuf.append(ic)
    ic_shuf_mean = pd.Series(ics_shuf).mean()
    log(f"    Shuffled IC mean = {ic_shuf_mean:.6f}")
    if abs(ic_shuf_mean) < 0.005:
        log("    ✅ 無明顯 future leak（shuffle 後 IC ≈ 0）")
    else:
        log("    ⚠ Shuffled IC 偏離 0，請檢查是否有 future leak！")

    # ─────────────────────────────────
    # 10  Sanity Summary
    # ─────────────────────────────────
    log("\n" + "=" * 55)
    log("SANITY SUMMARY")
    log("=" * 55)
    log(f"    D10 exit_gain_proxy (return>9.5%)  : {d10_lu:.4%}  ← 持有期大漲比例（非直接執行風險）")
    if lu_flag == 'entry_limit_up':
        d10_entry_lu = df[df['decile'] == 10]['entry_limit_up'].mean()
        log(f"    D10 entry_limit_up (買不到)        : {d10_entry_lu:.4%}  ← 真正執行風險")
    else:
        log("    D10 entry_limit_up                 : N/A  （需 prev_close + open 欄位）")
    log(f"    IC mean                            : {ic_mean_val:.6f}")
    log(f"    Shuffled IC mean                   : {ic_shuf_mean:.6f}")
    log(f"    Monotonicity violations            : {violations}")
    log(f"    D10 mean return                    : {d10_ret:.6f}")
    log(f"    D10/D1 std ratio                   : {d10_std/d1_std:.2f}x")

    report = '\n'.join(lines)
    (output_dir / 'sanity_report.txt').write_text(report, encoding='utf-8')
    print(f"\n✅ sanity_report.txt 已儲存")
    return report


# ══════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════

def main():
    base_dir       = Path(__file__).resolve().parent
    experiment_dir = base_dir / 'database' / 'experiment'
    output_dir     = base_dir / 'database' / 'experiment_backtest'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 合併 ──
    print("=" * 60)
    print("Step 1  合併 predictions.csv")
    print("=" * 60)
    merged = merge_predictions(experiment_dir, output_dir)

    # ── Step 2: 每日分組 PnL ──
    print("=" * 60)
    print("Step 2  建立每日 Decile PnL")
    print("=" * 60)
    daily_pnl, stock_lists = build_daily_pnl(merged)
    daily_pnl.to_csv(output_dir / 'daily_pnl.csv', index=False, encoding='utf-8-sig')
    print(f"✅ daily_pnl.csv 完成  ({len(daily_pnl)} 筆, {daily_pnl['date'].nunique()} 個交易日)")

    # ── Step 3: IC & Factor Turnover ──
    print("\n計算 IC 序列...")
    ic_ser    = ic_series(merged)
    ic_mean   = float(ic_ser.mean())
    ic_std    = float(ic_ser.std())
    ic_ir     = float(ic_mean / ic_std) if ic_std > 0 else np.nan
    print(f"   IC Mean={ic_mean:.4f}  IC Std={ic_std:.4f}  IC IR={ic_ir:.4f}")

    print("計算 Factor Turnover...")
    ft = factor_turnover_series(merged)
    print(f"   Factor Turnover={ft:.4f}")

    # ── Step 4: 各組 checklist 指標 ──
    print("\n" + "=" * 60)
    print("Step 3  計算各組指標")
    print("=" * 60)

    market_ret = merged.groupby('年月日')['return'].mean()
    market_ret.index = pd.to_datetime(market_ret.index, infer_datetime_format=True)

    print("   計算個股層級 Win Rate...")
    merged_copy = merged.copy()
    merged_copy['date'] = pd.to_datetime(merged_copy['年月日'], infer_datetime_format=True)
    stock_winrate = {}
    for date, group in merged_copy.groupby('date'):
        if len(group) < 10:
            continue
        group = group.copy()
        group['decile'] = pd.qcut(group['y_prob'], q=10,
                                   labels=range(1, 11), duplicates='drop')
        for decile, dg in group.groupby('decile', observed=True):
            key = int(decile)
            if key not in stock_winrate:
                stock_winrate[key] = []
            stock_winrate[key].extend((dg['return'] > 0).tolist())

    d10_returns, d1_returns = [], []
    for date, group in merged_copy.groupby('date'):
        if len(group) < 10:
            continue
        group = group.copy()
        group['decile'] = pd.qcut(group['y_prob'], q=10,
                                   labels=range(1, 11), duplicates='drop')
        d10_group = group[group['decile'] == 10]
        d1_group  = group[group['decile'] == 1]
        d10_returns.extend((d10_group['return'] > 0).tolist())
        d1_returns.extend((d1_group['return'] < 0).tolist())
    stock_winrate[11] = d10_returns + d1_returns

    sorted_dates = sorted(daily_pnl['date'].unique())
    all_metrics  = []

    for decile in range(1, 11):
        dec_df  = daily_pnl[daily_pnl['decile'] == decile].set_index('date')['pnl']
        metrics = calc_metrics(dec_df, benchmark_series=market_ret)

        to_series, avg_to = calc_decile_turnover(stock_lists, decile, sorted_dates)
        n_avg    = daily_pnl[daily_pnl['decile'] == decile]['n_stocks'].mean()
        tx_cost  = calc_transaction_cost(to_series, n_avg)
        metrics['avg_turnover']   = round(avg_to, 6) if not np.isnan(avg_to) else np.nan
        metrics['annual_tx_cost'] = tx_cost

        wr_stock = float(np.mean(stock_winrate[decile])) if decile in stock_winrate else np.nan
        metrics['win_rate_stock']  = round(wr_stock, 6) if not np.isnan(wr_stock) else np.nan
        metrics['decile']          = decile
        metrics['ic_mean']         = round(ic_mean, 6)
        metrics['ic_ir']           = round(ic_ir, 6) if not np.isnan(ic_ir) else np.nan
        metrics['factor_turnover'] = round(ft, 6)
        all_metrics.append(metrics)
        print(f"   Decile {decile:2d}  Sharpe={metrics.get('sharpe_ratio', 'N/A'):.3f}  "
              f"MaxDD={metrics.get('max_drawdown', 'N/A'):.3f}  "
              f"WR_daily={metrics['win_rate_daily']:.2%}  WR_stock={wr_stock:.2%}")

    # ── D11: L/S ──
    d10 = daily_pnl[daily_pnl['decile'] == 10].set_index('date')['pnl']
    d1  = daily_pnl[daily_pnl['decile'] ==  1].set_index('date')['pnl']
    ls_dates   = d10.index.intersection(d1.index)
    ls_pnl     = (d10.loc[ls_dates] - d1.loc[ls_dates]) / 2
    ls_metrics = calc_metrics(ls_pnl, benchmark_series=None)

    to10, avg_to10 = calc_decile_turnover(stock_lists, 10, sorted_dates)
    to1,  avg_to1  = calc_decile_turnover(stock_lists,  1, sorted_dates)
    ls_avg_to  = (avg_to10 + avg_to1) / 2
    ls_tx_cost = calc_transaction_cost(
        [(a + b) / 2 for a, b in zip(to10, to1[:len(to10)])], 0
    )
    ls_wr_stock = float(np.mean(stock_winrate[11])) if 11 in stock_winrate else np.nan
    ls_metrics['win_rate_stock']  = round(ls_wr_stock, 6) if not np.isnan(ls_wr_stock) else np.nan
    ls_metrics['avg_turnover']    = round(ls_avg_to, 6) if not np.isnan(ls_avg_to) else np.nan
    ls_metrics['annual_tx_cost']  = ls_tx_cost
    ls_metrics['decile']          = 11
    ls_metrics['ic_mean']         = round(ic_mean, 6)
    ls_metrics['ic_ir']           = round(ic_ir, 6) if not np.isnan(ic_ir) else np.nan
    ls_metrics['factor_turnover'] = round(ft, 6)
    all_metrics.append(ls_metrics)
    print(f"   Decile 11 (L/S)  Sharpe={ls_metrics.get('sharpe_ratio', 'N/A'):.3f}  "
          f"MaxDD={ls_metrics.get('max_drawdown', 'N/A'):.3f}  "
          f"Turnover={ls_avg_to:.2%}  TxCost={ls_tx_cost:.2%}/yr")

    ls_rows = pd.DataFrame({
        'date':     ls_dates,
        'decile':   11,
        'n_stocks': 0,
        'pnl':      ls_pnl.values
    })
    daily_pnl = pd.concat([daily_pnl, ls_rows], ignore_index=True)
    daily_pnl = daily_pnl.sort_values(['date', 'decile']).reset_index(drop=True)

    metrics_df = pd.DataFrame(all_metrics)
    cols       = ['decile'] + [c for c in metrics_df.columns if c != 'decile']
    metrics_df = metrics_df[cols]
    metrics_df.to_csv(output_dir / 'backtest_metrics.csv', index=False, encoding='utf-8-sig')

    ic_df = ic_ser.reset_index()
    ic_df.columns = ['date', 'ic']
    ic_df.to_csv(output_dir / 'ic_series.csv', index=False, encoding='utf-8-sig')

    # ── Step 5: 健全性檢查 ──
    print("\n" + "=" * 60)
    print("Step 5  Sanity Check")
    print("=" * 60)
    merged_check = pd.read_csv(output_dir / 'experiment_merge_prediction.csv',
                                encoding='utf-8-sig')
    run_sanity_check(merged_check, output_dir)

    # ── Step 6: 圖表 ──
    print("\n" + "=" * 60)
    print("Step 6  生成圖表")
    print("=" * 60)
    plot_all(daily_pnl, metrics_df, ic_ser, output_dir)

    print(f"\n{'='*60}")
    print("✅ 回測完成！")
    print(f"{'='*60}")
    print(f"   {output_dir / 'backtest_metrics.csv'}")
    print(f"   {output_dir / 'sanity_report.txt'}")

    print(f"\n{'─'*60}")
    print(f"{'Decile':>8} {'TotalRet':>10} {'Sharpe':>8} {'MaxDD':>8} {'WR_daily':>9} {'WR_stock':>9}")
    print(f"{'─'*60}")
    for _, row in metrics_df.iterrows():
        print(f"  D{int(row['decile']):02d}    "
              f"{row['total_return']:>9.2%}  "
              f"{row['sharpe_ratio']:>7.3f}  "
              f"{row['max_drawdown']:>7.2%}  "
              f"{row['win_rate_daily']:>8.2%}  "
              f"{row['win_rate_stock']:>8.2%}")


if __name__ == "__main__":
    main()