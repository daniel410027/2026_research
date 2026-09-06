"""
7_stock_corr_v2.py
==================

在原版基礎上新增大盤相關性分析，計算超額報酬版本的 LOO 相關性。

【新增功能】
  1. 從 database/market/market.csv 讀入加權指數收盤價，轉為日報酬
  2. 計算每支個股的 Beta（對大盤）與 Alpha
  3. 計算「超額報酬版 LOO 相關性」：
       excess_return_i     = return_i     - market_ret
       excess_portfolio    = portfolio_ret - market_ret
       excess_portfolio_loo = LOO 版本的 excess_portfolio
       ρ_excess = corr(excess_return_i, excess_portfolio_loo)
  4. 比較 corr_loo vs corr_excess：
       若 corr_loo 高但 corr_excess 低 → 兩者相關主要來自共同大盤暴露
       若 corr_excess 仍高             → 兩者有真實的因子共線

【輸入】
  database/experiment_backtest/experiment_merge_prediction.csv
  database/experiment_backtest/daily_pnl.csv
  database/market/market.csv   （格式：證券代碼, 年月日, 收盤價(元)）

【輸出】
  database/experiment_backtest/stock_portfolio_corr_v2.csv
  database/experiment_backtest/chart_corr_v2.png
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

MIN_DAYS      = 60
ANNUAL_FACTOR = 252

BASE       = Path(__file__).resolve().parent
PRED_PATH  = BASE / "database" / "experiment_backtest" / "experiment_merge_prediction.csv"
PNL_PATH   = BASE / "database" / "experiment_backtest" / "daily_pnl.csv"
MKT_PATH   = BASE / "database" / "market" / "market.csv"
OUTPUT_DIR = BASE / "database" / "experiment_backtest"


# ══════════════════════════════════════════════
# 1  讀入並轉換大盤日報酬
# ══════════════════════════════════════════════

print("讀入大盤資料...")

mkt_raw = pd.read_csv(MKT_PATH, encoding='utf-8-sig')

# 找收盤價欄位（容錯不同命名）
price_col = [c for c in mkt_raw.columns if '收盤' in c][0]

# 取加權指數（Y9999 開頭）
mkt_raw = mkt_raw[mkt_raw['證券代碼'].astype(str).str.contains('Y9999')]

mkt_raw['date'] = pd.to_datetime(mkt_raw['年月日'].astype(str), format='%Y%m%d', errors='coerce')
mkt_raw = mkt_raw.dropna(subset=['date'])
mkt_raw = mkt_raw.sort_values('date')
mkt_raw[price_col] = pd.to_numeric(mkt_raw[price_col], errors='coerce')
mkt_raw = mkt_raw.dropna(subset=[price_col])

# 日報酬 = pct_change
mkt_ret = mkt_raw.set_index('date')[price_col].pct_change().dropna()
mkt_ret.name = 'market_ret'

print(f"  大盤日報酬筆數 : {len(mkt_ret):,}")
print(f"  期間           : {mkt_ret.index.min().date()} ~ {mkt_ret.index.max().date()}")
print(f"  日均報酬       : {mkt_ret.mean():.4%}")
print(f"  日標準差       : {mkt_ret.std():.4%}")


# ══════════════════════════════════════════════
# 2  讀入預測與 D10 組合報酬
# ══════════════════════════════════════════════

print("\n讀入回測資料...")

pred = pd.read_csv(PRED_PATH, encoding='utf-8-sig')
pred['date'] = pd.to_datetime(pred['年月日'].astype(str))

pnl = pd.read_csv(PNL_PATH, encoding='utf-8-sig')
pnl['date'] = pd.to_datetime(pnl['date'])

d10_pnl = (
    pnl[pnl['decile'] == 10][['date', 'pnl', 'n_stocks']]
    .rename(columns={'pnl': 'portfolio_ret', 'n_stocks': 'N'})
    .set_index('date')
)


# ══════════════════════════════════════════════
# 3  重建 D10 個股清單
# ══════════════════════════════════════════════

print("重建 D10 個股清單...")

result = []
for d, g in pred.groupby('date'):
    if len(g) < 10:
        continue
    g = g.copy()
    g['decile'] = pd.qcut(g['y_prob'], 10, labels=range(1, 11), duplicates='drop')
    result.append(g[g['decile'] == 10][['date', '證券代碼', 'return']])

d10_stocks = pd.concat(result, ignore_index=True)


# ══════════════════════════════════════════════
# 4  合併大盤報酬
# ══════════════════════════════════════════════

df = d10_stocks.join(d10_pnl, on='date', how='inner')
df = df.join(mkt_ret, on='date', how='left')

# 對齊後確認大盤缺值
missing_mkt = df['market_ret'].isna().sum()
if missing_mkt > 0:
    print(f"  ⚠ 大盤報酬缺值 {missing_mkt} 筆（交易日不匹配），將以全股票等權均值填補")
    # fallback: 用全體股票每日等權均值
    eq_market = pred.groupby('date')['return'].mean().rename('market_ret_eq')
    df = df.join(eq_market, on='date', how='left')
    df['market_ret'] = df['market_ret'].fillna(df['market_ret_eq'])
    df = df.drop(columns=['market_ret_eq'])

# 重算 actual N（以實際分組為準）
df['N'] = df.groupby('date')['證券代碼'].transform('count')

# LOO 組合報酬
df['portfolio_loo'] = np.where(
    df['N'] > 1,
    (df['N'] * df['portfolio_ret'] - df['return']) / (df['N'] - 1),
    np.nan
)

# 超額報酬（vs 大盤）
df['excess_stock']     = df['return']        - df['market_ret']
df['excess_portfolio'] = df['portfolio_ret'] - df['market_ret']
df['excess_port_loo']  = df['portfolio_loo'] - df['market_ret']


# ══════════════════════════════════════════════
# 5  逐股計算相關性與 Beta
# ══════════════════════════════════════════════

print("計算個股相關性與 Beta...")

records = []

for code, g in df.groupby('證券代碼'):

    g = g.dropna(subset=['return', 'portfolio_loo', 'excess_port_loo', 'market_ret'])
    n = len(g)
    if n < MIN_DAYS:
        continue

    r        = g['return'].values
    p        = g['portfolio_ret'].values
    plo      = g['portfolio_loo'].values
    mkt      = g['market_ret'].values
    ex_r     = g['excess_stock'].values
    ex_plo   = g['excess_port_loo'].values

    def safe_corr(a, b):
        if np.std(a) > 0 and np.std(b) > 0:
            return float(np.corrcoef(a, b)[0, 1])
        return np.nan

    corr_loo    = safe_corr(r, plo)      # 原版 LOO
    corr_raw    = safe_corr(r, p)        # 未修正（對照）
    corr_excess = safe_corr(ex_r, ex_plo)  # 超額報酬版 LOO
    corr_mkt    = safe_corr(r, mkt)      # 個股 vs 大盤

    # Beta（OLS: return_i = alpha + beta * market）
    if np.var(mkt) > 0:
        beta  = float(np.cov(r, mkt)[0, 1] / np.var(mkt))
        alpha = float(r.mean() - beta * mkt.mean())
        alpha_ann = alpha * ANNUAL_FACTOR
    else:
        beta = alpha = alpha_ann = np.nan

    mean_ret   = float(r.mean())
    std_ret    = float(r.std())
    sharpe_stk = float(mean_ret / std_ret * np.sqrt(ANNUAL_FACTOR)) if std_ret > 0 else np.nan

    # 超額 Sharpe（個股超額報酬 / 其標準差）
    ex_std = float(ex_r.std())
    sharpe_excess = float(ex_r.mean() / ex_std * np.sqrt(ANNUAL_FACTOR)) if ex_std > 0 else np.nan

    # 判斷相關性來源
    if not np.isnan(corr_loo) and not np.isnan(corr_excess):
        decay = corr_loo - corr_excess
        if corr_loo > 0.3 and decay > 0.15:
            source = 'market_driven'   # 相關主要來自大盤暴露
        elif corr_excess > 0.2:
            source = 'factor_driven'   # 超額後仍高，真實因子共線
        else:
            source = 'low_corr'        # 整體相關低，分散效果好
    else:
        source = 'unknown'

    records.append({
        '證券代碼':       code,
        'n_days':         n,
        'corr_loo':       round(corr_loo,    4) if not np.isnan(corr_loo)    else np.nan,
        'corr_excess':    round(corr_excess, 4) if not np.isnan(corr_excess) else np.nan,
        'corr_decay':     round(corr_loo - corr_excess, 4) if not (np.isnan(corr_loo) or np.isnan(corr_excess)) else np.nan,
        'corr_mkt':       round(corr_mkt,   4) if not np.isnan(corr_mkt)    else np.nan,
        'corr_raw':       round(corr_raw,   4) if not np.isnan(corr_raw)    else np.nan,
        'beta':           round(beta,       4) if not np.isnan(beta)        else np.nan,
        'alpha_ann':      round(alpha_ann,  4) if not np.isnan(alpha_ann)   else np.nan,
        'mean_return':    round(mean_ret,   6),
        'std_return':     round(std_ret,    6),
        'sharpe_stock':   round(sharpe_stk, 4) if not np.isnan(sharpe_stk)  else np.nan,
        'sharpe_excess':  round(sharpe_excess, 4) if not np.isnan(sharpe_excess) else np.nan,
        'corr_source':    source,
    })

result_df = pd.DataFrame(records).sort_values('corr_loo', ascending=False)
print(f"  符合門檻 (≥{MIN_DAYS}天) 股票數 : {len(result_df):,}")


# ══════════════════════════════════════════════
# 6  存檔
# ══════════════════════════════════════════════

out_csv = OUTPUT_DIR / 'stock_portfolio_corr_v2.csv'
result_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n✅ 已儲存 → {out_csv}")


# ══════════════════════════════════════════════
# 7  Console Report
# ══════════════════════════════════════════════

def print_dist(series, label):
    vals = series.dropna()
    print(f"\n  {label}")
    print(f"    Mean={vals.mean():.4f}  Median={vals.median():.4f}  Std={vals.std():.4f}  "
          f"Min={vals.min():.4f}  Max={vals.max():.4f}")

print("\n" + "=" * 55)
print("相關性分析結果")
print("=" * 55)

print_dist(result_df['corr_loo'],    "LOO 相關係數（原版）")
print_dist(result_df['corr_excess'], "超額報酬 LOO 相關係數（扣大盤後）")
print_dist(result_df['corr_decay'],  "Decay（corr_loo - corr_excess），>0 代表部分相關來自大盤")
print_dist(result_df['corr_mkt'],    "個股 vs 大盤相關係數（Beta 側面指標）")
print_dist(result_df['beta'],        "個股 Beta（對大盤）")

# 相關性來源分布
print("\n\n  相關性來源分類：")
for src, cnt in result_df['corr_source'].value_counts().items():
    pct = cnt / len(result_df) * 100
    desc = {
        'market_driven': '大盤暴露主導（corr_loo 高但 corr_excess 低）',
        'factor_driven': '因子共線（扣大盤後仍高相關）',
        'low_corr':      '低相關（分散效果好）',
        'unknown':       '無法判斷'
    }.get(src, src)
    print(f"    {src:15s}  {cnt:4d} 支  ({pct:5.1f}%)  {desc}")

# 最值得關注：高 Alpha + 低 corr_excess
print("\n\n  ★ 理想持倉：高 alpha_ann + 低 corr_excess（≥60天）")
ideal = (
    result_df
    .dropna(subset=['alpha_ann', 'corr_excess'])
    .query("corr_excess < 0.2 and alpha_ann > 0")
    .sort_values('alpha_ann', ascending=False)
    .head(15)
)
print(ideal[['證券代碼','n_days','corr_loo','corr_excess','corr_decay',
             'beta','alpha_ann','sharpe_excess']].to_string(index=False))

# 負超額相關
print("\n\n  負超額相關股票（扣大盤後仍與組合負相關，最佳分散股）：")
neg_ex = result_df[result_df['corr_excess'] < 0].sort_values('corr_excess')
if len(neg_ex) > 0:
    print(neg_ex[['證券代碼','n_days','corr_loo','corr_excess',
                  'beta','alpha_ann','sharpe_excess']].to_string(index=False))
else:
    print("  （無）")


# ══════════════════════════════════════════════
# 8  視覺化
# ══════════════════════════════════════════════

print("\n生成圖表...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor('#0d1117')

# ── 8-1  LOO vs Excess 直方圖比較 ──
ax = axes[0, 0]
ax.set_facecolor('#0d1117')
vals_loo = result_df['corr_loo'].dropna()
vals_exc = result_df['corr_excess'].dropna()
bins = np.linspace(-0.3, 0.8, 45)
ax.hist(vals_loo, bins=bins, alpha=0.6, color='#4488cc', label=f'LOO  Mean={vals_loo.mean():.3f}')
ax.hist(vals_exc, bins=bins, alpha=0.6, color='#ff8844', label=f'Excess  Mean={vals_exc.mean():.3f}')
ax.axvline(vals_loo.mean(), color='#4488cc', linewidth=1.5, linestyle='--')
ax.axvline(vals_exc.mean(), color='#ff8844', linewidth=1.5, linestyle='--')
ax.set_title('LOO vs Excess Correlation Distribution', color='white', fontsize=11)
ax.set_xlabel('Correlation', color='#aaaaaa')
ax.set_ylabel('Count', color='#aaaaaa')
ax.tick_params(colors='#aaaaaa')
ax.spines[:].set_color('#333333')
ax.legend(fontsize=9, framealpha=0.2, labelcolor='white', facecolor='#1a1a2e')

# ── 8-2  LOO vs Excess 散點（每支股票一點）──
ax = axes[0, 1]
ax.set_facecolor('#0d1117')
both = result_df.dropna(subset=['corr_loo', 'corr_excess'])
color_map = {'market_driven': '#ff4444', 'factor_driven': '#ffaa00',
             'low_corr': '#44cc88', 'unknown': '#888888'}
for src, grp in both.groupby('corr_source'):
    ax.scatter(grp['corr_loo'], grp['corr_excess'],
               alpha=0.5, s=12, color=color_map.get(src, '#888888'), label=src)
diag = np.linspace(-0.3, 0.8, 100)
ax.plot(diag, diag, color='#555555', linewidth=1, linestyle='--', label='y=x (no decay)')
ax.axhline(0, color='#444444', linewidth=0.7)
ax.axvline(0, color='#444444', linewidth=0.7)
ax.set_title('LOO vs Excess Correlation\n(below diagonal = market-driven)', color='white', fontsize=11)
ax.set_xlabel('corr_loo', color='#aaaaaa')
ax.set_ylabel('corr_excess', color='#aaaaaa')
ax.tick_params(colors='#aaaaaa')
ax.spines[:].set_color('#333333')
ax.legend(fontsize=8, framealpha=0.2, labelcolor='white', facecolor='#1a1a2e')

# ── 8-3  Beta 分布 ──
ax = axes[1, 0]
ax.set_facecolor('#0d1117')
betas = result_df['beta'].dropna()
ax.hist(betas, bins=40, color='#aa66cc', edgecolor='#0d1117', linewidth=0.3)
ax.axvline(betas.mean(),   color='#FFD700', linewidth=1.5, linestyle='--', label=f'Mean={betas.mean():.3f}')
ax.axvline(1.0, color='#ff6666', linewidth=1.0, linestyle=':',  label='Beta=1')
ax.set_title('Stock Beta Distribution (vs Market)', color='white', fontsize=11)
ax.set_xlabel('Beta', color='#aaaaaa')
ax.set_ylabel('Count', color='#aaaaaa')
ax.tick_params(colors='#aaaaaa')
ax.spines[:].set_color('#333333')
ax.legend(fontsize=9, framealpha=0.2, labelcolor='white', facecolor='#1a1a2e')

# ── 8-4  Excess Sharpe vs corr_excess（理想股票右上角）──
ax = axes[1, 1]
ax.set_facecolor('#0d1117')
valid = result_df.dropna(subset=['corr_excess', 'sharpe_excess'])
valid = valid[valid['sharpe_excess'].abs() < 10]
sc = ax.scatter(valid['corr_excess'], valid['sharpe_excess'],
                alpha=0.4, s=12, c=valid['n_days'], cmap='YlOrRd')
ax.axhline(0, color='#555555', linewidth=0.8)
ax.axvline(0, color='#555555', linewidth=0.8)
ax.axvline(0.2, color='#336633', linewidth=0.8, linestyle=':', alpha=0.7)
ax.text(0.21, valid['sharpe_excess'].max() * 0.85, '← ideal zone',
        color='#44cc88', fontsize=8)
cb = plt.colorbar(sc, ax=ax)
cb.ax.tick_params(colors='#aaaaaa')
cb.set_label('n_days', color='#aaaaaa')
ax.set_title('Excess Sharpe vs Excess Correlation\n(ideal: top-left)', color='white', fontsize=11)
ax.set_xlabel('corr_excess (after removing market)', color='#aaaaaa')
ax.set_ylabel('Sharpe (excess return)', color='#aaaaaa')
ax.tick_params(colors='#aaaaaa')
ax.spines[:].set_color('#333333')

plt.tight_layout()
out_png = OUTPUT_DIR / 'chart_corr_v2.png'
fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"✅ 已儲存 → {out_png}")

print("\n完成！")
print("  stock_portfolio_corr_v2.csv  →  含 corr_loo / corr_excess / beta / alpha_ann")
print("  chart_corr_v2.png            →  4 張子圖")
print("\n  欄位說明：")
print("  corr_loo      : 個股 vs D10 組合（LOO 修正）")
print("  corr_excess   : 個股超額報酬 vs D10 超額報酬（扣大盤後）")
print("  corr_decay    : corr_loo - corr_excess，愈大代表相關愈多來自大盤")
print("  beta          : 個股對大盤的系統性暴露")
print("  alpha_ann     : 個股年化超額報酬（扣 beta * 大盤）")
print("  sharpe_excess : 個股超額報酬 Sharpe")
print("  corr_source   : market_driven / factor_driven / low_corr")