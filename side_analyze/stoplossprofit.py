#!/usr/bin/env python3
"""
stoplossprofit.py
Stop-loss / stop-profit grid-search backtest
─────────────────────────────────────────────
Train : 2016-2021
Valid : 2022-2023
Test  : 2024-2025

賣出觸發：
  (A) 當日 y_prob rank > 100  → 以當日開盤價賣出
  (B) 日內 Low  <= entry*(1-sl) → 以 sl_price 賣出
  (C) 日內 High >= entry*(1+sp) → 以 sp_price 賣出
  (B+C 同時) 比較開盤與兩觸發價距離，近者優先

買入規則：
  - 賣出後隔日開盤價買入
  - 合併當日所有賣出 proceeds，若 > 200萬 → 買兩檔，否則買一檔
  - 候選股按當日 y_prob 排名，排除現有持股
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "database/experiment_backtest/experiment_merge_prediction.csv"
OUTPUT_DIR  = "database/stoplossprofit"
INITIAL_CAP = 10_000_000
N_STOCKS    = 10
PROC_THRESH = 2_000_000          # 超過此金額 → 買兩檔

# ── 台灣交易成本 ──────────────────────────────────────────────────────────────
COMMISSION_RATE = 0.001425       # 手續費 0.1425%（買賣各收）
TAX_RATE        = 0.003          # 證券交易稅 0.3%（僅賣出）
MIN_COMMISSION  = 20             # 最低手續費 NT$20

# ── 參數搜索空間 ──────────────────────────────────────────────────────────────
SL_BOUNDS   = (0.03, 0.30)     # stop-loss  連續
SP_BOUNDS   = (0.05, 0.25)     # stop-profit 連續
RANK_BOUNDS = (100, 600)       # rank 閾值  整數

# ── 遺傳演算法超參數 ───────────────────────────────────────────────────────────
GA_POP_SIZE   = 50     # 族群大小
GA_GENS       = 10     # 世代數
GA_ELITE      = 3      # 精英保留數
GA_CX_PROB    = 0.7    # 交叉機率
GA_MUT_PROB   = 0.25   # 突變機率
GA_TOURN_SIZE = 4      # 錦標賽選擇 k
GA_SEED       = 42

SPLITS = {
    "train": ("2016-01-01", "2022-12-31"),   # 7 年
    "test":  ("2023-01-01", "2025-12-31"),   # 3 年
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pv: pd.Series) -> dict:
    pv = pv.dropna()
    if len(pv) < 2:
        return {k: np.nan for k in
                ["total_return", "annual_return", "annual_vol", "sharpe", "max_drawdown"]}
    daily_ret = pv.pct_change().dropna()
    total_ret  = pv.iloc[-1] / pv.iloc[0] - 1
    n_days     = len(pv)
    ann_ret    = (1 + total_ret) ** (252 / n_days) - 1
    ann_vol    = daily_ret.std() * np.sqrt(252)
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd     = ((pv - pv.cummax()) / pv.cummax()).min()
    return {
        "total_return":  round(total_ret, 6),
        "annual_return": round(ann_ret,   6),
        "annual_vol":    round(ann_vol,   6),
        "sharpe":        round(sharpe,    4),
        "max_drawdown":  round(max_dd,    6),
    }


# ── Backtest Core ─────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, sl_pct: float, sp_pct: float,
                 rank_thresh: int) -> tuple:
    """
    Returns:
        pv          : pd.Series  daily portfolio value
        trade_stats : dict       交易成本、換手率、單次交易勝率、賣出原因比例
    """
    dates = sorted(df["年月日"].unique())
    if not dates:
        empty_stats = {
            "total_cost": np.nan, "cost_pct": np.nan,
            "annual_turnover": np.nan,
            "trade_win_rate": np.nan, "n_trades": 0,
            "sell_A": 0, "sell_B": 0, "sell_C": 0, "sell_BC": 0,
            "pct_A": np.nan, "pct_B": np.nan,
            "pct_C": np.nan, "pct_BC": np.nan,
        }
        return pd.Series(dtype=float), empty_stats

    cash        = float(INITIAL_CAP)
    portfolio   = {}       # code -> {shares, entry_price, cost, buy_cost_paid}
    pending_buy = None
    pv_list     = []

    # ── 統計累計器 ────────────────────────────────────────────────────────────
    total_cost_paid  = 0.0   # 所有手續費 + 交易稅
    total_buy_value  = 0.0   # 所有買入名義金額（用於換手率分母）
    total_sell_value = 0.0   # 所有賣出名義金額
    trade_pnl        = []    # 每筆已平倉交易的損益（含成本）
    sell_reason_cnt  = {"A": 0, "B": 0, "C": 0, "BC": 0}  # 賣出原因計數

    def _buy(alloc: float, op: float, code: str) -> dict:
        """執行買入，回傳 position dict，並累計統計。"""
        nonlocal total_cost_paid, total_buy_value
        commission      = max(alloc * COMMISSION_RATE, MIN_COMMISSION)
        total_cost_paid += commission
        total_buy_value += alloc
        net_alloc        = alloc - commission
        shares           = net_alloc / op
        return {"shares": shares, "entry_price": op,
                "cost": alloc, "buy_cost_paid": commission}

    def _sell(pos: dict, sell_at: float) -> float:
        """執行賣出，回傳稅後 proceeds，並累計統計。"""
        nonlocal total_cost_paid, total_sell_value
        sh               = pos["shares"]
        gross            = sh * sell_at
        commission       = max(gross * COMMISSION_RATE, MIN_COMMISSION)
        tax              = gross * TAX_RATE
        total_cost_paid += commission + tax
        total_sell_value += gross
        net_proceeds     = gross - commission - tax
        # 單筆 PnL = 淨收入 - 原始買入支出（含買入手續費）
        trade_pnl.append(net_proceeds - pos["cost"])
        return net_proceeds

    # ── 初始建倉 ──────────────────────────────────────────────────────────────
    d0 = (df[df["年月日"] == dates[0]]
          .set_index("證券代碼")
          .query("`開盤價元` > 0"))
    init_stocks = d0.nlargest(N_STOCKS, "y_prob").index.tolist()
    if init_stocks:
        alloc = cash / len(init_stocks)
        for code in init_stocks:
            op             = d0.loc[code, "開盤價元"]
            pos            = _buy(alloc, op, code)
            portfolio[code] = pos
            cash           -= alloc

    # ── 逐日迴圈 ──────────────────────────────────────────────────────────────
    for date in dates:
        day = df[df["年月日"] == date].copy().set_index("證券代碼")
        day["prob_rank"] = day["y_prob"].rank(ascending=False, method="min")

        # Step 1: 執行昨日掛單（以今日開盤價買入）
        if pending_buy is not None:
            buy_cash, candidates = pending_buy
            n_buy = 2 if buy_cash > PROC_THRESH else 1
            avail = [
                c for c in candidates
                if c in day.index
                and c not in portfolio
                and day.loc[c, "開盤價元"] > 0
            ]
            n_buy = min(n_buy, len(avail))
            if n_buy > 0:
                per_stock = buy_cash / n_buy
                for code in avail[:n_buy]:
                    op              = day.loc[code, "開盤價元"]
                    pos             = _buy(per_stock, op, code)
                    portfolio[code] = pos
                    cash           -= per_stock
            else:
                cash += buy_cash
            pending_buy = None

        # Step 2: 檢查賣出條件
        sell_pool = 0.0
        to_sell   = []

        for code, pos in portfolio.items():
            if code not in day.index:
                continue
            row   = day.loc[code]
            entry = pos["entry_price"]
            sh    = pos["shares"]
            hi    = row["最高價元"]
            lo    = row["最低價元"]
            op    = row["開盤價元"]
            rank  = row["prob_rank"]

            sp_price = entry * (1 + sp_pct)
            sl_price = entry * (1 - sl_pct)
            sell_at  = None

            sell_reason = None

            if rank > rank_thresh:
                sell_at     = max(op, 0.01)
                sell_reason = "A"

            if sell_at is None:
                hit_sp = hi >= sp_price
                hit_sl = lo <= sl_price
                if hit_sp and hit_sl:
                    sell_at     = sp_price if abs(op - sp_price) <= abs(op - sl_price) \
                                  else sl_price
                    sell_reason = "BC"
                elif hit_sp:
                    sell_at     = sp_price
                    sell_reason = "C"
                elif hit_sl:
                    sell_at     = sl_price
                    sell_reason = "B"

            if sell_at is not None and sell_at > 0:
                sell_pool += _sell(pos, sell_at)
                to_sell.append(code)
                sell_reason_cnt[sell_reason] += 1

        for code in to_sell:
            del portfolio[code]
        cash += sell_pool

        # Step 3: 設定隔日買入掛單
        if sell_pool > 0:
            held = set(portfolio.keys())
            candidates = (
                day[~day.index.isin(held)]
                .sort_values("y_prob", ascending=False)
                .index.tolist()
            )
            pending_buy = (sell_pool, candidates)

        # Step 4: Mark-to-market
        pv = cash
        for code, pos in portfolio.items():
            if code in day.index:
                cp = day.loc[code, "收盤價元"]
                pv += pos["shares"] * (cp if cp > 0 else pos["entry_price"])
            else:
                pv += pos["cost"]
        pv_list.append({"date": date, "value": pv})

    # ── 計算 Trade Stats ──────────────────────────────────────────────────────
    n_days       = len(dates)
    n_years      = n_days / 252
    n_trades     = len(trade_pnl)

    # 換手率 = 單邊買入金額 / 平均資產規模 / 年數
    avg_pv       = np.mean([r["value"] for r in pv_list]) if pv_list else INITIAL_CAP
    annual_to    = (total_buy_value / avg_pv / n_years) if n_years > 0 else np.nan

    # 交易成本佔初始資本比
    cost_pct     = total_cost_paid / INITIAL_CAP

    # 單次交易勝率（已平倉）
    win_rate     = (np.sum(np.array(trade_pnl) > 0) / n_trades) if n_trades > 0 else np.nan

    trade_stats = {
        "total_cost":      round(total_cost_paid, 0),
        "cost_pct":        round(cost_pct, 6),
        "annual_turnover": round(annual_to, 4),
        "trade_win_rate":  round(win_rate, 4),
        "n_trades":        n_trades,
        # 賣出原因絕對筆數
        "sell_A":          sell_reason_cnt["A"],
        "sell_B":          sell_reason_cnt["B"],
        "sell_C":          sell_reason_cnt["C"],
        "sell_BC":         sell_reason_cnt["BC"],
        # 賣出原因佔比（以已平倉總筆數為分母）
        "pct_A":  round(sell_reason_cnt["A"]  / n_trades, 4) if n_trades else np.nan,
        "pct_B":  round(sell_reason_cnt["B"]  / n_trades, 4) if n_trades else np.nan,
        "pct_C":  round(sell_reason_cnt["C"]  / n_trades, 4) if n_trades else np.nan,
        "pct_BC": round(sell_reason_cnt["BC"] / n_trades, 4) if n_trades else np.nan,
    }

    return pd.DataFrame(pv_list).set_index("date")["value"], trade_stats


# ── Plot PnL ──────────────────────────────────────────────────────────────────
def plot_pnl(pv: pd.Series, sl: float, sp: float, rank: int,
             split: str, metrics: dict):
    fig, ax = plt.subplots(figsize=(12, 5))
    cum = pv / pv.iloc[0]
    cum.plot(ax=ax, color="steelblue", lw=1.5)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.fill_between(cum.index, cum, 1.0,
                    where=cum < 1.0, alpha=0.15, color="red")
    ax.fill_between(cum.index, cum, 1.0,
                    where=cum > 1.0, alpha=0.10, color="steelblue")
    ax.set_title(
        f"SL={sl*100:.0f}%  SP={sp*100:.0f}%  R={rank}  |  {split.upper()}  |  "
        f"Ann.Ret={metrics['annual_return']*100:.1f}%  "
        f"Sharpe={metrics['sharpe']:.2f}  "
        f"MDD={metrics['max_drawdown']*100:.1f}%",
        fontsize=10,
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("")
    plt.tight_layout()
    out = os.path.join(
        OUTPUT_DIR,
        f"pnl_{split}_sl{int(sl*100):02d}_sp{int(sp*100):02d}_r{rank}.png"
    )
    plt.savefig(out, dpi=120)
    plt.close()


# ── Heatmap ───────────────────────────────────────────────────────────────────
def plot_heatmap(df_sum: pd.DataFrame, metric: str, split: str):
    sub = df_sum[df_sum["split"] == split]
    if sub.empty:
        return
    pivot = sub.pivot(index="sl", columns="sp", values=metric)
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = "RdYlGn" if metric != "max_drawdown" else "RdYlGn_r"
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"{v*100:.0f}%" for v in pivot.columns])
    ax.set_yticklabels([f"{v*100:.0f}%" for v in pivot.index])
    ax.set_xlabel("Stop-Profit (%)")
    ax.set_ylabel("Stop-Loss (%)")
    ax.set_title(f"{metric}  |  {split.upper()}", fontsize=11)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            txt = f"{v:.2f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"heatmap_{split}_{metric}.png")
    plt.savefig(out, dpi=120)
    plt.close()


# ── Combined best-combo PnL (train 最佳組合在兩個 split 的走勢) ──────────────
def plot_best_combo(best_sl: float, best_sp: float,
                    pv_dict: dict):   # {split: Series}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    colors = {"train": "steelblue", "test": "seagreen"}
    for ax, split in zip(axes, ["train", "test"]):
        if split not in pv_dict:
            ax.set_visible(False)
            continue
        pv  = pv_dict[split]
        cum = pv / pv.iloc[0]
        cum.plot(ax=ax, color=colors[split], lw=1.5)
        m   = compute_metrics(pv)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax.set_title(
            f"{split.upper()}\nAnn={m['annual_return']*100:.1f}%  "
            f"Sharpe={m['sharpe']:.2f}  MDD={m['max_drawdown']*100:.1f}%",
            fontsize=9,
        )
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    fig.suptitle(
        f"Best Combo (Train Min-MDD):  SL={best_sl*100:.0f}%  SP={best_sp*100:.0f}%",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "best_combo_all_splits.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


# ── GA 工具函數 ───────────────────────────────────────────────────────────────

def _clip(ind: list) -> list:
    """將個體的三個基因 clamp 到合法範圍。"""
    sl   = float(np.clip(ind[0], *SL_BOUNDS))
    sp   = float(np.clip(ind[1], *SP_BOUNDS))
    rank = int(np.clip(round(ind[2]), *RANK_BOUNDS))
    return [sl, sp, rank]


def _random_ind(rng: random.Random) -> list:
    sl   = rng.uniform(*SL_BOUNDS)
    sp   = rng.uniform(*SP_BOUNDS)
    rank = rng.randint(*RANK_BOUNDS)
    return [sl, sp, rank]


def _evaluate(ind: list, df_train: pd.DataFrame) -> tuple:
    """回傳 (fitness_scalar, metrics_dict, tstats_dict)。只跑一次 backtest。
    目標函數：Ann - |MDD| = ann + mdd（兩者皆為小數，越大越好）
    """
    sl, sp, rank = ind
    pv, tstats = run_backtest(df_train, sl, sp, rank)
    m       = compute_metrics(pv)
    ann     = m["annual_return"]
    mdd     = m["max_drawdown"]   # 負數
    fitness = ann + mdd            # = Ann - |MDD|
    if np.isnan(fitness):
        fitness = -2.0             # 無效個體懲罰
    return float(fitness), m, tstats


def _fitness(ind: list, df_train: pd.DataFrame) -> float:
    """目標函數：Ann - |MDD|（越大越好）。"""
    f, _, _ = _evaluate(ind, df_train)
    return f


def _tournament(pop: list, fits: list, k: int, rng: random.Random) -> list:
    """錦標賽選擇，回傳勝者的複製。"""
    contenders = rng.sample(range(len(pop)), k)
    winner = max(contenders, key=lambda i: fits[i])
    return pop[winner][:]


def _blx_crossover(p1: list, p2: list, alpha: float,
                   rng: random.Random) -> tuple:
    """BLX-α 交叉（連續基因），rank 使用均勻交叉。"""
    c1, c2 = p1[:], p2[:]
    for i in range(2):   # sl, sp 連續
        lo = min(p1[i], p2[i]) - alpha * abs(p1[i] - p2[i])
        hi = max(p1[i], p2[i]) + alpha * abs(p1[i] - p2[i])
        c1[i] = rng.uniform(lo, hi)
        c2[i] = rng.uniform(lo, hi)
    # rank 整數均勻交叉
    if rng.random() < 0.5:
        c1[2], c2[2] = p2[2], p1[2]
    return _clip(c1), _clip(c2)


def _mutate(ind: list, rng: random.Random) -> list:
    """高斯突變（連續）+ 均勻步進突變（整數）。"""
    ind = ind[:]
    # SL 突變
    if rng.random() < 0.5:
        ind[0] += rng.gauss(0, (SL_BOUNDS[1] - SL_BOUNDS[0]) * 0.1)
    # SP 突變
    if rng.random() < 0.5:
        ind[1] += rng.gauss(0, (SP_BOUNDS[1] - SP_BOUNDS[0]) * 0.1)
    # Rank 突變
    if rng.random() < 0.5:
        ind[2] += rng.randint(-50, 50)
    return _clip(ind)


# ── GA Plot ───────────────────────────────────────────────────────────────────

def plot_ga_convergence(best_history: list, mean_history: list):
    fig, ax = plt.subplots(figsize=(10, 4))
    gens = range(len(best_history))
    ax.plot(gens, [v * 100 for v in best_history],
            color="steelblue", lw=2, label="Best MDD")
    ax.plot(gens, [v * 100 for v in mean_history],
            color="gray", lw=1, ls="--", label="Mean MDD")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Ann - |MDD|")
    ax.set_title("GA Convergence — Train  Ann - |MDD|  (目標：最大化)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "ga_convergence.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


def plot_ga_population(all_evals: list):
    """散點圖：所有曾評估過的個體，顏色代表 train MDD。"""
    df_e = pd.DataFrame(all_evals,
                        columns=["sl", "sp", "rank", "mdd"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [("sl", "sp"), ("sl", "rank"), ("sp", "rank")]
    for ax, (x, y) in zip(axes, pairs):
        sc = ax.scatter(df_e[x] * (100 if x != "rank" else 1),
                        df_e[y] * (100 if y != "rank" else 1),
                        c=df_e["mdd"] * 100,
                        cmap="RdYlGn", s=20, alpha=0.6)
        ax.set_xlabel(x.upper() + (" (%)" if x != "rank" else ""))
        ax.set_ylabel(y.upper() + (" (%)" if y != "rank" else ""))
        plt.colorbar(sc, ax=ax, label="MDD (%)")
    fig.suptitle("GA Search Space — All Evaluated Individuals", fontsize=11)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "ga_search_space.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Loading data …")
    df = pd.read_csv(DATA_PATH, parse_dates=["年月日"])
    df = df.dropna(subset=["y_prob", "開盤價元", "最高價元", "最低價元", "收盤價元"])
    df = df[(df["開盤價元"] > 0) & (df["收盤價元"] > 0)].copy()
    print(f"  Total rows: {len(df):,}  |  "
          f"Dates: {df['年月日'].min().date()} ~ {df['年月日'].max().date()}")

    # ── Split ────────────────────────────────────────────────────────────────
    split_dfs = {}
    for name, (s, e) in SPLITS.items():
        mask = (df["年月日"] >= s) & (df["年月日"] <= e)
        split_dfs[name] = df[mask].reset_index(drop=True)
        print(f"  {name:5s}: {mask.sum():7,} rows  "
              f"{split_dfs[name]['年月日'].nunique():4d} days")

    df_train = split_dfs["train"]
    df_test  = split_dfs["test"]

    rng = random.Random(GA_SEED)
    np.random.seed(GA_SEED)

    # ── 初始族群 ─────────────────────────────────────────────────────────────
    pop  = [_random_ind(rng) for _ in range(GA_POP_SIZE)]
    print(f"Evaluating initial population ({GA_POP_SIZE} individuals) …")
    print(f"  {'#':>4}  {'SL':>6} {'SP':>6} {'R':>5}  │  "
          f"{'MDD':>8} {'Ann':>8} {'Std':>8} {'SR':>6} {'WinR':>6} {'TO':>6}  │  A    B    C   BC")
    print("  " + "─"*95)
    fits = []
    all_evals = []
    for i, ind in enumerate(pop):
        f, m, ts = _evaluate(ind, df_train)
        fits.append(f)
        all_evals.append([*ind, f])
        print(f"  [{i+1:3d}]  "
              f"SL={ind[0]*100:5.2f}%  SP={ind[1]*100:5.2f}%  R={int(ind[2]):4d}  │  "
              f"MDD={m['max_drawdown']*100:6.1f}%  "
              f"Ann={m['annual_return']*100:6.1f}%  "
              f"Std={m['annual_vol']*100:6.1f}%  "
              f"SR={m['sharpe']:5.2f}  "
              f"WinR={ts['trade_win_rate']*100:5.1f}%  "
              f"TO={ts['annual_turnover']:5.1f}x  │  "
              f"A={ts['pct_A']*100:4.0f}%  "
              f"B={ts['pct_B']*100:3.0f}%  "
              f"C={ts['pct_C']*100:3.0f}%  "
              f"BC={ts['pct_BC']*100:2.0f}%")

    print(f"\n{'='*60}")
    print(f"GA: pop={GA_POP_SIZE}  gens={GA_GENS}  "
          f"cx={GA_CX_PROB}  mut={GA_MUT_PROB}  elite={GA_ELITE}")
    print(f"Fitness = Ann - |MDD|  (越大越好)\n")

    best_ever_fit = max(fits)
    best_ever_ind = pop[fits.index(best_ever_fit)][:]
    best_ever_m, best_ever_ts = None, None   # 首次在 gen=0 時填入

    best_history = []
    mean_history = []

    for gen in range(GA_GENS):
        # ── 精英保留 ──────────────────────────────────────────────────────────
        elite_idx  = sorted(range(len(pop)),
                            key=lambda i: fits[i], reverse=True)[:GA_ELITE]
        new_pop    = [pop[i][:] for i in elite_idx]
        new_fits   = [fits[i]   for i in elite_idx]

        # ── 產生子代 ──────────────────────────────────────────────────────────
        while len(new_pop) < GA_POP_SIZE:
            p1 = _tournament(pop, fits, GA_TOURN_SIZE, rng)
            p2 = _tournament(pop, fits, GA_TOURN_SIZE, rng)

            if rng.random() < GA_CX_PROB:
                c1, c2 = _blx_crossover(p1, p2, alpha=0.3, rng=rng)
            else:
                c1, c2 = p1[:], p2[:]

            for child in [c1, c2]:
                if rng.random() < GA_MUT_PROB:
                    child[:] = _mutate(child, rng)
                child[:] = _clip(child)
                f, m_c, ts_c = _evaluate(child, df_train)
                new_pop.append(child)
                new_fits.append(f)
                all_evals.append([*child, f])
                print(f"  [{len(new_pop):3d}/{GA_POP_SIZE}]  "
                      f"SL={child[0]*100:5.2f}%  SP={child[1]*100:5.2f}%  R={int(child[2]):4d}  │  "
                      f"MDD={m_c['max_drawdown']*100:6.1f}%  "
                      f"Ann={m_c['annual_return']*100:6.1f}%  "
                      f"Std={m_c['annual_vol']*100:6.1f}%  "
                      f"SR={m_c['sharpe']:5.2f}  "
                      f"WinR={ts_c['trade_win_rate']*100:5.1f}%  "
                      f"TO={ts_c['annual_turnover']:5.1f}x  │  "
                      f"A={ts_c['pct_A']*100:4.0f}%  "
                      f"B={ts_c['pct_B']*100:3.0f}%  "
                      f"C={ts_c['pct_C']*100:3.0f}%  "
                      f"BC={ts_c['pct_BC']*100:2.0f}%",
                      end="\r")

        pop  = new_pop[:GA_POP_SIZE]
        fits = new_fits[:GA_POP_SIZE]

        gen_best = max(fits)
        gen_mean = float(np.mean(fits))
        best_history.append(gen_best)
        mean_history.append(gen_mean)

        if gen_best > best_ever_fit:
            best_ever_fit = gen_best
            best_ever_ind = pop[fits.index(gen_best)][:]
            # 重新 evaluate 取完整指標
            _, best_ever_m, best_ever_ts = _evaluate(best_ever_ind, df_train)
        else:
            # 若非新最佳則不需重算，用現有 best_ever_m/ts（初次需初始化）
            if gen == 0:
                _, best_ever_m, best_ever_ts = _evaluate(best_ever_ind, df_train)

        print(
            f"\n  ── Gen {gen+1:2d}/{GA_GENS} ──  "
            f"GenBest fitness={gen_best:.4f}  Mean fitness={gen_mean:.4f}\n"
            f"  BestEver: SL={best_ever_ind[0]*100:.2f}%  "
            f"SP={best_ever_ind[1]*100:.2f}%  R={int(best_ever_ind[2]):4d}  │  "
            f"MDD={best_ever_m['max_drawdown']*100:.1f}%  "
            f"Ann={best_ever_m['annual_return']*100:.1f}%  "
            f"Std={best_ever_m['annual_vol']*100:.1f}%  "
            f"SR={best_ever_m['sharpe']:.2f}  "
            f"WinR={best_ever_ts['trade_win_rate']*100:.1f}%  "
            f"TO={best_ever_ts['annual_turnover']:.1f}x  │  "
            f"A={best_ever_ts['pct_A']*100:.0f}%  "
            f"B={best_ever_ts['pct_B']*100:.0f}%  "
            f"C={best_ever_ts['pct_C']*100:.0f}%  "
            f"BC={best_ever_ts['pct_BC']*100:.0f}%"
        )

    # ── 最終結果 ─────────────────────────────────────────────────────────────
    best_sl   = best_ever_ind[0]
    best_sp   = best_ever_ind[1]
    best_rank = int(best_ever_ind[2])

    print(f"\n{'='*60}")
    print(f"★  GA Best: SL={best_sl*100:.2f}%  "
          f"SP={best_sp*100:.2f}%  Rank={best_rank}")

    pv_train, ts_train = run_backtest(df_train, best_sl, best_sp, best_rank)
    pv_test,  ts_test  = run_backtest(df_test,  best_sl, best_sp, best_rank)
    m_train = compute_metrics(pv_train)
    m_test  = compute_metrics(pv_test)

    for split, m, ts in [("Train", m_train, ts_train),
                          ("Test",  m_test,  ts_test)]:
        print(f"  {split:5s} → "
              f"Sharpe={m['sharpe']:.2f}  "
              f"Ann={m['annual_return']*100:.1f}%  "
              f"MDD={m['max_drawdown']*100:.1f}%  |  "
              f"Cost={ts['cost_pct']*100:.2f}%  "
              f"TO={ts['annual_turnover']:.2f}x  "
              f"WinR={ts['trade_win_rate']*100:.1f}%  "
              f"N={ts['n_trades']}  |  "
              f"A={ts['pct_A']*100:.0f}%  "
              f"B={ts['pct_B']*100:.0f}%  "
              f"C={ts['pct_C']*100:.0f}%  "
              f"BC={ts['pct_BC']*100:.0f}%")

    # ── 儲存所有評估記錄 ──────────────────────────────────────────────────────
    df_evals = pd.DataFrame(all_evals, columns=["sl", "sp", "rank", "train_mdd"])
    df_evals["sl_pct"]  = (df_evals["sl"]   * 100).round(2).astype(str) + "%"
    df_evals["sp_pct"]  = (df_evals["sp"]   * 100).round(2).astype(str) + "%"
    df_evals["mdd_pct"] = (df_evals["train_mdd"] * 100).round(2).astype(str) + "%"
    df_evals.to_csv(os.path.join(OUTPUT_DIR, "ga_all_evals.csv"), index=False)

    # ── 最終結果 summary ─────────────────────────────────────────────────────
    summary = {
        "best_sl": best_sl, "best_sp": best_sp, "best_rank": best_rank,
        **{f"train_{k}": v for k, v in m_train.items()},
        **{f"test_{k}":  v for k, v in m_test.items()},
        **{f"train_{k}": v for k, v in ts_train.items()},
        **{f"test_{k}":  v for k, v in ts_test.items()},
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(OUTPUT_DIR, "ga_best_result.csv"), index=False)

    # ── 圖表 ─────────────────────────────────────────────────────────────────
    plot_ga_convergence(best_history, mean_history)
    plot_ga_population(all_evals)
    plot_best_combo(best_sl, best_sp,
                    {"train": pv_train, "test": pv_test})
    plot_pnl(pv_train, best_sl, best_sp, best_rank, "train", m_train)
    plot_pnl(pv_test,  best_sl, best_sp, best_rank, "test",  m_test)

    print(f"\nAll outputs → {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()