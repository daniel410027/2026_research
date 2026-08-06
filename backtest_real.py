"""
backtest_real.py
================
**可交易性回測**：把 `2026_daily` 每天實際下單的那套規則，原封不動搬到歷史預測值上重跑。

回答的問題是 §8-#13：`backtest_0709.py` 的 D1 年化 219.5% / Sharpe 3.86 是
「每日重排十分位、零成本、無任何濾網」算出來的，**那是 daily 不會下的單**。
這支則是逐日維護一份最多 10 檔的持倉清單，換股走 daily 的 prob hysteresis，
新買入要過 daily 的每一道濾網，並計入手續費與證交稅。

─────────────────────────────────────────────
  與 backtest_0709.py 的差異（為什麼要有這支）
─────────────────────────────────────────────
                     backtest_0709.py            backtest_real.py
  部位               每日重排 D1（約 20 檔）     逐日維護的 10 檔清單
  換股               每日全換                    prob hysteresis（2×std 才動）
  新買入限制         無                          6 道濾網（見下）
  成本               COST_BPS = 0                買 0.1425% / 賣 0.4425%
  持倉不滿           不會發生                    會，缺額留現金隔日回補
  輸出               output/backtest/            output/backtest_real/

  兩邊互不 import 對方的設定，`backtest_0709.py` 不受本檔影響（只借用它的
  載入 / 指標 / 字型工具函式，見下方 import）。

─────────────────────────────────────────────
  複製自 daily 的規則（來源逐條標註）
─────────────────────────────────────────────
  [換股] `order_recorder/recorder.py::_run_update_scenario`
    · 挑戰者的 pred_score 需高出「最差衛冕者」超過 PROB_STD_MULT × std 才換股；
      std 以當日 in_filter=1 宇宙的 pred_score 母體標準差（ddof=0）計算。
    · 衛冕者由差到好、挑戰者由好到差逐組比對，一組不滿足就停（後面只會更不滿足）。
    · 掉出當日宇宙（查無預測值）的持倉 prob 視為 -inf → 排在最前面優先淘汰。

  [強制賣出] 同上 + `trading_config.find_stale_codes`
    · 行情凍結（近 STALE_WINDOW 個交易日報價逐字不變 = 已下市 / 長期停牌被 ffill）
      的持倉不走 prob 門檻，無條件出場。daily 的 6806 案例：終止上市後行情被複製，
      chg 恆為 0、成交金額停在最後一筆真實值，原有濾網全數放行。
    · ⚠ 本 repo 的 `database_make/` 只有 close / amount，沒有成交股數，
      所以判定用兩欄（daily 是三欄）。條件較寬 → 可能多抓，不會漏抓。

  [新買入濾網] `recorder.py::select_additions` + `_pick_with_constraints`
    1. in_filter = 1        本 repo 天生成立（predictions.csv 只含篩選後宇宙）
    2. 5MA 成交金額 > 1 億   MIN_AMOUNT_TO_BUY；NaN 視為不合格
    3. 當日漲幅 <= 7%        MAX_CHG_TO_BUY（2026-08-05 由 5% 放寬）；chg 為 NaN
                            （無前一交易日收盤）→ 剔除
    4. 非行情凍結
    5. pred_score >= 0       模型預期報酬為負就不買，寧可留現金（daily 2026-08-01 起）
    6. 估得出張數            以漲停價 × 手續費估整張成本，1 張即超 BUY_MAX_AMOUNT → 跳過
    7. 同群集中度 <= 2       同一 clu_id_daily 已持有 2 檔就不買第 3 檔，往下遞補
    不合格就往下遞補次佳排名；整池湊不滿 TARGET_N 就留缺額，隔日回補。

  [沒有複製的東西]
    · **處置股**（daily `stock_list/disposition.py`）：daily 只有
      `disposition_codes.csv` 的當日快照（43 列，資料日期 2026-08-05），
      沒有歷史處置清單，無法重建。→ 本回測會買到當年的處置股，
      實單則是「處置二以上跳過、處置一拆單」。這是**已知的樂觀偏差**。
    · **個人持倉排除**（mystock.csv）：不適用於回測。
    · **實際金額帳本**：本檔採等權（見下），不模擬 1000 萬資金上限與整張金額分檔。

─────────────────────────────────────────────
  部位與報酬定義
─────────────────────────────────────────────
  · 每檔權重固定 1/TARGET_N（= 1/10）。持倉不滿 10 檔時，**空位就是現金、報酬 0**，
    不會把權重攤到剩下的股票上——這是刻意保留 daily「寧缺勿追高 / 缺額隔日回補」
    的實際結果：濾網擋掉的部位不會偷偷變成加碼。
  · `return` 欄 = open[T+1] → open[T+2]，與 backtest_0709 同一定義。
    第 T 天收盤後決定的持倉，在 T+1 開盤成交、吃的就是第 T 列的 return。
  · 成本在部位變動的那個開盤點收取，記在當日：
        cost_T = (新買檔數 × BUY_COST_RATE + 賣出檔數 × SELL_COST_RATE) / TARGET_N
    買 0.1425%（手續費全額，未打折）、賣 0.4425%（手續費 0.1425% + 證交稅 0.3%）。
    ⚠ 行情凍結的強制賣出照收賣出成本。實務上這種單多半送不出去（見 daily 註解），
      但「賣不掉」的替代假設是部位永遠掛著，那更失真。檔數極少，影響可忽略。
  · 報酬 NaN（該日該股無行情）視為 0 並計數，印在報表尾。

─────────────────────────────────────────────
  資料流向
─────────────────────────────────────────────
  [輸入]
    database/experiment/*/predictions.csv   證券代碼 / 年月日 / return / y_true / y_pred
                                            （只含 in_filter=1 宇宙，每日約 206 檔）
    database_make/YYYY.csv                  close / amount / clu_id_daily / return
                                            （持倉掉出宇宙後仍需價格與報酬，故讀全表；
                                             已驗證兩邊 return 逐筆相同）
    database_make/market_return_series.csv   大盤同窗報酬（fallback：掃 database_make/*.csv）

  [輸出] output/backtest_real/
    metrics.csv              淨值 / 毛值 / 大盤 / 宇宙等權 / D1(cost0) 五條序列的指標
    daily_returns.csv        每日毛報酬、成本、淨報酬、持倉數、買賣檔數、對照序列
    holdings.csv             每日持倉明細（long format：日期 / 代碼 / 預測值 / 報酬）
    trades.csv               每筆進出（日期 / 代碼 / 買賣 / 原因）
    filter_stats.csv         每日各濾網剔除的候選檔數 → 用來歸因「哪道濾網最貴」
    cumulative_returns.png   淨 / 毛 / D1(cost0) / 大盤 / 宇宙等權 累積曲線
    drawdown.png             淨值水下曲線
    annual_returns.png       逐年報酬（淨 vs 大盤 vs D1 cost0）
    holdings_turnover.png    持倉檔數與換手率時序（看濾網何時咬得最兇）
    run_config.json          本次跑的完整設定（可重現）

作者：Daniel Huang
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# 借用 backtest_0709 的工具函式（不改動該檔；它的 main() 有 __main__ guard，
# import 不會觸發回測）。共用是刻意的：載入邏輯（含 L2/L4 混用偵測）與指標
# 定義（含 2026-07-23 修好的標準 Sharpe）必須兩邊一致，否則數字不可比。
from backtest_0709 import (
    setup_cjk_font,
    load_all_predictions,
    load_market_return,
    calc_metrics,
    DATE_COL,
)


# ============================================================
#  設定
# ============================================================

_ROOT = Path(__file__).resolve().parent

EXPERIMENT_DIR    = _ROOT / "database" / "experiment"
DATABASE_MAKE_DIR = _ROOT / "database_make"
OUTPUT_DIR        = _ROOT / "output" / "backtest_real"

# ── 組合規則（對應 daily/stock_list/order_recorder/recorder_config.py）──────
TARGET_N          = 10       # 目標持倉檔數
PROB_STD_MULT     = 2.0      # 換股門檻：挑戰者需高出衛冕者 N × std(pred_score)
CLUSTER_CAP       = 2        # 同一 clu_id_daily 最多持有幾檔

# ── 新買入濾網（對應 daily/stock_list/trading_config.py）────────────────────
# ★★ 2026-08-05：MAX_CHG_TO_BUY 0.05 → 0.07，對齊 daily 線上（使用者決定）★★
#   來源與完整證據見 ~/Desktop/2026_tune/summary_tune.md §9.2–§9.5，以及
#   daily stock_list/trading_config.py 的註解（含保留意見與回滾方式）。
#   摘要：逐年勝 7/9，排除 2021 仍 +4.0pp (p_boot=0.068)；換手率幾乎不變
#   （不是靠多交易換來的）。⚠ 未達傳統顯著水準、滑價未模擬、只在一份
#   predictions 源上驗過。
#
#   ★ 2026-08-06 更正效果幅度（~/Desktop/1111_monthholding §11）★
#     這裡原本寫的是 sharpe 1.5023→1.6301、年化 60.4%→68.5%、
#     excess_mdd −33.4%→−29.8%——那是**單一抽樣**（一份 predictions、一個 seed）。
#     只換 LightGBM 的 seed 重訓，年化本身就會晃 23.6pp：單一抽樣的 Δ 把共模的
#     seed 噪音算進了效果，系統性高估。改用「同一 seed 內只改這道濾網」的配對
#     設計（6 抽樣、共模噪音抵銷）後：
#         幾何年化   +2.96pp   5/6 正向   符號檢定 p=0.109
#         sharpe     +0.045    5/6 正向   p=0.109
#     ⟹ 引用一律用 +2.96pp / +0.045，舊數字高估約 2.7 倍。
#     ⟹ 方向仍成立，**不需要回滾 0.07**。
#     ❌ **「同時降低回撤」已撤回**：配對複驗下 MDD 只有 4/6 為正、p=0.344
#        （有兩個抽樣反而更差）。年化與 sharpe 的方向可信，MDD 不可信。
#     ⚠ 1111 用的是 nz5 predictions，與本 repo 不同源 → 絕對數字不可對照，
#       只能看配對差的方向。
#   ⚠ 不要再往上放：0.10 等於實質關閉（台股上限就是 10%），其好處 100% 來自
#     2021–2023，2024–2026 為 0。
#   ⚠ **此值一改，先前所有以 0.05 跑出的 backtest_real 數字都不再同基準可比**
#     （含 2026_tune §9.1/§9.2 的表）。要沿用舊數字就顯式設回 0.05。
MAX_CHG_TO_BUY    = 0.07         # 當日漲幅上限（追高 / 漲停買不到）
MIN_AMOUNT_TO_BUY = 100_000_000  # 5MA 成交金額下限（元）＝公司版一億
AMOUNT_MA_DAYS    = 5            # 成交金額移動平均天數
STALE_WINDOW      = 5            # 行情凍結判定視窗（交易日）
MIN_PRED_SCORE    = 0.0          # 新買入的預測值下限（負期望值不買）

# ── 張數估算（對應 daily/order_recorder/sizing.py::calc_units）─────────────
#   只用來判斷「這檔估不估得出張數」——估不出就跳過、往下遞補。
#   等權模式下不影響權重，但會影響「哪些股票買得到」，所以照抄公式。
FEE_RATE          = 1.001425     # 手續費率（估價用）
LIMIT_UP_MULT     = 1.1          # 以漲停價估最壞成交價
BUY_MAX_AMOUNT    = 1_600_000    # 單檔上限；1 張即超過 → units=0 → 跳過

# ── 交易成本（實際收取）────────────────────────────────────────────────────
BUY_COST_RATE     = 0.001425     # 買入：手續費 0.1425%（全額，未打折）
SELL_COST_RATE    = 0.004425     # 賣出：手續費 0.1425% + 證交稅 0.3%

# ── 濾網開關（做歸因用：關掉某道看它值多少）────────────────────────────────
ENABLE_AMOUNT_FILTER   = True
ENABLE_CHG_FILTER      = True
ENABLE_STALE_FILTER    = True
ENABLE_SCORE_FILTER    = True
ENABLE_UNITS_FILTER    = True
ENABLE_CLUSTER_FILTER  = True
ENABLE_HYSTERESIS      = True    # 關掉 → 每日重排前 10 名（成本會暴增）

ANNUAL_DAYS       = 252
MKT_COL           = "market_return_fwd"
D1_TOP_FRAC       = 0.1          # 對照組 D1（cost0）取宇宙前 10%

NEG_INF = -np.inf


# ============================================================
#  面板資料（wide matrices）
# ============================================================

class Panel:
    """把回測需要的所有欄位攤成 (日期 × 代碼) 的 numpy 矩陣，讓每日模擬是純陣列運算。

    只保留「曾經出現在 predictions.csv 裡的代碼」——持倉必定來自某天的宇宙，
    所以這個集合對持倉與候選都夠用，不必扛整個市場（約 1/2 的欄位量）。
    """

    def __init__(self, dates: list[pd.Timestamp], codes: list[str]):
        self.dates = dates
        self.codes = codes
        self.date_pos = {d: i for i, d in enumerate(dates)}
        self.code_pos = {c: j for j, c in enumerate(codes)}

    @property
    def shape(self):
        return (len(self.dates), len(self.codes))


def _pivot(df: pd.DataFrame, value_col: str, dates, codes) -> np.ndarray:
    """long → wide，缺格為 NaN。"""
    wide = (
        df.pivot_table(index=DATE_COL, columns="證券代碼", values=value_col, aggfunc="first")
        .reindex(index=dates, columns=codes)
    )
    return wide.to_numpy(dtype=float)


def load_panel(preds: pd.DataFrame, db_make_dir: Path) -> tuple[Panel, dict]:
    """讀 database_make/*.csv，組出模擬所需的所有矩陣。

    ⚠ 5MA 成交金額與行情凍結都要看「回測第一天之前」的交易日，
      所以先在完整的行情日曆上算，最後才切到預測日。
    """
    codes = sorted(preds["證券代碼"].unique())
    code_set = set(codes)
    pred_dates = sorted(preds[DATE_COL].unique())
    first_year = pd.Timestamp(pred_dates[0]).year
    last_year  = pd.Timestamp(pred_dates[-1]).year

    frames = []
    # 往前多讀一年，確保第一天的 5MA / 凍結判定有足夠的前置交易日
    for year in range(first_year - 1, last_year + 1):
        path = db_make_dir / f"{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(
            path,
            usecols=["證券代碼", "年月日", "close", "amount", "return", "clu_id_daily"],
            dtype={"證券代碼": str},
            low_memory=False,
        )
        df["證券代碼"] = df["證券代碼"].astype(str).str.strip()
        df = df[df["證券代碼"].isin(code_set)]
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        frames.append(df)
        print(f"  讀取 {path}  {df.shape}")

    if not frames:
        raise FileNotFoundError(f"{db_make_dir} 下找不到可用的年度 CSV")

    raw = pd.concat(frames, ignore_index=True)
    all_dates = sorted(raw[DATE_COL].unique())

    close  = _pivot(raw, "close",        all_dates, codes)
    amount = _pivot(raw, "amount",       all_dates, codes)
    ret    = _pivot(raw, "return",       all_dates, codes)
    clu    = _pivot(raw, "clu_id_daily", all_dates, codes)

    # ── 當日漲幅：與前一個交易日的收盤價比。前一日無收盤 → NaN（daily：視為不合格）
    prev_close = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        chg = close / prev_close - 1.0
    chg[~np.isfinite(chg)] = np.nan

    # ── 5MA 成交金額：NaN 略過（與 daily 的 groupby.mean() 一致）
    amt_df = pd.DataFrame(amount)
    amt5 = amt_df.rolling(AMOUNT_MA_DAYS, min_periods=1).mean().to_numpy()

    # ── 行情凍結：近 STALE_WINDOW 個交易日 close 與 amount 逐字不變
    #    （daily 另看成交股數，本 repo 無此欄，見檔頭說明）
    same = np.zeros_like(close, dtype=bool)
    same[1:] = (close[1:] == close[:-1]) & (amount[1:] == amount[:-1])
    same_df = pd.DataFrame(same.astype(float))
    # 連續 STALE_WINDOW-1 次「與前一日相同」⟺ 視窗內 STALE_WINDOW 列逐字相同
    stale = (same_df.rolling(STALE_WINDOW - 1, min_periods=STALE_WINDOW - 1).min() == 1).to_numpy()
    stale &= np.isfinite(close)          # 全 NaN 的空窗不算凍結（會被其他濾網擋掉）

    # ── 估得出張數？（漲停價 × 手續費，1 張成本 <= 上限）
    lot_cost = close * LIMIT_UP_MULT * FEE_RATE * 1000
    units_ok = np.isfinite(lot_cost) & (close > 0) & (lot_cost <= BUY_MAX_AMOUNT)

    # ── 切到預測日
    pos = {d: i for i, d in enumerate(all_dates)}
    missing = [d for d in pred_dates if d not in pos]
    if missing:
        raise ValueError(
            f"有 {len(missing)} 個預測日在 database_make 行情裡找不到（例：{missing[:3]}）。"
            f"多半是 database_make/ 與 database/experiment/ 不同批，請確認資料版本。"
        )
    rows = [pos[d] for d in pred_dates]
    panel = Panel([pd.Timestamp(d) for d in pred_dates], codes)

    arrays = {
        "ret":      ret[rows],
        "chg":      chg[rows],
        "amt5":     amt5[rows],
        "stale":    stale[rows],
        "units_ok": units_ok[rows],
        "clu":      clu[rows],
        "close":    close[rows],
    }

    # ── 預測值矩陣：不在當日宇宙者為 NaN（daily 的 _resolve_prob 回 -inf）
    pred_wide = (
        preds.pivot_table(index=DATE_COL, columns="證券代碼", values="y_pred", aggfunc="first")
        .reindex(index=panel.dates, columns=codes)
    )
    arrays["pred"] = pred_wide.to_numpy(dtype=float)

    print(f"\n  面板：{panel.shape[0]} 個交易日 × {panel.shape[1]} 檔  "
          f"（{panel.dates[0].date()} ~ {panel.dates[-1].date()}）")
    print(f"  行情凍結格數：{int(arrays['stale'].sum()):,}  "
          f"（其中在當日宇宙內：{int((arrays['stale'] & np.isfinite(arrays['pred'])).sum()):,}）")
    return panel, arrays


# ============================================================
#  每日模擬
# ============================================================

def simulate(panel: Panel, A: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """逐日跑 daily 的下單規則，回傳 (每日彙總, 持倉明細, 交易明細, 濾網統計)。"""
    pred, ret, chg, amt5 = A["pred"], A["ret"], A["chg"], A["amt5"]
    stale, units_ok, clu = A["stale"], A["units_ok"], A["clu"]

    n_days, _ = panel.shape
    codes = np.array(panel.codes)

    held: list[int] = []          # 目前持倉（欄位索引）
    daily_rows, holding_rows, trade_rows, filter_rows = [], [], [], []
    nan_ret_count = 0

    for i in range(n_days):
        date = panel.dates[i]
        p = pred[i]
        in_universe = np.isfinite(p)

        # ── 合格候選（daily::select_additions 的硬性濾網）─────────────────
        elig = in_universe.copy()
        rej = {}
        if ENABLE_AMOUNT_FILTER:
            ok = np.nan_to_num(amt5[i], nan=0.0) > MIN_AMOUNT_TO_BUY
            rej["amount"] = int((elig & ~ok).sum()); elig &= ok
        if ENABLE_CHG_FILTER:
            ok = np.isfinite(chg[i]) & (chg[i] <= MAX_CHG_TO_BUY)
            rej["chg"] = int((elig & ~ok).sum()); elig &= ok
        if ENABLE_STALE_FILTER:
            ok = ~stale[i]
            rej["stale"] = int((elig & ~ok).sum()); elig &= ok

        held_set = set(held)
        cand = [c for c in np.flatnonzero(elig) if c not in held_set]
        cand.sort(key=lambda c: -p[c])           # rank 順序 ＝ 預測值由高到低

        def prob(c: int) -> float:
            """查無預測（掉出當日宇宙）→ -inf，優先淘汰（daily::_resolve_prob）。"""
            return p[c] if np.isfinite(p[c]) else NEG_INF

        # ── 強制賣出：行情凍結的持倉（不走 prob 門檻）───────────────────
        stale_held = [c for c in held if ENABLE_STALE_FILTER and stale[i, c]]
        removed = list(stale_held)

        # ── 換股：prob hysteresis ────────────────────────────────────────
        universe_pred = p[in_universe]
        prob_std = float(np.std(universe_pred, ddof=0)) if universe_pred.size >= 2 else 0.0
        threshold = PROB_STD_MULT * prob_std

        if ENABLE_HYSTERESIS:
            incumbents = sorted((c for c in held if c not in set(stale_held)), key=prob)
            ci = 0
            for inc in incumbents:
                if ci >= len(cand):
                    break
                gap = p[cand[ci]] - prob(inc)
                if gap > threshold:
                    removed.append(inc)
                    trade_rows.append({"年月日": date, "證券代碼": codes[inc], "側": "賣",
                                       "原因": "換股", "pred": prob(inc), "gap": gap})
                    ci += 1
                else:
                    break
        else:
            # 對照模式：不做 hysteresis，持倉全數釋出、每日重挑前 TARGET_N 名
            for c in held:
                if c not in set(stale_held):
                    removed.append(c)
                    trade_rows.append({"年月日": date, "證券代碼": codes[c], "側": "賣",
                                       "原因": "每日重排", "pred": prob(c), "gap": np.nan})
            cand = [c for c in np.flatnonzero(elig)]
            cand.sort(key=lambda c: -p[c])

        for c in stale_held:
            trade_rows.append({"年月日": date, "證券代碼": codes[c], "側": "賣",
                               "原因": "行情凍結強制賣出", "pred": prob(c), "gap": np.nan})

        removed_set = set(removed)
        kept = [c for c in held if c not in removed_set]

        # ── 回補：目標 TARGET_N，含歷史缺額（非僅當日移除數）──────────────
        fill_count = max(0, TARGET_N - len(kept))
        pool = [c for c in cand if c not in removed_set]

        cluster_counts: dict = {}
        if ENABLE_CLUSTER_FILTER:
            for c in kept:
                cid = clu[i, c]
                if np.isfinite(cid):
                    cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

        picked: list[int] = []
        rej_score = rej_units = rej_cluster = 0
        for c in pool:
            if len(picked) >= fill_count:
                break
            if ENABLE_SCORE_FILTER and not (p[c] >= MIN_PRED_SCORE):
                rej_score += 1
                continue
            if ENABLE_UNITS_FILTER and not units_ok[i, c]:
                rej_units += 1
                continue
            cid = clu[i, c]
            if ENABLE_CLUSTER_FILTER and np.isfinite(cid) and cluster_counts.get(cid, 0) >= CLUSTER_CAP:
                rej_cluster += 1
                continue
            picked.append(c)
            if np.isfinite(cid):
                cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
            trade_rows.append({"年月日": date, "證券代碼": codes[c], "側": "買",
                               "原因": "回補", "pred": p[c], "gap": np.nan})

        rej["score"]   = rej_score
        rej["units"]   = rej_units
        rej["cluster"] = rej_cluster
        rej["年月日"]   = date
        rej["候選池"]   = len(pool)
        rej["缺額"]     = fill_count
        rej["實補"]     = len(picked)
        filter_rows.append(rej)

        held = kept + picked

        # ── 當日報酬（等權 1/TARGET_N，空位為現金）────────────────────────
        if held:
            r = ret[i, held]
            nan_ret_count += int(np.isnan(r).sum())
            gross = float(np.nansum(r)) / TARGET_N
        else:
            gross = 0.0

        cost = (len(picked) * BUY_COST_RATE + len(removed) * SELL_COST_RATE) / TARGET_N

        daily_rows.append({
            DATE_COL:      date,
            "gross":       gross,
            "cost":        cost,
            "net":         gross - cost,
            "n_holdings":  len(held),
            "n_buy":       len(picked),
            "n_sell":      len(removed),
            "n_universe":  int(in_universe.sum()),
            "n_eligible":  int(elig.sum()),
            "prob_std":    prob_std,
        })

        for c in held:
            holding_rows.append({DATE_COL: date, "證券代碼": codes[c],
                                 "y_pred": prob(c), "return": ret[i, c]})

    if nan_ret_count:
        print(f"  [WARN] 持倉報酬為 NaN 共 {nan_ret_count:,} 格（該股該日無行情），已以 0 計。")

    daily = pd.DataFrame(daily_rows).set_index(DATE_COL).sort_index()
    filters = pd.DataFrame(filter_rows)
    filters = filters[[DATE_COL, "候選池", "缺額", "實補",
                       *[c for c in ("amount", "chg", "stale", "score", "units", "cluster")
                         if c in filters.columns]]]
    return (daily,
            pd.DataFrame(holding_rows),
            pd.DataFrame(trade_rows),
            filters)


# ============================================================
#  對照序列
# ============================================================

def reference_series(preds: pd.DataFrame) -> pd.DataFrame:
    """算兩條零成本對照：宇宙等權，以及 backtest_0709 口徑的 D1（前 10%）。

    D1 用來量「可交易性約束吃掉多少」：同一份預測值，
    一邊是每日重排的紙上組合，一邊是 daily 真的會下的單。
    """
    df = preds.copy()
    df["_rank"] = df.groupby(DATE_COL)["y_pred"].rank(method="first", ascending=False)
    df["_size"] = df.groupby(DATE_COL)["y_pred"].transform("count")
    top = df[df["_rank"] <= np.ceil(df["_size"] * D1_TOP_FRAC)]
    out = pd.DataFrame({
        "universe_ew": df.groupby(DATE_COL)["return"].mean(),
        "D1_cost0":    top.groupby(DATE_COL)["return"].mean(),
    }).sort_index()
    return out


# ============================================================
#  圖表
# ============================================================

_SERIES_STYLE = {
    "net":         ("實單淨值（含成本）", "#c0392b", 2.2),
    "gross":       ("實單毛值（不含成本）", "#e67e22", 1.4),
    "D1_cost0":    ("D1 每日重排（cost0，紙上）", "#7f8c8d", 1.4),
    "market":      ("大盤", "#2c3e50", 1.6),
    "universe_ew": ("宇宙等權", "#16a085", 1.2),
}


def plot_cumulative(ret_df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(13, 7))
    for col, (label, color, lw) in _SERIES_STYLE.items():
        if col not in ret_df.columns:
            continue
        cum = (1 + ret_df[col].fillna(0)).cumprod()
        ax.plot(cum.index, cum, label=f"{label}  (×{cum.iloc[-1]:.2f})",
                color=color, linewidth=lw,
                linestyle="--" if col in ("gross", "D1_cost0") else "-")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.4g}"))
    ax.set_title(f"可交易性回測：累積報酬（對數刻度）　持倉 {TARGET_N} 檔　"
                 f"買 {BUY_COST_RATE:.4%} / 賣 {SELL_COST_RATE:.4%}")
    ax.set_xlabel("日期"); ax.set_ylabel("累積淨值（起始 = 1）")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_returns.png", dpi=140)
    plt.close(fig)


def plot_drawdown(ret_df: pd.DataFrame, output_dir: Path):
    cum = (1 + ret_df["net"].fillna(0)).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(dd.index, dd * 100, 0, color="#c0392b", alpha=0.35)
    ax.plot(dd.index, dd * 100, color="#c0392b", linewidth=1.0)
    ax.set_title(f"實單淨值水下曲線　MDD = {dd.min():.2%}")
    ax.set_xlabel("日期"); ax.set_ylabel("回撤 (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "drawdown.png", dpi=140)
    plt.close(fig)


def plot_annual(ret_df: pd.DataFrame, output_dir: Path):
    cols = [c for c in ("net", "D1_cost0", "market") if c in ret_df.columns]
    annual = (ret_df[cols].fillna(0) + 1).groupby(ret_df.index.year).prod() - 1
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(annual)); width = 0.8 / len(cols)
    for k, col in enumerate(cols):
        label, color, _ = _SERIES_STYLE[col]
        ax.bar(x + k * width, annual[col] * 100, width, label=label, color=color, alpha=0.85)
    ax.set_xticks(x + width * (len(cols) - 1) / 2)
    ax.set_xticklabels(annual.index)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("逐年報酬：實單淨值 vs 紙上 D1 vs 大盤")
    ax.set_xlabel("年"); ax.set_ylabel("報酬 (%)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "annual_returns.png", dpi=140)
    plt.close(fig)


def plot_holdings_turnover(daily: pd.DataFrame, output_dir: Path):
    """持倉檔數與換手率——濾網咬得最兇的時候，這兩條會同時掉下去 / 跳上來。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax1.plot(daily.index, daily["n_holdings"], color="#2980b9", linewidth=0.9)
    ax1.axhline(TARGET_N, color="#7f8c8d", linestyle="--", linewidth=1.0,
                label=f"TARGET_N = {TARGET_N}")
    ax1.set_ylabel("持倉檔數")
    ax1.set_title(f"持倉檔數（平均 {daily['n_holdings'].mean():.2f}　"
                  f"未滿 {TARGET_N} 檔的日數佔比 {(daily['n_holdings'] < TARGET_N).mean():.1%}）")
    ax1.legend(); ax1.grid(alpha=0.3)

    turnover = (daily["n_buy"] / TARGET_N).rolling(20).mean()
    ax2.plot(daily.index, turnover * 100, color="#8e44ad", linewidth=0.9)
    ax2.set_ylabel("單邊換手率 (%)"); ax2.set_xlabel("日期")
    ax2.set_title(f"單邊換手率（20 日移動平均；全期日均 {(daily['n_buy'] / TARGET_N).mean():.1%}）")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "holdings_turnover.png", dpi=140)
    plt.close(fig)


# ============================================================
#  主流程
# ============================================================

def main():
    print("=" * 70)
    print("  backtest_real.py — 可交易性回測（複製 2026_daily 下單規則）")
    print("=" * 70)
    setup_cjk_font()
    print(f"  持倉檔數        : {TARGET_N}")
    print(f"  換股門檻        : {PROB_STD_MULT} × std(pred_score)"
          f"{'' if ENABLE_HYSTERESIS else '  ← 已關閉（每日重排）'}")
    print(f"  成本            : 買 {BUY_COST_RATE:.4%} / 賣 {SELL_COST_RATE:.4%}"
          f"（來回 {BUY_COST_RATE + SELL_COST_RATE:.4%}）")
    print(f"  濾網            : 金額>{MIN_AMOUNT_TO_BUY / 1e8:.0f}億={ENABLE_AMOUNT_FILTER} "
          f"漲幅<={MAX_CHG_TO_BUY:.0%}={ENABLE_CHG_FILTER} 凍結={ENABLE_STALE_FILTER} "
          f"pred>=0={ENABLE_SCORE_FILTER} 張數={ENABLE_UNITS_FILTER} "
          f"群集<={CLUSTER_CAP}={ENABLE_CLUSTER_FILTER}")
    print(f"  輸出            : {OUTPUT_DIR}")
    print("  ⚠ 處置股未模擬（daily 無歷史處置清單）→ 結果偏樂觀，見檔頭說明")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] 讀取預測值 ...")
    preds = load_all_predictions(EXPERIMENT_DIR)

    print("\n[2/5] 讀取行情面板 ...")
    panel, arrays = load_panel(preds, DATABASE_MAKE_DIR)

    print("\n[3/5] 逐日模擬下單 ...")
    daily, holdings, trades, filters = simulate(panel, arrays)

    print("\n[4/5] 對齊對照序列 ...")
    refs = reference_series(preds)
    market = load_market_return(DATABASE_MAKE_DIR)
    ret_df = daily.join(refs, how="left")
    ret_df["market"] = market.reindex(ret_df.index)
    n_missing = ret_df["market"].isna().sum()
    if n_missing:
        print(f"  [WARN] 大盤報酬缺 {n_missing} 天（不影響策略序列）")

    print("\n[5/5] 指標與輸出 ...")
    rows = []
    for col in ("net", "gross", "D1_cost0", "universe_ew", "market"):
        if col not in ret_df.columns:
            continue
        m = calc_metrics(ret_df[col], ret_df["market"] if col != "market" else None,
                         annual_days=ANNUAL_DAYS)
        m["series"] = _SERIES_STYLE[col][0]
        m["key"] = col
        rows.append(m)
    metrics = pd.DataFrame(rows).set_index("key")
    metrics = metrics[["series", "ann_return", "ann_std", "sharpe", "mdd",
                       "calmar", "hit_rate", "alpha", "IR", "n_days"]]

    turnover_daily = (daily["n_buy"] / TARGET_N).mean()
    cost_drag = daily["cost"].sum() * ANNUAL_DAYS / len(daily)
    summary = {
        "平均持倉檔數":        round(float(daily["n_holdings"].mean()), 2),
        "持倉未滿日數佔比":    round(float((daily["n_holdings"] < TARGET_N).mean()), 4),
        "日均單邊換手率":      round(float(turnover_daily), 4),
        "年化成本拖累(算術)":  round(float(cost_drag), 4),
        "總買入筆數":          int(daily["n_buy"].sum()),
        "總賣出筆數":          int(daily["n_sell"].sum()),
        "強制賣出筆數":        int((trades["原因"] == "行情凍結強制賣出").sum()) if len(trades) else 0,
        "回測天數":            int(len(daily)),
    }

    ret_df.to_csv(OUTPUT_DIR / "daily_returns.csv", encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", encoding="utf-8-sig")
    holdings.to_csv(OUTPUT_DIR / "holdings.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(OUTPUT_DIR / "trades.csv", index=False, encoding="utf-8-sig")
    filters.to_csv(OUTPUT_DIR / "filter_stats.csv", index=False, encoding="utf-8-sig")

    run_config = {
        "TARGET_N": TARGET_N, "PROB_STD_MULT": PROB_STD_MULT, "CLUSTER_CAP": CLUSTER_CAP,
        "MAX_CHG_TO_BUY": MAX_CHG_TO_BUY, "MIN_AMOUNT_TO_BUY": MIN_AMOUNT_TO_BUY,
        "AMOUNT_MA_DAYS": AMOUNT_MA_DAYS, "STALE_WINDOW": STALE_WINDOW,
        "MIN_PRED_SCORE": MIN_PRED_SCORE, "BUY_MAX_AMOUNT": BUY_MAX_AMOUNT,
        "BUY_COST_RATE": BUY_COST_RATE, "SELL_COST_RATE": SELL_COST_RATE,
        "filters": {
            "amount": ENABLE_AMOUNT_FILTER, "chg": ENABLE_CHG_FILTER,
            "stale": ENABLE_STALE_FILTER, "score": ENABLE_SCORE_FILTER,
            "units": ENABLE_UNITS_FILTER, "cluster": ENABLE_CLUSTER_FILTER,
            "hysteresis": ENABLE_HYSTERESIS,
        },
        "date_range": [str(panel.dates[0].date()), str(panel.dates[-1].date())],
        "summary": summary,
        "未模擬": ["處置股（daily 無歷史處置清單）", "個人持倉排除", "資金上限與整張金額分檔"],
    }
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_cumulative(ret_df, OUTPUT_DIR)
    plot_drawdown(ret_df, OUTPUT_DIR)
    plot_annual(ret_df, OUTPUT_DIR)
    plot_holdings_turnover(daily, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("  績效指標")
    print("=" * 70)
    print(metrics.to_string())
    print("\n  組合統計")
    for k, v in summary.items():
        print(f"    {k:<18}{v}")
    print("\n  濾網剔除（日均候選檔數）")
    for col in ("amount", "chg", "stale", "score", "units", "cluster"):
        if col in filters.columns:
            print(f"    {col:<10}{filters[col].mean():.1f}")
    print(f"\n  輸出完成：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
