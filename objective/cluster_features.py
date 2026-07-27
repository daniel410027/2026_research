"""
cluster_features.py
====================
股票相關係數分群特徵（RMT 去噪 + Ward 階層式分群），移植自
904_group/606_corr/final2.py（獨立研究專案），供 make_new.py 直接呼叫，
不再需要手動從外部專案複製 cluster_panel.parquet。

兩種分群，用途與方法都不同：

  - build_periodic_cluster_features()：504 天滾動視窗、每 126 個交易日
    重新估計一次，流動性篩選用「估計視窗結束日往前 60 個交易日的入選比例」
    平滑掉單日雜訊（與原 606_corr 方法相同）。產出 6 個訓練特徵
    （clu_rank_ret_1/5/10、clu_rel_vol_5/10/240），會進模型訓練。

  - build_daily_cluster_labels()：每個交易日都重新估計一次（無 lookback
    平滑，直接用當天流動性篩選結果決定分群宇宙），只產出 clu_id_daily，
    不進模型訓練，供 record_order.py 查詢「這檔股票今天屬於哪一群」，
    做持倉集中度風控用。

兩者共用同一組 RMT 去噪 + Ward 分群數學工具，差別只在「多久重新分群一次」
與「流動性篩選要不要做 lookback 平滑」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ★ 2026-07-27：原本是 `from daily_model.liquidity import ...`——objective/ 反過來
#   依賴 daily_model/，層次顛倒（objective 是被 daily_model 使用的底層套件），
#   也正是 2026_research 當年必須繞過 objective/__init__.py 的根因。
#   公式移進 objective/liquidity.py 後改為套件內相對 import，依賴方向擺正。
from .liquidity import compute_liquidity_mask

# ─── 共用數學工具（移植自 904_group/606_corr/final2.py，原樣搬移）──────

def _corr_matrix(sub: pd.DataFrame) -> np.ndarray:
    """log return 子樣本 → 對稱、夾擠後的相關矩陣。"""
    C = sub.corr().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


def _mp_lambda_plus(vals: np.ndarray, N: int, T: int) -> float:
    """Marchenko-Pastur 上界（市場 λ_max 扣除後的 σ² 近似）。"""
    sigma2 = max(1.0 - vals[-1] / N, 1e-6)
    return sigma2 * (1.0 + np.sqrt(N / T)) ** 2


def _rmt_group_corr(C: np.ndarray, T: int) -> np.ndarray:
    """RMT 壓雜訊 + 移除 market mode，回傳去噪後的群組相關矩陣 Cg。"""
    vals, vecs = np.linalg.eigh(C)
    lam_plus = _mp_lambda_plus(vals, C.shape[0], T)
    v = vals.copy()
    v[vals < lam_plus] = 0.0   # 雜訊歸零
    v[-1] = 0.0                # 移除 market mode
    Cg = (vecs * v) @ vecs.T
    d = np.sqrt(np.clip(np.diag(Cg), 1e-12, None))
    Cg = np.clip(Cg / np.outer(d, d), -1.0, 1.0)
    np.fill_diagonal(Cg, 1.0)
    return Cg


def _corr_to_dist(C: np.ndarray) -> np.ndarray:
    """Mantegna 距離 d = √(2(1−ρ))，數值對稱化。"""
    D = np.sqrt(np.clip(2.0 * (1.0 - C), 0.0, None))
    np.fill_diagonal(D, 0.0)
    return (D + D.T) / 2.0


def _ward_labels(D: np.ndarray, k: int) -> np.ndarray:
    Z = linkage(squareform(D, checks=False), method="ward")
    return fcluster(Z, t=k, criterion="maxclust")


# ─── 流動性篩選（重用 daily_model/liquidity.py，避免另一份公式副本）────

def _liquidity_pass_wide(
    df: pd.DataFrame, w1: float, w2: float, keep_ratio: float,
) -> pd.DataFrame:
    """回傳 wide bool DataFrame（index=年月日, columns=證券代碼）：
    True = 當日通過 formula3 流動性篩選。"""
    mask = compute_liquidity_mask(df, w1, w2, keep_ratio)
    passed = df.loc[mask, ["證券代碼", "年月日"]].copy()
    passed["_pass"] = True
    wide = passed.pivot_table(
        index="年月日", columns="證券代碼", values="_pass", aggfunc="first"
    )
    return wide.fillna(False)


def _liquidity_pass_rate(
    wide: pd.DataFrame, asof_date, candidate_ids, lookback: int,
) -> pd.Series:
    """candidate_ids 在 asof_date 往前 lookback 個交易日的入選比例。"""
    ids = list(candidate_ids)
    pos = wide.index.get_indexer([asof_date], method="ffill")[0]
    if pos < 0:
        return pd.Series(np.nan, index=ids)
    lo = max(0, pos - lookback + 1)
    cols = [c for c in ids if c in wide.columns]
    return wide.iloc[lo:pos + 1][cols].mean(axis=0).reindex(ids)


def _liquidity_eligible_periodic(
    wide: pd.DataFrame, asof_date, candidate_ids, lookback: int, threshold: float,
) -> list:
    """週期性 panel 用：lookback 期間入選比例 ≥ threshold 才算夠格入群。"""
    rate = _liquidity_pass_rate(wide, asof_date, candidate_ids, lookback)
    if rate.isna().all():
        return list(candidate_ids)
    return rate[rate >= threshold].index.tolist()


def _liquidity_eligible_daily(wide: pd.DataFrame, asof_date, candidate_ids) -> list:
    """每日重估用：直接用當天的流動性篩選結果，不做 lookback 平滑。"""
    if asof_date not in wide.index:
        return []
    today_pass = wide.loc[asof_date]
    return [c for c in candidate_ids if today_pass.get(c, False)]


# ─── 分群核心：單一視窗 → cluster labels ───────────────────────────────

def _cluster_window(
    ret: pd.DataFrame, s: int, e: int, k: int,
    min_window_coverage: float, min_periods: int, eligible_ids_fn,
) -> pd.Series | None:
    """視窗 [s, e) 的 RMT 去噪 + Ward(k) 分群。
    eligible_ids_fn(asof_date, candidate_ids) -> list 決定合格名單
    （週期性 panel 用 lookback 平滑、每日重估用當天直接篩選——注入不同函式）。
    資料不足（歷史太短或合格股數 < k+2）時回傳 None。
    回傳 pd.Series[證券代碼 -> cluster label]。"""
    sub = ret.iloc[s:e]
    if sub.shape[0] < min_periods:
        return None
    cov = sub.notna().mean()
    keep = cov[cov >= min_window_coverage].index.tolist()
    keep = eligible_ids_fn(ret.index[e - 1], keep)
    if len(keep) < k + 2:
        return None
    sub = sub[keep]
    C = _corr_matrix(sub)
    Cg = _rmt_group_corr(C, len(sub))
    D = _corr_to_dist(Cg)
    labels = _ward_labels(D, k)
    return pd.Series(labels, index=keep)


def _estimation_windows(n_dates: int, window: int, step: int) -> list[tuple[int, int, int, int]]:
    """回傳 [(s, e, eff_start, eff_end), ...]（位置索引；e/eff_end 不含）。
    視窗 [s, e)，標籤自 e 起生效到下一視窗結束（最後一期到資料尾）。"""
    out = []
    s = 0
    while s + window <= n_dates:
        e = s + window
        eff_end = min(e + step, n_dates)
        if e < n_dates:
            out.append((s, e, e, eff_end))
        s += step
    return out


# ─── 對外函式 1：週期性 6 個訓練特徵（504天窗/126天重估）────────────────

def build_periodic_cluster_features(
    df: pd.DataFrame,
    *,
    window: int = 504,
    step: int = 126,
    k: int = 18,
    min_window_coverage: float = 0.9,
    liq_w1: float,
    liq_w2: float,
    liq_keep_ratio: float,
    liq_lookback: int = 60,
    liq_pass_threshold: float = 0.4,
) -> pd.DataFrame:
    """回傳長表 [證券代碼, 年月日, clu_rank_ret_1, clu_rank_ret_5,
    clu_rank_ret_10, clu_rel_vol_5, clu_rel_vol_10, clu_rel_vol_240]。

    方法與 904_group/606_corr/final2.py 的 run_v5 完全相同（504 天滾動窗、
    每 126 個交易日重估一次、point-in-time 標籤自視窗結束次一交易日起生效、
    流動性篩選用 lookback 平滑），只精簡到只算這 6 個目前實際會用到的欄位。
    """
    out_cols = ["證券代碼", "年月日",
                "clu_rank_ret_1", "clu_rank_ret_5", "clu_rank_ret_10",
                "clu_rel_vol_5", "clu_rel_vol_10", "clu_rel_vol_240"]

    wide_px = df.pivot_table(index="年月日", columns="證券代碼",
                              values="adj_close", aggfunc="first").sort_index()
    ret = np.log(wide_px).diff()
    dates = ret.index

    liq_wide = _liquidity_pass_wide(df, liq_w1, liq_w2, liq_keep_ratio)
    liq_wide = liq_wide.reindex(index=dates, columns=wide_px.columns).fillna(False)

    def eligible_fn(asof_date, ids):
        return _liquidity_eligible_periodic(
            liq_wide, asof_date, ids, liq_lookback, liq_pass_threshold
        )

    wins = _estimation_windows(len(dates), window, step)
    if not wins:
        print("  ⚠ [週期性分群] 資料長度不足一個估計視窗，略過")
        return pd.DataFrame(columns=out_cols)

    mom_horizons = (1, 5, 10, 240)
    Rmap = {h: (ret if h == 1 else ret.rolling(h).sum()) for h in mom_horizons}
    Vmap = {h: ret.rolling(h).std() for h in mom_horizons if h >= 5}

    blocks = []
    for i, (s, e, fs, fe) in enumerate(wins, 1):
        lab = _cluster_window(ret, s, e, k, min_window_coverage, window, eligible_fn)
        if lab is None:
            print(f"  ▸ [週期性分群] 期 {i}: 合格股數不足，跳過")
            continue
        eff_dates = dates[fs:fe]
        ids = list(lab.index)

        def melt_one(W: pd.DataFrame, name: str) -> pd.DataFrame:
            w = W.loc[eff_dates, ids]
            m = w.reset_index().melt(id_vars="年月日", var_name="證券代碼", value_name=name)
            return m

        base = melt_one(Rmap[1], "ret_1")
        for h in (5, 10):
            base[f"ret_{h}"] = melt_one(Rmap[h], f"ret_{h}")[f"ret_{h}"]
        for h in (5, 10, 240):
            base[f"vol_{h}"] = melt_one(Vmap[h], f"vol_{h}")[f"vol_{h}"]
        base["_clu"] = base["證券代碼"].map(lab.to_dict())

        g = base.groupby(["年月日", "_clu"], sort=False)
        for h in (1, 5, 10):
            base[f"clu_rank_ret_{h}"] = g[f"ret_{h}"].rank(pct=True).round(4)
        for h in (5, 10, 240):
            base[f"clu_rel_vol_{h}"] = (base[f"vol_{h}"] / g[f"vol_{h}"].transform("mean")).round(4)

        print(f"  ▸ [週期性分群] 期 {i}: 生效 {eff_dates[0].date()}~{eff_dates[-1].date()}  "
              f"n={len(ids)}  k={int(lab.nunique())}")
        blocks.append(base[out_cols])

    if not blocks:
        return pd.DataFrame(columns=out_cols)
    return pd.concat(blocks, ignore_index=True)


# ─── 對外函式 2：每日重新分群，僅供集中度風控用 ─────────────────────────

def build_daily_cluster_labels(
    df: pd.DataFrame,
    *,
    window: int = 504,
    k: int = 18,
    min_window_coverage: float = 0.9,
    liq_w1: float,
    liq_w2: float,
    liq_keep_ratio: float,
) -> pd.DataFrame:
    """回傳長表 [證券代碼, 年月日, clu_id_daily]。

    每個交易日都重新估計一次：用『當天』流動性篩選通過的股票池（不做
    lookback 平滑）+ 該日往前最多 window 個交易日報酬，做 RMT 去噪 +
    Ward(k) 分群。無 lookahead——每天只用當天為止的歷史。
    只回傳 clu_id_daily，不做動量/波動衍生特徵（那些留給週期性 panel）。
    """
    out_cols = ["證券代碼", "年月日", "clu_id_daily"]

    wide_px = df.pivot_table(index="年月日", columns="證券代碼",
                              values="adj_close", aggfunc="first").sort_index()
    ret = np.log(wide_px).diff()
    dates = ret.index

    liq_wide = _liquidity_pass_wide(df, liq_w1, liq_w2, liq_keep_ratio)
    liq_wide = liq_wide.reindex(index=dates, columns=wide_px.columns).fillna(False)

    def eligible_fn(asof_date, ids):
        return _liquidity_eligible_daily(liq_wide, asof_date, ids)

    rows = []
    n_skipped = 0
    for pos in range(len(dates)):
        e = pos + 1
        s = max(0, e - window)
        lab = _cluster_window(ret, s, e, k, min_window_coverage, window, eligible_fn)
        if lab is None:
            n_skipped += 1
            continue
        rows.append(pd.DataFrame({
            "證券代碼":    lab.index,
            "年月日":      dates[pos],
            "clu_id_daily": lab.to_numpy(),
        }))

    print(f"  ▸ [每日分群] {len(rows)}/{len(dates)} 個交易日成功分群"
          f"（{n_skipped} 天因歷史不足或合格股數不足而略過，多為資料起始段）")

    if not rows:
        return pd.DataFrame(columns=out_cols)
    return pd.concat(rows, ignore_index=True)
