"""
main_fix.py
===========
Walk-Forward Pipeline 入口（靜態研究版）。

超參數模式（RunConfig.TUNE_MODE 切換，2026-07-22 改版）：
    ★ "every_fold"（預設）：每個 fold 都重跑 Optuna（N_TRIALS 次），
      各折超參數不同。逐折參數見 database/experiment/{window}/best_params.json。

    ★ "first_only"：只有第一個 fold 執行 Optuna，之後所有 fold 凍結沿用。
      tune 出的參數會另存 best_params_regression.json。

    ★ "periodic"：每 RETUNE_EVERY_N 個 fold 重新 tune 一次。

    ★ "frozen"：完全不使用 Optuna，直接套用 best_params.json（或內嵌 dict）
      的固定超參數，所有 window 共用同一組。

  ⚠ 舊版用 TUNE_FIRST_FOLD(bool) 表達，但字面意思與實際行為不符：
    它同時對 WalkForwardConfig 傳 freeze_hyperparams=False，而
    model._should_tune 的第一個分支會在檢查 _frozen_params 之前就短路 →
      - TUNE_FIRST_FOLD=True  實際上是「每個 fold 都 tune」（非「第一折後凍結」），
        存出的 best_params_regression.json 其實是最後一折的參數；
      - TUNE_FIRST_FOLD=False 的預植參數被完全忽略，Optuna 照跑，
        best_params.json 被靜默丟棄，畫面卻印「Optuna 全程停用」。
    現行 TUNE_MODE="every_fold" 即舊版 TUNE_FIRST_FOLD=True 的實際行為，
    換寫法不改結果。詳見 summary_research.md 5.1。

    其餘行為各模式相同：
      - final model 仍用 train_final_lgb 預設 500 輪
      - OOF Spearman IC 仍由 collect_oof_prob 的 OOF 預測值計算（監控用，無 threshold）
      - 特徵集對齊（排除 amount 等）同 main.py

  超參數來源（擇一）：
    1. RunConfig.BEST_PARAMS_INLINE 不為 None → 用內嵌 dict
    2. 否則讀 RunConfig.BEST_PARAMS_PATH（預設 best_params.json）

─────────────────────────────────────────────
  ★ 2026-07-27：拆掉 importlib 載入 + monkeypatch（原 2026-06-15 兩節）
─────────────────────────────────────────────
  舊版做兩件互相綁死的事：
    (1) 用 importlib 直接載入 objective/model.py、繞過 objective/__init__.py
        （當年的理由是 __init__.py 內有 `from daily_model import ...`，
         而 daily_model 屬於 live 專案、不在本 repo）；
    (2) 在 trainer.run() 前 monkeypatch `objective.model._EXCLUDE_COLS`，
        把 amount / market_value / close 等欄排除於訓練之外。
  (2) 需要 (1) 保證「patch 到的模組就是 trainer 用的那份」，兩者是一組的。

  現在 (1) 的理由已不成立（__init__.py 只剩相對 import），而 (2) 本身是隱性的
  全域狀態改動：新入口腳本只要沒照抄整套載入順序，排除清單就靜默失效——與
  §6.1「宣稱凍結、實際照跑 Optuna」同一類 bug，且因為實驗是整包複製出去的
  （§0.1），每份副本都會繼承這個陷阱。

  改法：排除清單變成 WalkForwardConfig.extra_exclude_cols /
  extra_exclude_prefixes（見 objective/model.py），本檔改用一般 import。
  新寫實驗入口腳本時直接 `from objective.model import ...` 即可，不需要
  任何載入順序的儀式。

─────────────────────────────────────────────
  ★ 2026-06-23：新增 size/price 外部條件變數排除
─────────────────────────────────────────────
  為研究「市值(size) / 股價(price) × 模型訊號」對報酬的交互效果，
  make_new.py 已把 market_value、close 保留進 database_make/YYYY.csv。
  這兩欄屬【模型外部的 conditioning variable】，**不可進訓練**——
  否則模型直接學到 size/price，交互分析淪為循環論證（模型本就看了 size，
  再問它的 edge 是否隨 size 變化，訊號被自己吃掉）。
  故於 EXTRA_EXCLUDE_COLS 一併排除（與 amount 同性質：保留於 CSV、排除於訓練）。
  下游 interaction.py 以這兩欄作為 conditioning 軸。

─────────────────────────────────────────────
  ★ 2026-07-09：整合 805_group2 流動性篩選（formula3）
─────────────────────────────────────────────
  在 WalkForwardTrainer 讀取資料之前，先對全範圍
  [ML_WINDOW_START, ML_WINDOW_END] 套用 formula3_amount_mv_turnover
  篩選（w1=0.1004, w2=0.1896, keep_ratio=0.2000），train + predict
  一起篩（本支線純研究/回測，無 daily_model 的既有持倉問題，
  不需要區分 train-only vs. predict-all）。
  篩選後資料寫到 database_make_liq_filtered/{tag}/YYYY.csv（依參數
  tag 命名快取，非時間戳，重跑會自動重用），再把這個目錄餵給
  WalkForwardConfig.precomputed_dir，不改 model.py。
  同時寫一份 liquidity_filter_meta.json 到 database/experiment/，
  供 backtest.py 自動讀取並在報表上標示本次用了哪組篩選參數。

執行方式：
    python main_fix.py

作者：Daniel Huang
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

# ★ 2026-07-27：一般 import（原本繞過 __init__.py 的 importlib 機制已移除，
#   理由見檔頭）。本檔所在目錄即 repo 根目錄，直接執行時 sys.path 已包含它。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from objective.liquidity import compute_liquidity_mask, liq_tag   # noqa: E402
from objective.model           import WalkForwardConfig, WalkForwardTrainer  # noqa: E402
from objective.research_config import RunConfig                                # noqa: E402


# ============================================================
#  設定
# ============================================================
#  ★ 2026-07-27：RunConfig 已搬到 objective/research_config.py（參考
#    2026_daily 把設定放在 daily_model/model_config.py 的分層）。本檔只留
#    「怎麼跑」，不再留「跑什麼設定」。
#
#    要改設定的實驗**繼承後覆寫**，不要去改 research_config.py 本身：
#
#        from objective.research_config import RunConfig
#        class MyRunConfig(RunConfig):
#            LIQ_KEEP_RATIO = 0.15
#        cfg = MyRunConfig()
#
#    這樣本資料夾之後修好的預設值，複製出去的實驗（§0.1）會自動繼承，
#    不必靠人記得把新入口腳本的設定區塊抄一遍。
# ============================================================


# ============================================================
#  資料載入
# ============================================================

def load_precomputed(cfg: RunConfig) -> pd.DataFrame:
    """讀取 database_make/YYYY.csv（window_start ~ window_end）並合併。"""
    frames = []
    for year in range(cfg.ML_WINDOW_START, cfg.ML_WINDOW_END + 1):
        path = cfg.PRECOMPUTED_DIR / f"{year}.csv"
        if not path.exists():
            print(f"  ⚠ {path} 不存在，跳過")
            continue
        tmp = pd.read_csv(path, low_memory=False)
        frames.append(tmp)
        print(f"  讀取 {path.name}  {tmp.shape}")

    if not frames:
        raise FileNotFoundError(
            f"{cfg.PRECOMPUTED_DIR}/ 下找不到任何 CSV，"
            "請確認 database_make/ 已正確產出"
        )

    df = pd.concat(frames, ignore_index=True)
    df["年月日"]   = pd.to_datetime(df["年月日"])
    df["證券代碼"] = df["證券代碼"].astype(str)
    df = df.sort_values(["證券代碼", "年月日"]).reset_index(drop=True)

    print(f"\n  合併後 shape: {df.shape}")

    if cfg.LABEL not in df.columns:
        raise ValueError(
            f"Label '{cfg.LABEL}' 不存在於資料中，"
            "請確認 database_make/ 已正確產出 label"
        )
    return df


# ============================================================
#  ★ 流動性篩選（805_group2 formula3_amount_mv_turnover）
# ============================================================
#  ★ 2026-07-27：compute_liquidity_mask / liq_tag 已移到 objective/liquidity.py，
#    與 2026_daily 共用同一份（原本 daily 在 daily_model/liquidity.py、research
#    內嵌在本檔，是兩份各自維護的副本，且躲在入口腳本裡連 objective/ 的對拍
#    檢查都照不到）。公式改動請改那一支，兩個 repo 一起更新。
# ============================================================


# ─────────────────────────────────────────────────────────────
#  ★ 快取來源指紋（2026-07-27）
# ─────────────────────────────────────────────────────────────
#  問題（summary_research.md §2 / §5）：快取目錄只以參數 tag 命名，換過
#  database_make/（例如從 2026_daily 搬新特徵集過來）之後，舊快取不會失效，
#  會靜默沿用舊特徵訓練。這件事已經害過一次白跑 42 分鐘。
#
#  舊版只比 mtime，擋得住「原地覆蓋」，但擋不住 `cp -p` / rsync 保留時間戳、
#  或從備份還原（mtime 反而更舊）的情況。而真正咬人的失敗模式是**欄位集變了**
#  （macro 那次就是整組欄位不存在），mtime 對此毫無資訊。
#
#  改為記錄來源檔指紋到快取目錄的 _source.json：每年 (size, mtime_ns, 表頭雜湊)。
#  表頭雜湊直接抓到欄位集變動——讀一行就好，幾乎不花時間。
#  需要更嚴格時開 RunConfig.LIQ_CACHE_STRICT_HASH（改雜湊整份檔案內容；
#  database_make/ 約 3.2 GB，每次檢查多花數秒到數十秒的磁碟讀取）。
# ─────────────────────────────────────────────────────────────

_SOURCE_FP_FILE = "_source.json"


def _file_fingerprint(path: Path, strict: bool) -> dict:
    """單一來源 CSV 的指紋。strict=True 才雜湊整份檔案內容。"""
    st = path.stat()
    fp = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}

    with open(path, "rb") as f:
        if strict:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
            fp["content_sha256"] = h.hexdigest()[:16]
        else:
            # 只讀表頭那一行：欄位集變了就一定變（macro 欄位整組消失那次的
            # 失敗模式），而讀一行不需要碰整份 200 MB+ 的檔案。
            fp["header_sha256"] = hashlib.sha256(f.readline()).hexdigest()[:16]
    return fp


def _source_fingerprint(cfg: "RunConfig", years: list[int]) -> dict:
    """database_make/ 各年來源檔的指紋 + 本次篩選參數。"""
    return {
        "strict_hash": bool(cfg.LIQ_CACHE_STRICT_HASH),
        "source_dir":  str(cfg.PRECOMPUTED_DIR),
        "params": {
            "w1": cfg.LIQ_W1, "w2": cfg.LIQ_W2, "keep_ratio": cfg.LIQ_KEEP_RATIO,
        },
        "years": {
            str(y): _file_fingerprint(cfg.PRECOMPUTED_DIR / f"{y}.csv",
                                      cfg.LIQ_CACHE_STRICT_HASH)
            for y in years
            if (cfg.PRECOMPUTED_DIR / f"{y}.csv").exists()
        },
    }


def _liq_cache_stale_reason(cfg: "RunConfig", out_dir: Path, years: list[int]) -> str | None:
    """
    快取是否已過期。回傳過期原因字串；None = 可安全重用。

    ⚠ 沒有 _source.json 的舊快取一律視為過期並重算。這是刻意的保守選擇：
      「不確定是用哪份來源算的」與「確定是舊的」在研究上是同一件事——
      兩者都不能拿來當可比較的實驗上游。
    """
    fp_path = out_dir / _SOURCE_FP_FILE
    if not fp_path.exists():
        return f"快取缺少 {_SOURCE_FP_FILE}（2026-07-27 之前產生的舊快取，無法驗證來源）"

    try:
        with open(fp_path, encoding="utf-8") as f:
            cached_fp = json.load(f)
    except Exception as e:                                    # noqa: BLE001
        return f"{fp_path} 讀取失敗（{type(e).__name__}），視為過期"

    current = _source_fingerprint(cfg, years)

    if cached_fp.get("params") != current["params"]:
        return f"篩選參數不符（快取 {cached_fp.get('params')} vs 現在 {current['params']}）"

    # 快取是寬鬆模式算的、現在要求嚴格 → 沒有可比的欄位，重算
    if current["strict_hash"] and not cached_fp.get("strict_hash"):
        return "快取是非 strict 模式建立的，但本次要求 LIQ_CACHE_STRICT_HASH=True"

    key         = "content_sha256" if current["strict_hash"] else "header_sha256"
    cached_yrs  = cached_fp.get("years", {})
    current_yrs = current["years"]

    missing = sorted(set(current_yrs) - set(cached_yrs))
    if missing:
        return f"快取未涵蓋來源年份 {missing}"

    mtime_only = []
    for y, cur in current_yrs.items():
        old = cached_yrs.get(y, {})
        if old.get(key) != cur.get(key):
            return f"{cfg.PRECOMPUTED_DIR}/{y}.csv 的{'內容' if current['strict_hash'] else '欄位表頭'}已變更"
        if old.get("size") != cur.get("size"):
            return f"{cfg.PRECOMPUTED_DIR}/{y}.csv 檔案大小已變更（{old.get('size')} → {cur.get('size')}）"
        if old.get("mtime_ns") != cur.get("mtime_ns"):
            mtime_only.append(y)

    # 大小與表頭都相同、只有 mtime 變 → 幾乎必然是同一份資料被重新複製過來
    # （220 MB 的 CSV 要「重生成後大小與表頭完全一致但內容不同」機率極低）。
    # 這種情況重算沒有意義，但值得提醒：若是刻意換過資料，請開 strict 模式確認。
    if mtime_only:
        print(f"  ℹ 來源檔 {mtime_only} 的 mtime 變了，但大小與欄位表頭相同 → "
              f"視為同一份資料，沿用快取。"
              f"（若確定換過內容，設 RunConfig.LIQ_CACHE_STRICT_HASH=True 重驗）")

    return None


def build_liquidity_filtered_dir(cfg: "RunConfig") -> Path:
    """
    對 [ML_WINDOW_START, ML_WINDOW_END] 全範圍資料套用流動性篩選
    （train + predict 一起篩，本支線純研究/回測、無既有持倉概念），
    輸出成同樣的 per-year CSV 結構到 LIQ_FILTERED_ROOT/{tag}/，
    供 WalkForwardTrainer 直接當作 precomputed_dir 讀取。

    快取：若目錄已存在、涵蓋所需年份、且來源檔（database_make/）的指紋與
    建立快取時相同 → 直接重用，不重算。指紋不符（換過資料、欄位集變了、
    篩選參數不同）或快取沒有指紋檔 → 自動重算，避免靜默沿用舊特徵訓練。
    見上方 _liq_cache_stale_reason 的說明。
    """
    tag     = liq_tag(cfg.LIQ_W1, cfg.LIQ_W2, cfg.LIQ_KEEP_RATIO)
    out_dir = cfg.LIQ_FILTERED_ROOT / tag
    years   = list(range(cfg.ML_WINDOW_START, cfg.ML_WINDOW_END + 1))

    existing = [out_dir / f"{y}.csv" for y in years if (out_dir / f"{y}.csv").exists()]
    if out_dir.exists() and len(existing) == len(years):
        stale_reason = _liq_cache_stale_reason(cfg, out_dir, years)
        if stale_reason is None:
            print(f"  ✓ 快取命中，重用已篩選資料: {out_dir}/（{len(years)} 年）")
            return out_dir
        print(f"  ⚠ 快取已過期（{stale_reason}），重新計算流動性篩選 → {out_dir}/")
    else:
        print(f"  快取未命中，重新計算流動性篩選 → {out_dir}/")
    print(f"  w1(amount)={cfg.LIQ_W1}  w2(mv)={cfg.LIQ_W2}  "
          f"w3(turnover)={1 - cfg.LIQ_W1 - cfg.LIQ_W2:.4f}  "
          f"keep_ratio={cfg.LIQ_KEEP_RATIO}")

    df = load_precomputed(cfg)   # 讀原始 database_make/（含 amount, market_value）
    mask = compute_liquidity_mask(df, cfg.LIQ_W1, cfg.LIQ_W2, cfg.LIQ_KEEP_RATIO)
    n_before = len(df)
    df = df[mask].copy()
    n_after = len(df)

    kept_per_day = df.groupby("年月日").size()
    print(f"  篩選結果: {n_before:,} → {n_after:,} 列 ({n_after / n_before:.1%})  "
          f"每日平均保留 {kept_per_day.mean():.0f} 檔")

    out_dir.mkdir(parents=True, exist_ok=True)
    for year, part in df.groupby(df["年月日"].dt.year):
        part = part.sort_values(["證券代碼", "年月日"])
        part.to_csv(out_dir / f"{year}.csv", index=False, encoding="utf-8-sig")
        print(f"    ✓ {out_dir}/{year}.csv  {part.shape}")

    meta = {
        "formula":       "formula3_amount_mv_turnover",
        "w1_amount":     cfg.LIQ_W1,
        "w2_market_value": cfg.LIQ_W2,
        "w3_turnover":   round(1 - cfg.LIQ_W1 - cfg.LIQ_W2, 4),
        "keep_ratio":    cfg.LIQ_KEEP_RATIO,
        "tag":           tag,
        "rows_before":   n_before,
        "rows_after":    n_after,
        "window_start":  cfg.ML_WINDOW_START,
        "window_end":    cfg.ML_WINDOW_END,
        "source":        "805_group2 summary.md（f3_w101004_w201896_kr01520）",
    }
    with open(out_dir / "liquidity_filter_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ★ 2026-07-27：來源指紋。必須在資料寫完之後才寫，否則中途失敗會留下
    #   「指紋說是新的、內容其實不完整」的快取——比沒有快取更糟。
    with open(out_dir / _SOURCE_FP_FILE, "w", encoding="utf-8") as f:
        json.dump(_source_fingerprint(cfg, years), f, ensure_ascii=False, indent=2)

    # ★ 同步複製一份到 database/experiment/（backtest.py 預設 EXPERIMENT_DIR 的
    #   上層），供 backtest.py 自動偵測本次訓練用了哪組篩選參數，避免回測時
    #   誤以為是全市場訓練的結果。
    exp_meta_dir = Path("database/experiment")
    exp_meta_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_meta_dir / "liquidity_filter_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return out_dir


# ============================================================
#  ★ 固定超參數載入
# ============================================================

def load_best_params(cfg: RunConfig) -> dict:
    """
    取得固定超參數（不經 Optuna）。
    優先用 BEST_PARAMS_INLINE，否則讀 BEST_PARAMS_PATH（best_params.json）。
    回傳的 dict 不含 base params（objective/metric 等），
    由 lgb_utils 在 collect_oof_prob / train_final_lgb 內以 {**_LGB_BASE_PARAMS, **best_params} 合併。
    """
    if cfg.BEST_PARAMS_INLINE is not None:
        params = dict(cfg.BEST_PARAMS_INLINE)
        print("  超參數來源：RunConfig.BEST_PARAMS_INLINE（內嵌）")
    else:
        path = Path(cfg.BEST_PARAMS_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"找不到固定超參數檔：{path}\n"
                f"請放置 best_params.json，或改用 RunConfig.BEST_PARAMS_INLINE 內嵌。"
            )
        with open(path, encoding="utf-8") as f:
            params = json.load(f)
        print(f"  超參數來源：{path}")

    if not isinstance(params, dict) or not params:
        raise ValueError(f"固定超參數內容無效（需非空 dict）：{params}")

    for k, v in params.items():
        print(f"    {k:18s}: {v}")
    return params


# ============================================================
#  ★ 特徵集對齊（與 live/daily 一致）
# ============================================================

def resolve_extra_exclude_cols(cfg: RunConfig) -> frozenset[str]:
    """
    回傳要額外排除的欄位，交給 WalkForwardConfig.extra_exclude_cols。

    ★ 2026-07-27：取代舊的 align_features_with_live() —— 那支直接改
      `objective.model._EXCLUDE_COLS` 這個模組全域，需要入口腳本用 importlib
      控制載入順序才保證 patch 到對的模組（見檔頭）。現在排除清單跟著 config
      走，會一起被記進 run_manifest.json，事後查得到某份產出用的是哪一組。
    """
    if not cfg.ALIGN_FEATURES_WITH_LIVE:
        print("  ▸ ALIGN_FEATURES_WITH_LIVE=False，只用 model.py 的基底排除清單")
        return frozenset()

    extra = frozenset(cfg.EXTRA_EXCLUDE_COLS)
    print(f"  ▸ 對齊 live 特徵集，額外自訓練排除 {len(extra)} 欄：{sorted(extra)}")
    return extra


def align_features_with_live(cfg: RunConfig):        # pragma: no cover - 遷移用
    """★ 已移除（2026-07-27）。舊呼叫端會拿到遷移說明，而不是靜默無效。"""
    raise RuntimeError(
        "align_features_with_live() 已於 2026-07-27 移除（它 monkeypatch "
        "objective.model._EXCLUDE_COLS，需要搭配 importlib 載入順序才生效）。\n"
        "改法：把排除清單傳進 config —\n"
        "    WalkForwardConfig(..., extra_exclude_cols=resolve_extra_exclude_cols(cfg))\n"
        "前綴排除則用 extra_exclude_prefixes=(...)。詳見 summary_research.md §3.3。"
    )


# ============================================================
#  主流程
# ============================================================

def run_ml(cfg: RunConfig):
    # ── ★ 超參數決策 ─────────────────────────────────────────
    if cfg.TUNE_MODE == "frozen":
        print("\n► 超參數模式：frozen（不使用 Optuna，套用固定超參數）")
        best_params = load_best_params(cfg)
    else:
        _desc = {
            "every_fold": "每個 fold 都重跑 Optuna",
            "first_only": "第一個 fold tune，之後凍結沿用",
            "periodic":   f"每 {cfg.RETUNE_EVERY_N} 個 fold 重新 tune 一次",
        }[cfg.TUNE_MODE]
        print(f"\n► 超參數模式：{cfg.TUNE_MODE}（{cfg.N_TRIALS} trials）— {_desc}")
        best_params = None

    # ── ★ 訓練特徵集對齊（走 config，不再改模組全域）──────────
    print("\n► 特徵集對齊（train/serve consistency）")
    extra_exclude_cols = resolve_extra_exclude_cols(cfg)

    # ── ★ 流動性篩選（805_group2 formula3）──────────────────
    if cfg.LIQ_FILTER_ENABLED:
        print("\n► 流動性篩選（formula3_amount_mv_turnover）")
        train_precomputed_dir = build_liquidity_filtered_dir(cfg)
    else:
        train_precomputed_dir = cfg.PRECOMPUTED_DIR

    print("\n► Walk-Forward Training")
    ml_config = WalkForwardConfig(
        window_start      = cfg.ML_WINDOW_START,
        window_end        = cfg.ML_WINDOW_END,
        precomputed_dir   = train_precomputed_dir,
        target_col        = cfg.LABEL,
        n_trials          = cfg.N_TRIALS,
        use_vol_weight    = False,
        use_return_weight = cfg.USE_RETURN_WEIGHT,
        tune_mode         = cfg.TUNE_MODE,
        retune_every_n    = cfg.RETUNE_EVERY_N,
        extra_exclude_cols = extra_exclude_cols,
    )
    trainer = WalkForwardTrainer(ml_config)

    if cfg.TUNE_MODE == "frozen":
        # 預植固定超參數 → _should_tune 對所有 fold 回傳 False。
        # 若忘了給參數，run() 會直接拋 RuntimeError（不會靜默改跑 Optuna）。
        trainer.preload_params(best_params)

    trainer.run()

    # ── ★ 另存 tune 出的參數，供之後 TUNE_MODE="frozen" 重跑 ──
    #    注意：every_fold / periodic 模式下每折的參數都不同，這裡存的是
    #    **最後一折**的參數，不代表全程使用的超參數。逐折參數請看
    #    database/experiment/{window}/best_params.json。
    if cfg.TUNE_MODE == "first_only" and trainer._frozen_params:
        out_path = Path("best_params_regression.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trainer._frozen_params, f, ensure_ascii=False, indent=2)
        print(f"\n  ✓ 本次 tune 出的超參數已另存：{out_path}")
        print(f"    （之後可設 TUNE_MODE=\"frozen\" 並將 BEST_PARAMS_PATH 指向此檔，"
              f"跳過 tune 快速重跑）")
    elif cfg.TUNE_MODE in ("every_fold", "periodic"):
        print(f"\n  ▸ TUNE_MODE={cfg.TUNE_MODE}：各折超參數不同，未另存單一檔案。"
              f"逐折參數見 {ml_config.output_dir}/{{window}}/best_params.json")


def main():
    cfg = RunConfig()

    print(f"\n{'='*60}")
    print(f"  Walk-Forward Pipeline")
    print(f"  資料來源  : {cfg.PRECOMPUTED_DIR}/")
    print(f"  年份範圍  : {cfg.ML_WINDOW_START} ~ {cfg.ML_WINDOW_END}")
    print(f"  Label     : {cfg.LABEL}")
    if cfg.TUNE_MODE == "frozen":
        print(f"  Optuna    : 停用（TUNE_MODE=frozen，固定超參數）")
    else:
        print(f"  Optuna    : TUNE_MODE={cfg.TUNE_MODE}（{cfg.N_TRIALS} trials）")
    print(f"  對齊 live  : {cfg.ALIGN_FEATURES_WITH_LIVE}")
    if cfg.LIQ_FILTER_ENABLED:
        print(f"  流動性篩選: 開啟（formula3  w1={cfg.LIQ_W1}  w2={cfg.LIQ_W2}  "
              f"keep_ratio={cfg.LIQ_KEEP_RATIO}  tag={liq_tag(cfg.LIQ_W1, cfg.LIQ_W2, cfg.LIQ_KEEP_RATIO)}）")
    else:
        print(f"  流動性篩選: 關閉（全市場訓練）")
    print(f"{'='*60}")

    if not cfg.RUN_ML:
        print("  RUN_ML=False，結束。")
        return

    t0 = time.time()
    run_ml(cfg)
    print(f"\n  總耗時: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()