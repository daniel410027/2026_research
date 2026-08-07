"""
make_full_predictions_0709.py
=============================
**對未篩選的全市場打分**，產生 `database/experiment_full/predictions_full.csv`，
給 `exp_oob_policy_0709.py` 的 `OOB_POLICY="model"` 用。

─────────────────────────────────────────────
  為什麼要有這支
─────────────────────────────────────────────
  `database/experiment/*/predictions.csv` 只含 `in_filter=1` 的宇宙
  （每日約 196 檔 ＝ 全市場流動性排名前 20%），因為 `main_fix_0709.py` 是
  **train + predict 一起篩**（見 `build_liquidity_filtered_dir` 檔頭）。
  於是持倉一旦掉出宇宙就查無分數 → `backtest_real.py` 視為 −inf 無條件淘汰。
  那是資料缺口造成的強制汰換，不是模型的判斷（佔全部賣出的 18.4%）。

  本檔**不改訓練集**，只把預測階段從「篩選後的測試年」換成「未篩選的測試年」。

─────────────────────────────────────────────
  怎麼保證是「同一個模型，只是多打幾檔」
─────────────────────────────────────────────
  1. 訓練資料：沿用產生現有 predictions 的那份流動性篩選快取。
     ⚠ 這裡有個陷阱。`database_make_liq_filtered/` 下有兩份快取：
         f3_..._kr02000       EXCLUDE_STALE_QUOTES=False
         f3_..._kr02000_nz5   EXCLUDE_STALE_QUOTES=True（現行 RunConfig）
     而 `database/experiment/liquidity_filter_meta.json` 的逐年列數對得上的是
     **前者**——也就是現有的 predictions 是舊設定跑的，跟現行 config 已經漂移。
     本檔自動比對 meta 挑出「真正產生現有 predictions 的那一份」，而不是照
     現行 config 重建，否則重訓出來的是另一個模型，比較就沒有意義。
  2. 超參數：讀各 window 自己的 `database/experiment/{window}/best_params.json`
     （`TUNE_MODE="every_fold"` 逐折不同），不重跑 Optuna。
  3. 其餘（特徵解析、排除清單、樣本權重、訓練輪數、seed）全部沿用
     `WalkForwardTrainer`——本檔只覆寫 `_preprocess_window` 的資料來源，
     不複製任何訓練程式碼。
  4. ★ **驗證關卡**：重訓後，落在原宇宙內的那些列的預測值必須與現有
     `predictions.csv` 逐筆相同。對不上就代表沒重現成功，本檔會直接報錯，
     不會靜默產出一份「別的模型」的分數。

─────────────────────────────────────────────
  已知限制
─────────────────────────────────────────────
  · **外插**：模型只在流動性前 20% 的樣本上訓練，對被剔除的股票打分是外插到
    訓練分布之外。分數算得出來，可信度與宇宙內的不同。要根治得連訓練集一起
    放寬，但那會改變模型本身，就不再是乾淨對照。
  · `collect_oof_prob` 被停用（見下方 _NoOOF）：那只影響 metrics.json 裡的
    `oof_ic` 監控值（會是 NaN），不影響模型與預測值。省下 n_splits 倍的訓練時間。
  · 測試年不做行情凍結剔除：殭屍股會拿到分數，但 `backtest_real.py` 自己的
    `ENABLE_STALE_FILTER` 會強制賣出，不影響結論。
  · 目標值（`excess_return`）為 NaN 的列不會有分數（沿用 `_preprocess_window`
    的 `dropna`）。覆蓋率印在報表尾。

  [輸出] database/experiment_full/
           {window}_L4/predictions.csv   逐 window 的全市場預測
           predictions_full.csv          ★ 九年合併，exp_oob_policy 讀這份
           coverage.csv                  逐年覆蓋率與驗證關卡的比對結果

─────────────────────────────────────────────
  ⚠ 在本 repo（2026_research）直接執行會卡在驗證關卡
─────────────────────────────────────────────
  2026-08-07 由 ~/Desktop/1111_monthholding 移植過來。**現有的
  database/experiment_full/ 是在 1111_monthholding 產生後複製過來的**，
  不是在這裡跑出來的。理由：

    · 本 repo 的 `objective/model.py` 有 2026-08-06 的 random_state 修正
      （把 cfg.random_state 傳給 train_final_lgb / collect_oof_prob），
      1111_monthholding 那份沒有。
    · 但兩邊現有的 `database/experiment/*/predictions.csv` 是 **byte 相同**的
      ——都是修正**之前**用 LightGBM 預設種子產生的。
    · 於是在這裡重訓會得到 seed=42 的模型，第 4 條驗證關卡（宇宙內預測必須與
      現有 predictions.csv 逐筆相同）會直接報錯。那是關卡正常運作，不是 bug。

  所以複製過來是安全的：兩邊的 predictions.csv 相同 ⟹ 1111 那份 experiment_full
  通過的驗證關卡對本 repo 同樣成立（coverage.csv 九個 window 的最大絕對差皆為 0.0）。

  什麼時候才該在這裡重跑本檔：等 `database/experiment/` 用修正後的 model.py
  重新產生之後——那時 predictions.csv 會改變，這份 experiment_full 就失效，
  必須連同重跑。

作者：Daniel Huang
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from objective.research_config import RunConfig
from objective.model import WalkForwardTrainer, WalkForwardConfig
import objective.model as OM
from main_fix_0709 import resolve_extra_exclude_cols

_ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = _ROOT / "database" / "experiment"
OUT_DIR        = _ROOT / "database" / "experiment_full"
LIQ_ROOT       = _ROOT / "database_make_liq_filtered"
UNFILTERED_DIR = _ROOT / "database_make"

TOL = 1e-10          # 驗證關卡：宇宙內預測值的最大容許差


# ============================================================
#  找出「真正產生現有 predictions 的那份訓練資料」
# ============================================================

def resolve_train_dir() -> Path:
    """比對 liquidity_filter_meta.json 的逐年列數，挑出與現有 experiment 相符的快取。"""
    exp_meta = json.loads((EXPERIMENT_DIR / "liquidity_filter_meta.json").read_text(encoding="utf-8"))
    want = exp_meta.get("rows_by_year", {})
    if not want:
        raise ValueError(f"{EXPERIMENT_DIR}/liquidity_filter_meta.json 沒有 rows_by_year，無法比對")

    cands = []
    for d in sorted(LIQ_ROOT.iterdir()):
        meta = d / "liquidity_filter_meta.json"
        if not meta.exists():
            continue
        got = json.loads(meta.read_text(encoding="utf-8")).get("rows_by_year", {})
        shared = [y for y in want if y in got]
        match = sum(1 for y in shared if got[y].get("after") == want[y].get("after"))
        cands.append((match, len(shared), d))
        print(f"  {d.name:<40} 共同年份 {len(shared):>2} 逐年列數相符 {match:>2}")

    cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_match, best_shared, best_dir = cands[0]
    if best_match < len(want):
        print(f"  ⚠ 最佳快取 {best_dir.name} 只有 {best_match}/{len(want)} 年對得上；"
              f"缺的年份會讓該 window 的驗證關卡失敗。")
    print(f"  ► 訓練資料採用：{best_dir}")
    return best_dir


# ============================================================
#  只換資料來源的 Trainer
# ============================================================

class _NoOOF:
    """停用 collect_oof_prob：只影響 metrics.json 的 oof_ic 監控值，不影響模型。"""
    def __init__(self):
        self.orig = OM.collect_oof_prob

    def __enter__(self):
        OM.collect_oof_prob = lambda X, y, **kw: np.full(len(X), np.nan)
        print("  ▸ collect_oof_prob 已停用（oof_ic 會是 NaN，模型與預測值不受影響）")

    def __exit__(self, *a):
        OM.collect_oof_prob = self.orig


class FullUniverseTrainer(WalkForwardTrainer):
    """訓練年讀流動性篩選後的資料（同原流程），**測試年改讀未篩選的全市場**。"""

    def __init__(self, config: WalkForwardConfig, unfiltered_dir: Path):
        super().__init__(config)
        self.unfiltered_dir = unfiltered_dir

    def _preprocess_window(self, start: int, end: int) -> pd.DataFrame:
        key = (start, end)
        if key in self._df_cache:
            return self._df_cache[key]

        test_year   = end
        train_years = self.cfg.train_years(test_year)

        frames = []
        for year in train_years:                       # 篩選後（與原流程相同）
            path = self.cfg.precomputed_dir / f"{year}.csv"
            if not path.exists():
                raise FileNotFoundError(f"找不到訓練資料：{path}")
            frames.append(pd.read_csv(path, low_memory=False))
            print(f"  讀取 {path}（訓練，已篩選）")

        path = self.unfiltered_dir / f"{test_year}.csv"  # ★ 未篩選的全市場
        if not path.exists():
            raise FileNotFoundError(f"找不到測試年全市場資料：{path}")
        test_raw = pd.read_csv(path, low_memory=False)
        print(f"  讀取 {path}（測試，未篩選，{len(test_raw):,} 列）")
        frames.append(test_raw)

        df = pd.concat(frames, ignore_index=True)
        df["年月日"]   = pd.to_datetime(df["年月日"])
        df["證券代碼"] = df["證券代碼"].astype(str)
        df = df.sort_values(["證券代碼", "年月日"]).reset_index(drop=True)
        return df.dropna(subset=[self.cfg.target_col])


# ============================================================

def run(train_dir: Path | None = None,
        out_dir:   Path | None = None,
        verify:    bool = True,
        seed_override: int | None = None) -> pd.DataFrame:
    """重訓 + 全市場打分。回傳逐 window 的覆蓋率／驗證表。

    train_dir : 訓練用的流動性篩選目錄。None = 自動比對出「產生現有 predictions
                的那一份」（見 resolve_train_dir）。
    out_dir   : 輸出根目錄。None = database/experiment_full。
    verify    : 是否啟用驗證關卡（宇宙內預測值須與 database/experiment 逐筆相同）。
                ⚠ 只有在 train_dir 就是產生現有 predictions 的那一份時才該開。
                換了訓練資料（例如改 EXCLUDE_STALE_QUOTES）預測值本來就會不同，
                這時要關掉，否則會誤報成「沒重現成功」。
    seed_override : 塞進 best_params 的 LightGBM `seed`，用來量重訓噪音。
                ⚠ 這是**必要的**：`_run_window` 呼叫 `train_final_lgb` 時沒有傳
                  random_state，而 `_LGB_BASE_PARAMS` 與存檔的 best_params 都沒有
                  `seed` 這一欄 → 實際吃的是 LightGBM 預設值 0。所以
                  `WalkForwardConfig.random_state` 只影響 Optuna 與 CV 切折，
                  **不影響最終模型**。要變動最終模型的隨機性只能從這裡塞。
                  → seed_override=0 應該與不指定時逐位元相同（可當重現關卡）。
    """
    cfg = RunConfig()
    if train_dir is None:
        print("\n[1/4] 比對快取，找出產生現有 predictions 的訓練資料 ...")
        train_dir = resolve_train_dir()
    else:
        print(f"\n[1/4] 訓練資料（呼叫端指定）：{train_dir}")
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if not verify:
        print("  ⚠ 驗證關卡已關閉（訓練資料與現有 predictions 不同源，預測值本來就會不同）")

    ml_cfg = WalkForwardConfig(
        window_start           = cfg.ML_WINDOW_START,
        window_end             = cfg.ML_WINDOW_END,
        precomputed_dir        = train_dir,
        target_col             = cfg.LABEL,
        n_trials               = cfg.N_TRIALS,
        use_vol_weight         = cfg.USE_VOL_WEIGHT,
        use_return_weight      = cfg.USE_RETURN_WEIGHT,
        tune_mode              = "frozen",          # ★ 不跑 Optuna，逐折讀存檔參數
        retune_every_n         = cfg.RETUNE_EVERY_N,
        extra_exclude_cols     = resolve_extra_exclude_cols(cfg),
        train_years_n          = cfg.TRAIN_YEARS_N,
        train_year_offset      = cfg.TRAIN_YEAR_OFFSET,
        decay_half_life_months = cfg.DECAY_HALF_LIFE_MONTHS,
        output_dir             = out_dir,
        random_state           = cfg.RANDOM_STATE,
    )

    print("\n[2/4] 建立 Trainer ...")
    trainer = FullUniverseTrainer(ml_cfg, UNFILTERED_DIR)
    windows = ml_cfg.windows()
    print(f"  {len(windows)} 個 window：{windows[0]} → {windows[-1]}")

    print("\n[3/4] 逐 window 重訓 + 全市場打分 ...")
    rows, preds = [], []
    with _NoOOF():
        for fold, (start, end) in enumerate(windows, 1):
            name = trainer.window_dir_name(start, end)
            bp_path = EXPERIMENT_DIR / name / "best_params.json"
            # ★ 訓練目錄含 "_nz" 時 window_dir_name 會加尾綴 z（*_L4z），但存檔的
            #   超參數在不帶 z 的 *_L4 目錄下。沿用它是刻意的：本實驗只換宇宙定義，
            #   超參數必須固定，否則差異會混進 Optuna 的隨機性。
            if not bp_path.exists() and name.endswith("z"):
                alt = EXPERIMENT_DIR / name[:-1] / "best_params.json"
                if alt.exists():
                    print(f"  ▸ {name} 無存檔參數 → 沿用 {name[:-1]} 的（只換宇宙、不動超參數）")
                    bp_path = alt
            if not bp_path.exists():
                raise FileNotFoundError(
                    f"{bp_path} 不存在——無法重現該折的模型。"
                    f"（TUNE_MODE='every_fold' 時逐折參數都存在各 window 目錄下）")
            params = json.loads(bp_path.read_text(encoding="utf-8"))
            if seed_override is not None:
                params["seed"] = seed_override    # 進 {**_LGB_BASE_PARAMS, **best_params}
            trainer.preload_params(params)

            print(f"\n{'='*66}\n  [{fold}/{len(windows)}] window {name}\n{'='*66}")
            trainer._run_window(start, end, fold_index=fold)
            trainer._df_cache.clear()          # 每折資料很大，跑完就丟

            new = pd.read_csv(out_dir / name / "predictions.csv",
                              dtype={"證券代碼": str}, encoding="utf-8-sig")
            new["年月日"] = pd.to_datetime(new["年月日"])
            new["證券代碼"] = new["證券代碼"].str.strip()

            if verify:
                old = pd.read_csv(EXPERIMENT_DIR / name / "predictions.csv",
                                  dtype={"證券代碼": str}, encoding="utf-8-sig")
                old["年月日"] = pd.to_datetime(old["年月日"])
                old["證券代碼"] = old["證券代碼"].str.strip()

                # ── 驗證關卡：宇宙內的預測值必須逐筆相同 ──────────────
                j = old.merge(new, on=["證券代碼", "年月日"], how="left",
                              suffixes=("_old", "_new"))
                missing = int(j["y_pred_new"].isna().sum())
                diff = float(np.nanmax(np.abs(j["y_pred_new"] - j["y_pred_old"]))) if len(j) else 0.0
                print(f"\n  驗證：原宇宙 {len(old):,} 列，新預測涵蓋 {len(old) - missing:,} 列，"
                      f"最大絕對差 {diff:.3e}")
                if missing or diff > TOL:
                    raise RuntimeError(
                        f"window {name} 未能重現原預測值（缺 {missing} 列、最大差 {diff:.3e} > {TOL:.0e}）。\n"
                        f"代表重訓出來的不是同一個模型，比較會失去意義。請先查訓練資料與超參數來源。")
                rows.append({"window": name, "test_year": end,
                             "原宇宙列數": len(old), "全市場列數": len(new),
                             "倍數": round(len(new) / len(old), 2),
                             "最大絕對差": diff})
            else:
                rows.append({"window": name, "test_year": end,
                             "原宇宙列數": np.nan, "全市場列數": len(new),
                             "倍數": np.nan, "最大絕對差": np.nan})
            preds.append(new[["證券代碼", "年月日", "y_pred"]])

    print("\n[4/4] 合併輸出 ...")
    full = pd.concat(preds, ignore_index=True).sort_values(["年月日", "證券代碼"])
    full.to_csv(out_dir / "predictions_full.csv", index=False, encoding="utf-8-sig")
    cov = pd.DataFrame(rows)
    cov.to_csv(out_dir / "coverage.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(cov.to_string(index=False))
    print(f"\n  ✓ 九折全部通過驗證關卡（宇宙內預測值逐筆相同）")
    print(f"  輸出：{out_dir / 'predictions_full.csv'}  {full.shape}")


if __name__ == "__main__":
    main()
