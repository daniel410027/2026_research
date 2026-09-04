# 上市+上櫃（otc）母體擴張——2026_research 內的追加資料

完整 pilot 實驗方法與數字見 `1208_sii/summary_sii.md`。2026-08-17 更新：`download.py`/
`make_new.py` 的 otc 支援已正式併入 `2026_daily`（不再是隔離的 `otc_pilot/` 副本），
`database_make_otc/` 也已從 `2026_daily` 重新產出並補齊到完整歷史，見下方。

## 新增了什麼

- `database_make_otc/2014.csv ~ 2026.csv`：上市+上櫃母體的**成品**資料，93 欄，
  與 `database_make/` 欄位集合一致。**已涵蓋完整 2014–2026**（2026-08-17 從 2026_daily
  重新產出，取代舊的 2022–2026 pilot 版；`TRAIN_YEARS_N` 不再受限於 ≤2，可比照
  `database_make/` 用完整 12 年 walk-forward）。

★ 抓資料/建特徵的腳本（`download.py --markets sii,otc`、`make_new.py --dst-dir
database_make_otc`）正式放在 `2026_daily`（`--markets`/`--src-dir`/`--dst-dir` 皆為
新增的可選參數，預設值與純上市版行為完全一致，不影響 `2026_daily` 正式產出）。
`2026_research` 的定位仍是拿 `2026_daily` 處理好的資料做實驗，不放資料產製腳本，
避免這個 repo 或之後任何切片出去的副本視野被那些腳本干擾。若要重建或延伸
`database_make_otc/`，回 `2026_daily` 跑（見該 repo 的 `README_otc.md`）。

## 既有的 `database_make/`（純上市）完全沒被動過

`objective/research_config.py::RunConfig.PRECOMPUTED_DIR` 預設仍指向 `database_make/`，
任何沒有特別覆寫的實驗行為不變。

## 怎麼跑 otc 版本的實驗

```python
from objective.research_config import RunConfig
from main_fix_0709 import run_ml, narrow_to_test_year
from pathlib import Path

class OtcConfig(RunConfig):
    PRECOMPUTED_DIR = Path("database_make_otc")
    TRAIN_YEARS_N   = 2     # 現在已有完整 2014-2026，可視需要調大做完整
                             # walk-forward；沿用 pilot 當初驗證過的 2 年視窗
                             # 只是預設值，不再是資料範圍的硬限制。

base = OtcConfig()
for y in (2024, 2025, 2026):
    cfg = narrow_to_test_year(base, y)
    run_ml(cfg)
```

## 已知限制（跑之前務必看）

1. `database_make_otc/` 現已涵蓋完整 2014–2026，可以做完整 12 年 walk-forward，
   但**目前只有 pilot 當初的 2024-2026 三折（2 年訓練視窗）跑過模型/回測驗證**——
   資料補齊不等於驗證補齊，下結論前建議先用完整歷史重新走一次 walk-forward。
2. `best_params.json` 是舊的 2 年視窗 frozen 參數，**已知對現在的特徵集偏舊**（見
   `summary_sii.md` §1.2 的教訓）。要下結論前務必用 `TUNE_MODE="every_fold"` 重新調參，
   不要直接沿用 frozen 參數比較。
3. 流動性篩選、成本假設沿用既有純上市版本的參數，未針對上櫃另外校準。
4. 只涵蓋上櫃（otc），興櫃（rotc）尚未納入。
5. `download.py`/`make_new.py` 的 otc 支援已併入 `2026_daily`，但**線上模型
   （`daily_model/`）仍讀純上市的 `database_make/`，尚未切換**——這次只是把資料
   補齊到完整歷史，正式切換前仍需要先做完整 12 年驗證與上櫃專屬的成本/風控校準。

## 已完成的驗證結果摘要（2 年視窗 pilot，2024-2026 三折，皆獨立重新調參）

| 指標 | 純上市 | 上市+上櫃 | 差異 |
|---|---|---|---|
| 平均 ic_daily | 0.0481 | 0.0566 | +18% |
| 理想化回測 D1 Sharpe | 2.14 | 2.57 | — |
| 含成本回測 net Sharpe | 0.64 | 1.00 | — |

完整數字、方法論、caveat 見 `1208_sii/summary_sii.md`。
