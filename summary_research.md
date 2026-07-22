# 2026_research 架構總覽

台股選股模型的**研究／回測環境**。姊妹資料夾 `2026_daily`（`~/Desktop/2026_daily`）
是每日實際下單的正式流程。兩邊共用 `objective/` 這個特徵工程＋模型套件，但
**各自維護一份副本**（見第 7 節），不是同一份程式碼。

這邊沒有排程、沒有下單、沒有寄信，只有離線的 walk-forward 回測。

---

## 0. 這個 repo 的定位

### 0.1 它是實驗的「起點」

這個資料夾的用途是**當各種實驗的乾淨起點**。要做新實驗時，直接把整個資料夾複製到
新資料夾（不透過 git branch），在副本裡改。實驗之間靠**不同的資料夾**隔離，
不是靠 git。

在副本裡，掛在不同階段的實驗會另外寫新的入口檔、指向新的輸出路徑，例如：

- 想試不同流動性 threshold → 另寫一支 `main_fix_*.py`，輸出到新路徑
- 想試 rho / beta 等下游算法 → 另寫一支 `backtest_*.py`，輸出到新路徑

所以固定的輸出路徑（`database/experiment/`、`output/backtest/`）不需要加 tag 或
namespace——同一個資料夾內本來就只跑一種設定。

**這個定位的直接後果：這裡的每個瑕疵都會被複製到未來所有實驗。** 一個沒修的 bug
不是影響一次實驗，是影響之後每一次。所以第 6 節那些「已知問題」的優先度，比在一般
專案裡高很多。同理，**修東西要修在這裡然後往外複製，不要只在某個副本裡修掉**，
否則下次複製又帶著同一個 bug 出去。

### 0.2 可重現性是這裡的預設

實驗常常是「固定上游、只改下游」——例如 rho/beta 實驗會反覆跑 backtest，但上游的
`database/experiment/` 預測值必須保持不變，否則比較的就不只是 rho 了。

因此 **`objective/lgb_utils.py::_PARALLEL_TRIALS` 預設 `False`**（見 4.4）：
慢 4 倍，但同 seed 同資料保證跑出同一組超參數與同一份預測。想快的時候可以打開，
但打開跑出來的東西不要拿去當別的實驗的上游。

同樣理由，`objective/` 裡的隨機來源都應該吃 `random_state`；新增邏輯時留意別引入
沒有 seed 的隨機性。

## 1. 怎麼跑

```bash
.venv/bin/python main_fix_0709.py     # walk-forward 訓練 → database/experiment/
.venv/bin/python backtest_0709.py     # 依預測值回測 → output/backtest/
```

所有設定集中在 `main_fix_0709.py` 的 `RunConfig` class，直接改該 class 的屬性，
不吃命令列參數。常動的幾個：

| 屬性 | 現值 | 意義 |
|---|---|---|
| `ML_WINDOW_START/END` | 2014 / 2025 | walk-forward 範圍，2026 不納入（資料未滿一年） |
| `LABEL` | `excess_return` | 連續值 regression（2026-07-14 由 0/1 分類改版） |
| `TUNE_MODE` | `"every_fold"` | 超參數策略，四選一（見 6.1） |
| `N_TRIALS` | 15 | 每次 tune 的 trial 數 |
| `LIQ_FILTER_ENABLED` | `True` | 流動性篩選（見 3.2） |
| `EXTRA_EXCLUDE_COLS` | 見 3.1 | 訓練特徵排除清單 |

## 2. 資料流

```
database_make/YYYY.csv                    ← 從 2026_daily 搬過來的成品
   │  （make_new.py + build_macro_external.py 的產出，本 repo 不重算）
   ▼
database_make_liq_filtered/{tag}/YYYY.csv ← main_fix_0709.py 的流動性篩選快取
   │                                         tag 依參數命名，同 tag 存在則重用
   ▼
objective/model.py  WalkForwardTrainer     ← 2014–2025，train 2 年 / test 1 年，10 個 window
   │
   ▼
database/experiment/{start}_{end}/        predictions.csv, metrics.json,
   │                                      feature_importance.csv, best_params.json
   ▼
backtest_0709.py → output/backtest/
```

**特徵資料不在這裡生成。** `database_make/` 是從 `2026_daily` 搬過來的成品，
含 macro 特徵（`sox_zscore20` + 5 個 `{macro}_x_beta` 交互項，共 88 欄）。
要更新特徵集就去 daily 那邊重跑 `make_new.py` 再搬過來，本 repo 沒有
`build_macro_external.py` / `make_new.py`。

⚠ 搬新資料過來後，`database_make_liq_filtered/` 的舊快取**不會自動失效**
（判斷依據只有參數 tag，不看來源檔 mtime）。換過 `database_make/` 就手動砍掉
對應的 tag 目錄，否則會靜默沿用舊特徵集。

## 3. 關鍵設計決策

### 3.1 訓練特徵排除

兩層機制，都在 `objective/model.py`：

- `_EXCLUDE_COLS`：精確欄名比對。基底清單在 model.py，`main_fix_0709.py` 的
  `EXTRA_EXCLUDE_COLS` 會在 `trainer.run()` 前 monkeypatch 進去（`align_features_with_live()`）。
- `_EXCLUDE_PREFIXES`：前綴比對，目前只有 `"clu_id"`。

排除的理由分三類：

| 欄位 | 理由 |
|---|---|
| `market_return_fwd` | 前視 T+1→T+2，look-ahead |
| `market_value`, `close` | **外部條件變數**：要研究 size/price × 模型訊號的交互效果，讓模型先看過就變循環論證 |
| `amount` | 只作流動性過濾用 |
| `clu_id*` | 群組編號是任意標籤，跨期 label switching 無對應意義 |
| `market_return` | per-day constant（日期指紋）→ **與 daily 分歧，見 7.2** |

### 3.2 流動性篩選（universe）

`formula3_amount_mv_turnover`：
`score = w1·Z(ln(amount)) + w2·Z(ln(market_value)) + (1−w1−w2)·Z(ln(amount/mv))`，
每日 cross-sectional 排名取前 20%。參數 `w1=0.1004, w2=0.1896, keep_ratio=0.20`
來自 805_group2 的 Optuna 搜尋。

train + predict 一起篩——本 repo 是純回測，沒有 daily 那種「既有持倉不能被篩掉」
的問題，不需要區分 train-only vs. predict-all。

### 3.3 `main_fix_0709.py` 繞過 `objective/__init__.py`（已無必要）

`main_fix_0709.py` 用 `importlib` 直接載入 `objective/model.py`（`_load_obj_module()`），
繞過 `__init__.py`。檔頭註解說原因是 `__init__.py` 內有 `from daily_model import ...`
而 `daily_model` 不在本 repo。

**這個理由已經不成立**——現在的 `objective/__init__.py` 只有相對 import，
`from objective.model import WalkForwardConfig` 直接跑得動（2026-07-22 實測）。
繞過機制留著不影響正確性，但新腳本**不需要照抄**，直接 import 即可。

（保留 `_load_obj_module()` 的副作用：它把 `objective.model` 塞進 `sys.modules`，
所以 `align_features_with_live()` 對 `_EXCLUDE_COLS` 的 monkeypatch 才會作用到
trainer 實際用的那份模組。改成正常 import 的話這段要一起確認。）

### 3.4 CV 切折以「唯一交易日」為單位

panel 資料同一天有上千檔股票，row-based `TimeSeriesSplit` 會把同一天拆在
train/valid 兩側 → 同日 cross-sectional 外洩。`lgb_utils._make_cv_folds` 改以唯一
交易日切分。另外 OOF 以 NaN（非 0）初始化，因為最早一段樣本永遠不在任何 valid fold。

---

## 4. 超參數：已確立的結論

來源：`1001_hyper_rhobeta` 的 300-trial walk-forward 研究（2026-07-21）。
**一句話：`learning_rate` 是唯一重要的旋鈕；調參的價值在「避開爛區」而非「找到最佳」。**

### 4.1 最重要的單一結果

每折取 CV 排名的 best / median / worst 三個 trial，各自訓練最終模型比 test daily IC：

| | 平均 test IC | vs best |
|---|---|---|
| best trial | +0.0745 | — |
| median trial | +0.0733 | −0.0012（p=0.81，勝負 5:5）|
| worst trial | +0.0424 | **−0.0322（p=0.0036，best 勝 9/10）** |

超參數確實重要（爛組態掉 43% IC），但 **CV 只能分辨「爛 vs 不爛」，分不出
「好 vs 最好」**。這解釋了為何各種選參目標／空間調整的對照實驗全部測不出差異——
它們比較的都是已落在好區域內的組態。也因此 `N_TRIALS` 15~20 就夠，多跑無用。

### 4.2 各參數的 Spearman ρ（300 trials，與目標值）

```
learning_rate +0.514  ≫  min_data_in_leaf −0.233  >  bagging_freq +0.159
> feature_fraction +0.122  >  bagging_fraction +0.024  >  num_leaves −0.013
```

`learning_rate` 分箱（fold 內 rank 標準化，越小越好）：

| lr 區間 | 平均 rank | n |
|---|---|---|
| ≤ 0.005 | **0.383** | 129 |
| 0.005–0.01 | 0.399 | 40 |
| 0.01–0.02 | 0.489 | 41 |
| 0.02–0.05 | 0.649 | 37 |
| 0.05–0.1 | 0.796 | 19 |
| 0.1–0.3 | **0.896** | 34 |

### 4.3 選參目標用 RMSE（不是 IC）

10 折成對實驗（唯一變因 = objective）：RMSE 臂平均 test IC 0.0807、IC(ICIR) 臂
0.0888，差 +0.0081 但 **p=0.113、95% CI [−0.0023, +0.0185] 含 0** → 統計上無法區分。
加上 IC objective **慢 2.29 倍**（daily-IC feval 每個 boosting iteration 都要跑，
且是 GIL-bound 的 Python 程式碼，會跟平行 trial 搶 GIL）→ 取 RMSE。

兩點未達顯著但值得記著：(a) IC 的優勢集中在 RMSE 表現最差的幾折（訊號弱時 IC
較撐得住）；(b) IC objective 的 test R² 為負（−0.016 vs RMSE 的 +0.005），
若下游需要預測值本身有意義（而非僅排序），IC 的校準明顯較差。

⚠ 這只跑了 **1 個 seed**，無法區分「+0.0081 是真效果」或「Optuna 抽樣運氣」。
要定案需 3–5 seed（每輪 13–30 分鐘）。

### 4.4 2026-07-22 移植進本 repo 的三項

**(1) 搜尋空間**（`objective/lgb_utils.py`）

| 參數 | 舊 | 新 | 依據 |
|---|---|---|---|
| `learning_rate` | [0.005, 0.3] | **[0.001, 0.03]** | >0.02 全為爛區，舊空間約一半 trial 浪費 |
| `min_data_in_leaf` | [5, 100] | **[5, 600]** | 舊上界在綁：放寬後立刻跑到 144–599 |
| `max_depth` | 搜 [5, 20] | **移出，固定 −1** | 與 `num_leaves` 冗餘 |

`max_depth` 的問題：它是硬上限（depth=d → 葉數 ≤ 2^d），10 折實測有 3 折的
`num_leaves` 完全失效（`num_leaves=957` 但 `max_depth=3` → 實際 8 個葉子）。
後果是浪費 trial 預算，且重要性分析會把 `num_leaves` 誤判為不重要（被遮蔽）。
移除後 `num_leaves` 的 ρ 仍只有 −0.013，確認它是真的不重要。

**(2) 平行化開關 `_PARALLEL_TRIALS`（`lgb_utils.py`，預設 `False`）**

實測 8 trials / 8 核：

| 配置 | 耗時 |
|---|---|
| 8 trials 並行 × 1 thread（`True`） | **42.0s** |
| 4 × 2 | 65.4s |
| 2 × 4 | 78.0s |
| 1 trial × 8 threads（`False`，預設） | 170.3s |

trial 層平行同樣核數快 **4.05 倍**，但 **`n_jobs > 1` 時 TPESampler 的抽樣順序受
排程影響，即使固定 seed 也不再逐次可重現**。

**預設關閉（維持可重現）**，理由見 0.2：本 repo 的 `database/experiment/` 預測值
常被下游實驗當成「固定的上游」反覆使用，上游一旦不可重現，下游對照全部被污染。
只有在「跑完就丟、純粹想快點看個大概」時才打開。

`n_jobs` 與 `num_threads` 必須互斥，程式已綁在一起，不要單獨改其中一個：

| `_PARALLEL_TRIALS` | `n_jobs` | `num_threads` / `OMP_NUM_THREADS` |
|---|---|---|
| `False`（預設） | 1 | `_N_CORES` |
| `True` | `_N_CORES` | 1 |

（`_N_CORES = cpu_count − 2`，留 2 核給系統。兩個都設 1 會變成完全單核、比舊設定
更慢；兩個都設 N 會超訂 CPU。macOS 上 `OMP_NUM_THREADS` 不設會偶發 segfault。）

**(3) `clu_id*` 前綴排除**（`objective/model.py`）

606_corr 群組編號 2017–2023 叫 `clu_id_local`、2024 起改名 `clu_id_daily`，
而排除清單只列了前者 → `clu_id_daily` 以 importance 0.152 / rank 41 進了訓練
（`database_make_liq_filtered/.../2025.csv` 兩個欄名同時存在）。改用
`_EXCLUDE_PREFIXES` 前綴比對，之後再改名不會漏。

---

## 5. 當前 baseline（2026-07-22）

**這是之後每個實驗該比較的基準。** commit `0ff4069`，`_PARALLEL_TRIALS=False`
（可重現），`TUNE_MODE="every_fold"`，`N_TRIALS=15`，74 個訓練特徵，
資料為 `f3_w101004_w201896_kr02000` 篩選後、含 macro 特徵的 `database_make/`。
walk-forward 總耗時 **6541.8 秒**（約 1h49m）。

### 5.1 逐 window

| window | test IC (pooled) | test IC (daily) | OOF IC | RMSE | R² | lr | num_leaves | min_data_in_leaf |
|---|---|---|---|---|---|---|---|---|
| 2014_2016 | 0.1056 | 0.1167 | 0.0863 | 0.0240 | −0.0071 | 0.0170 | 121 | 317 |
| 2015_2017 | 0.1324 | 0.1337 | 0.0818 | 0.0271 | +0.0045 | 0.0138 | 221 | 157 |
| 2016_2018 | 0.1017 | 0.1035 | 0.1311 | 0.0318 | +0.0068 | 0.0043 | 160 | 223 |
| 2017_2019 | 0.1016 | 0.0978 | 0.0954 | 0.0244 | +0.0008 | 0.0119 | 358 | 152 |
| 2018_2020 | 0.0941 | 0.0754 | 0.0598 | 0.0341 | −0.0012 | 0.0109 | 215 | 97 |
| 2019_2021 | 0.0743 | 0.0640 | 0.0700 | 0.0380 | +0.0058 | 0.0028 | 64 | 300 |
| 2020_2022 | 0.0737 | 0.0512 | 0.0617 | 0.0304 | −0.0154 | 0.0137 | 246 | 402 |
| 2021_2023 | 0.0515 | 0.0444 | 0.0542 | 0.0309 | −0.0162 | 0.0170 | 121 | 317 |
| 2022_2024 | 0.0333 | 0.0155 | 0.0131 | 0.0344 | −0.0001 | 0.0021 | 336 | 21 |
| 2023_2025 | 0.1013 | 0.0436 | 0.0315 | 0.0317 | +0.0102 | 0.0023 | 359 | 134 |
| **平均** | **0.0869** | **0.0746** | **0.0685** | | | | | |

`corr(oof_ic, test_ic) = 0.602`。

⚠ **兩種 IC 不可混用**：`metrics.json` 的 `ic` 欄是 **pooled**（整年所有
(股票, 日期) 攤平算一個 Spearman），而 §4 引用的 1001 研究數字是 **daily**
（每天算一次橫斷面 IC 再平均）。選股實際在乎的是 daily。兩者可以差很多——
2023_2025 pooled 0.1013 但 daily 只有 0.0436。**引用前先確認是哪一種。**

### 5.2 超參數落點（10 折，新搜尋空間）

| 參數 | 觀測範圍 | 空間 | 是否貼界 |
|---|---|---|---|
| `learning_rate` | 0.0021 – 0.0170 | [0.001, 0.03] | 否（上下都有餘裕）|
| `min_data_in_leaf` | 21 – 402 | [5, 600] | 否（舊上界 100 已被突破 7/10 折）|
| `num_leaves` | 64 – 359 | [16, 512] | 否 |

放寬 `min_data_in_leaf` 是對的：10 折裡有 **7 折**取到 >100，舊空間全部會被截在 100。
`learning_rate` 全部落在 ≤0.017，與 §4.2 的分箱證據一致（新上界 0.03 綽綽有餘）。

### 5.3 回測（`output/backtest/`，2016-01-04 – 2025-12-31，2438 個交易日，cost0）

| | 幾何年化 | 算術年化 | 年化 σ | **報表 Sharpe** | **標準 Sharpe** |
|---|---|---|---|---|---|
| D1 | 177.42% | 107.25% | 31.61% | 5.61 | **3.39** |
| 大盤 | 14.40% | 14.55% | 14.76% | 0.98 | 0.99 |
| 超額 | — | 92.70% | 26.33% | — | IR **3.52** |

D1 其他指標：MDD −28.08%、Calmar 6.32、hit rate 63.45%、alpha 1.6302、IR 3.5207。
單調性完好：D1 → D10 年化從 +177% 遞減到 −48%。

相對大盤最深水下 −14.41%（2025-08-28 → 谷底 2025-10-02，70 天）；
最長水下 130 天（2021-07-23 → 2021-11-30，−12.58%）。

⚠ **報表 Sharpe 5.61 是灌水的**（分子幾何 ÷ 分母算術，見 6.3），標準定義是 **3.39**。
對外引用一律用 3.39。⚠ 這是 **cost0**，未計交易成本（見 8.8）。

---

## 6. 已知問題

### 6.1 `_should_tune` 的凍結機制失效 ✅ 已於 2026-07-22 修正

**問題**：`_should_tune` 的第一個分支 `if not self.cfg.freeze_hyperparams: return True`
會在檢查 `_frozen_params` 之前就短路，而 `main_fix_0709.py` 傳的正是
`freeze_hyperparams=False`。後果：

- `TUNE_FIRST_FOLD=True` 實際上是「每個 fold 都重跑 N_TRIALS」，不是宣稱的
  「第一個 fold tune 後凍結」；存出的 `best_params_regression.json` 是
  **最後一折**的參數。
- `TUNE_FIRST_FOLD=False` 的預植參數被完全忽略，**Optuna 照跑**，
  `best_params.json` 被靜默丟棄，畫面卻印「Optuna 全程停用」。這條路等於謊報。

**修法**：兩個旗標收斂成單一 `tune_mode`（狀態互斥）：

| `tune_mode` | 行為 | 逐 fold |
|---|---|---|
| `"every_fold"`（預設） | 每個 fold 都重跑 Optuna | `TTTTTT` |
| `"first_only"` | 只有第一折 tune，之後凍結 | `T-----` |
| `"periodic"` | 每 `retune_every_n` 折重 tune | `T--T--`（n=3）|
| `"frozen"` | 完全不 tune，用預植參數 | `------` |

配套：`frozen` 未預植參數時在 `run()` 開頭直接 `RuntimeError`（不再靜默改跑
Optuna）；新增 `preload_params()` 取代直接戳 `trainer._frozen_params`；舊旗標
若仍被傳入會拋出含遷移對照表的 `ValueError`；`tune_mode` 錯字在建構時就擋下。

`RunConfig.TUNE_MODE` 預設 `"every_fold"`，**就是舊版實際跑的行為，換寫法不改結果**。

⚠ **但既有的舊結果仍需重新確認**：任何宣稱「用固定超參數跑」的產出，實際上跑的
是 Optuna。

### 6.2 `fillna(0)` 破壞 LightGBM 原生缺失值處理 🟡 未修

`objective/model.py` 的 `X_train/X_test = ...fillna(0)` 把「缺失」壓成「值 = 0」。
對值域不含 0 的特徵等於送進一個乾淨的哨兵值——例如 `clu_rank_ret_1` 真實值域
[0.05, 1.0]、真正等於 0 的筆數為 0 → 模型一刀切在 0 附近就能還原「這筆資料有沒有」，
把缺失模式當訊號用。應改為保留 NaN 交給 LightGBM 處理。

實測對 test IC 的影響在雜訊內，但這是正確性問題，且會污染 importance 解讀。
（注意：`objective/features/*.py` 裡也有大量 `fillna(0)`，那些是特徵構造的一部分，
要分開判斷，別一起改。）

### 6.3 回測 Sharpe 非標準定義 🟡 未修

`backtest_0709.py`：

```python
ann_ret = (1 + r).prod() ** (annual_days / len(r)) - 1   # 幾何
ann_std = r.std() * np.sqrt(annual_days)                  # 算術
sharpe  = ann_ret / ann_std                               # 幾何 ÷ 算術
```

分子幾何、分母算術，在強複利下系統性灌水。**本次 baseline 實測**（見 5.3）：
D1 報表 **5.61** → 標準定義 **3.39**，差 **1.65 倍**。
（1001 研究是 5.68 → 3.41；704_beta_rho 的 2.52 同樣偏高。）

注意大盤那列幾乎沒差（0.98 vs 0.99）——因為灌水幅度隨報酬率放大，只有高報酬的
decile 會嚴重失真，所以**光看大盤對得上不能當作公式正確的佐證**。

對外引用前需換算。建議兩個都輸出（`sharpe_std` / `sharpe_geom`），免得歷史報表
突然對不上。

### 6.4 `clu_*` 系列的宇宙錯位 🔴 未修

`clu_*` 特徵是在**另一套流動性篩選後的宇宙**上算的，與本 pipeline 的 formula3 不同。
2021 年實測：本篩選通過 46,735 筆、clu 宇宙 45,881 筆，兩者大小幾乎相同，
但**交集只有 26,999（58%）**。

兩個後果：

- **參照組錯位**：`clu_rank_ret_*`（群組內相對排名）、`clu_rel_vol_*`（群組內相對
  量能）的「群組」是在另一套宇宙裡分的，相對量的分母對不上。
- **缺失模式變成特徵**：訓練集內 clu_* 缺失率 42–45%，經 `fillna(0)`（見 6.2）後
  模型可一刀切出「是否屬於另一套流動性宇宙」＝一個殘餘流動性指示變數。
  （這也是 `clu_valid` 自身 importance 恆為 0 的原因——與其他 clu_* 的缺失模式
  完全冗餘。）

→ 量到的 clu importance 混合了真分群訊號與宇宙成員指示，**無法區分**。
目前 `clu_id*` 已排除但 `clu_rank_ret_*` / `clu_rel_vol_*` / `clu_valid` 仍在訓練中。
要用它們必須先讓兩邊宇宙對齊（流動性篩選改用 `clu_valid==1`，或在本 pipeline
的宇宙上重算 clu）。**在對齊之前，任何 clu 特徵的重要性數字都不可信。**
整組排除的實測代價：test IC 平均 −0.0009（雜訊等級）。

---

## 7. 與 2026_daily 的關係

### 7.1 同步狀態（2026-07-22）

兩邊 `objective/` 是**人工同步的兩份副本**，沒有任何自動檢查，而且已經證明會漏。

| | 2026_daily | 2026_research |
|---|---|---|
| `max_depth` 固定 −1 | ✅ 07-21 | ✅ 07-22 |
| `learning_rate` [0.001, 0.03] | ✅ 07-21 | ✅ 07-22 |
| `min_data_in_leaf` 上界 600 | ❌ 仍 100 | ✅ 07-22 |
| trial 層平行開關 | ❌（固定全核 thread） | ✅ 07-22（`_PARALLEL_TRIALS`，預設關） |
| `clu_id*` 前綴排除 | 部分（列舉 `clu_id_daily`） | ✅ 前綴 |
| `feature_mixin` 拆 `features/` | ✅ | ✅ |
| regression 改版 | ✅ | ✅ |

其中 `min_data_in_leaf` 上界、平行化開關、`clu_id*` 前綴這三項是 research 領先，
驗過之後值得回推 daily。**回推時做定點移植，不要整包覆蓋**——兩邊 `_EXCLUDE_COLS`
有意分歧（見 7.2），research 另有 `main_fix_0709.py`、`backtest_0709.py`、
`database_make_liq_filtered/` 等 daily 沒有的東西。

⚠ 因為實驗是整個資料夾複製出去的（見 0.1），**回推 daily 時要從這個資料夾拿，
不要從某個實驗副本拿**——副本裡的 `objective/` 通常混了該實驗自己的改動。
反過來，從 daily 移植過來的東西也要落在這個資料夾，才會被之後的複製繼承。

### 7.2 刻意的分歧：`market_return`

research 排除、daily 不排除。它是 per-day constant，兩年訓練僅約 490 個唯一交易日，
樹模型有可能記住日期指紋而非學到真訊號；daily 端的判斷是「應該沒有資料洩漏問題」。

**在這個分歧存在期間，兩邊的 IC/RMSE 不能互相比較。** 這正是本 repo 該做的實驗
之一（見第 8 節）。

### 7.3 daily 端待辦（本 repo 有責任的部分）

daily 的 `FIXED_PARAMS` 是在**舊搜尋空間**下校準的：`learning_rate=0.04495`
（在新上界 0.03 之外，且落在 4.2 分箱證據的爛區）、`max_depth=5` + `num_leaves=486`
（冗餘組合，實際只有 32 個葉子）。也就是說 daily 移植了搜尋空間但還沒拿到好處，
現行部署參數與搜尋空間互相矛盾。

重新校準要跑 walk-forward，天生屬於本 repo。

---

## 8. Backlog（未確立 / 值得做）

依價值排序：

1. **重新校準超參數交回 daily**（見 7.3）——目前線上跑在爛區。
2. **`market_return` 過擬合實驗**（見 7.2）——同時也解掉兩邊無法比較的問題。
3. **objective 對拍檢查**——加個 `diff` script 比對本資料夾與 `2026_daily` 的
   `objective/`，人工同步已經證明會漏。
4. **`clu_*` 宇宙對齊或整組排除**（見 6.4）。
5. **多 seed 驗證 objective 選擇**（見 4.3）——3–5 seed 才能定案 RMSE vs IC。
6. **`retune_every_n` 的最佳值未知**。本次 baseline 的 OOF IC 隨 fold 衰退
   （0.086 → 0.013），`corr(oof_ic, test_ic) = 0.602`。凍結超參數隨時間失效的
   程度沒做過系統性研究——`tune_mode="periodic"` 已可用（見 6.1），但 n 該設多少不知道。
7. **`max_depth` 移除的單因子對照**——6 維空間平均 test IC 0.0875、7 維 RMSE 臂
   0.0807，但該輪**同時改了 `min_data_in_leaf` 上界**，不是單因子對照，
   不可宣稱移除 max_depth 帶來提升。
8. **成本未納入**（本 repo 範圍外，日常流程另行處理）。第 5 節的 baseline 是 cost0。
   1001 研究的參考數字：D1 每日換手 76.1%（年化約 192 次單邊），損益兩平來回成本
   0.56%；台股實際 0.44%（手續費 5 折）–0.585%。該次 cost0 下 D1 幾何年化 180.2% /
   標準 Sharpe 3.41，加 0.44% 成本後降到 20.8% / 0.75（同期大盤 14.5% / 0.99）。
   本次 baseline 數字接近（177.4% / 3.39），成本後應該也是同一個量級。
9. **訊號來源未拆解**。模型 IC 0.0716 vs 零參數反轉（`-報酬率1`）IC 0.0564；
    D1 年化 180.2% vs 29.0%，成分股僅重疊 24.5%。模型確實在反轉之上加了東西，
    但「加了什麼」不知道。

---

## 9. 方法論陷阱（都踩過，別重蹈）

1. **不要用 RMSE 的相對差判斷「曲面平坦度」**。R² ≈ 1% 時 RMSE 的相對變化會把訊號
   差異壓縮近兩個數量級：30 個 trial 的 CV RMSE 全距僅 0.1–1.6%，看似「超參數不
   重要」，但直接實驗顯示 best 與 worst 的 test IC 差 0.032（43%）。
2. **log-scale 參數的貼界檢查必須在對數尺度上做**。`lr=0.0034` 相對 [0.001, 0.3]
   線性只佔 0.8%（誤判為貼界），對數尺度上是 21%（離邊界很遠）。
3. **參數沒有貼界，不代表界沒有在綁**。`min_data_in_leaf` 在 [5,100] 的 10 折觀測值
   是 41–91（看似未貼界），放寬到 600 後立刻跑到 144–599。噪音大、樣本少時觀測
   最大值會系統性低估最適區 → 應定期往外試探。
   **本次 baseline 再次確認**：放寬後 10 折有 7 折取到 >100（見 5.2）。
4. **跨 fold 比較無法評估選參指標**。`corr(CV RMSE, test IC) = −0.302 (p=0.40)`，
   而作為健全性檢查的 `corr(CV RMSE, test RMSE)` 也只有 +0.12——跨 fold 變異幾乎
   全來自「那兩年好不好預測」，模型品質訊號被淹沒。**必須同 fold 成對比較。**
5. **一次只改一個變因**。v1→v2 曾同時改 6 項，導致整輪對照無法歸因，必須另做成對
   實驗才得到結論。
6. **`ps aux` 第 10 欄是累計 CPU 時間，不是經過時間**。在 278% CPU 下誤讀會讓時間
   估計膨脹 7 倍。
7. **「IC」有兩種算法，數字差很多**。pooled（整年攤平算一個 Spearman）vs
   daily（每天算橫斷面 IC 再平均）。`metrics.json` 的 `ic` 是 pooled，1001 研究引用的
   是 daily。本次 baseline 平均 pooled 0.0869 vs daily 0.0746，單一 window 更可能
   差到兩倍以上（2023_2025：0.1013 vs 0.0436）。**跨來源引用 IC 前先確認定義。**
8. **報表指標可能不是標準定義**。`backtest_0709.py` 的 Sharpe 是幾何年化 ÷ 算術年化 σ，
   比標準定義高 1.65 倍（5.61 vs 3.39）。對外引用前先確認公式，別直接抄報表。
