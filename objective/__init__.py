"""
objective/
==========
新專案模型核心套件。資料來源由 TEJ CSV 改為 FinLab API。

對外介面：
    from objective.preprocess        import PreprocessConfig, DataPreprocessor
    from objective.model             import WalkForwardConfig, WalkForwardTrainer
    from objective.lgb_utils         import tune_lgb, train_final_lgb, evaluate_oof_ic, build_imp_df
    from objective.db_loader         import DBLoader
    from objective.liquidity         import compute_liquidity_mask, liq_tag
    from objective.run_manifest      import write_run_manifest
"""

from .preprocess_config import PreprocessConfig
from .preprocess        import DataPreprocessor
from .model             import WalkForwardConfig, WalkForwardTrainer
from .lgb_utils         import tune_lgb, train_final_lgb, evaluate_oof_ic, build_imp_df
from .db_loader         import DBLoader
from .liquidity         import compute_liquidity_mask, liq_tag
from .run_manifest      import write_run_manifest

__all__ = [
    "PreprocessConfig", "DataPreprocessor",
    "WalkForwardConfig", "WalkForwardTrainer",
    "tune_lgb", "train_final_lgb", "evaluate_oof_ic", "build_imp_df",
    "DBLoader",
    "compute_liquidity_mask", "liq_tag",
    "write_run_manifest",
]
