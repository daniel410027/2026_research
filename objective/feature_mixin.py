"""
feature_mixin.py
================
向後相容 re-export：實際特徵工程邏輯已拆分至 objective/features/ 子模組
（technical / price_return / risk / institutional / fundamental），
設計說明與 look-ahead bias 修正歷史請見 objective/features/__init__.py。

保留本檔僅為維持既有 `from objective.feature_mixin import FeatureMixin` 匯入路徑。
"""

from __future__ import annotations

from .features import FeatureMixin

__all__ = ["FeatureMixin"]
