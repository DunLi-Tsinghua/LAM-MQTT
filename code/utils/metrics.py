from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def binary_classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "accuracy": _clean_float(accuracy_score(y_true, y_pred)),
        "macro_precision": _clean_float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": _clean_float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": _clean_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": _clean_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": None,
        "pr_auc": None,
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = _clean_float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = _clean_float(average_precision_score(y_true, y_score))
        except Exception:
            metrics["pr_auc"] = None
    return metrics
