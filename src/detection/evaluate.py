"""
evaluate.py

Evaluation metrics and evaluation suite for attack detection models.
Calculates binary classification metrics (Precision, Recall, F1, ROC-AUC, PR-AUC,
False Positive Rate, Confusion Matrix) suited for imbalanced time-series data.
Explicitly distinguishes window-level detection metrics from component-level
localization metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class DetectionEvaluator:
    """Evaluates binary attack detection performance on window/time-step predictions.

    Attributes:
        threshold: Classification decision threshold on predicted probabilities (default 0.5)
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute binary classification metrics on test predictions.

        Args:
            y_true: 1D ground-truth binary labels (0 = normal, 1 = attack).
            y_probs: 1D predicted probabilities for class 1 (attack).
            threshold: Optional override for decision threshold.

        Returns:
            Dict containing:
              - precision: Precision score (TP / (TP + FP))
              - recall: Recall / Sensitivity (TP / (TP + FN))
              - f1_score: Harmonic mean of precision and recall
              - roc_auc: Area under ROC curve (if positive and negative classes exist)
              - pr_auc: Precision-Recall Area Under Curve (robust for imbalanced data)
              - false_positive_rate: FPR = FP / (FP + TN)
              - confusion_matrix: Dict with TN, FP, FN, TP counts
              - metric_category: Explicitly set to 'DETECTION_METRICS'
        """
        thresh = threshold if threshold is not None else self.threshold
        y_true = np.asarray(y_true).ravel()
        y_probs = np.asarray(y_probs).ravel()

        if len(y_true) != len(y_probs):
            raise ValueError("y_true and y_probs must contain the same number of samples")

        labelled = np.isfinite(y_true) & (y_true != -999)
        y_true = y_true[labelled].astype(int)
        y_probs = y_probs[labelled]

        y_pred = (y_probs >= thresh).astype(int)

        # Confusion matrix elements
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        # AUC metrics require both classes to be present in y_true
        has_both_classes = len(np.unique(y_true)) > 1
        if has_both_classes:
            roc_auc = float(roc_auc_score(y_true, y_probs))
            pr_auc = float(average_precision_score(y_true, y_probs))
        else:
            roc_auc = None
            pr_auc = None
            logger.warning("ROC-AUC and PR-AUC cannot be computed because y_true contains only 1 class.")

        results = {
            "metric_category": "DETECTION_METRICS",
            "threshold": thresh,
            "samples": len(y_true),
            "attack_samples": int(np.sum(y_true)),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "false_positive_rate": fpr,
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "localization_note": (
                "These metrics evaluate window-level binary attack detection. "
                "Component-level localization precision/recall is a separate metric "
                "category evaluated in Phase 3 (Multi-level Localization)."
            ),
        }

        logger.info(
            "Detection Evaluation: F1=%.4f, Precision=%.4f, Recall=%.4f, PR-AUC=%s",
            f1,
            precision,
            recall,
            f"%.4f" % pr_auc if pr_auc is not None else "N/A",
        )

        return results
