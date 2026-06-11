"""
Metric computation utilities — all metrics used in the paper.
"""
import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, classification_report,
    precision_recall_curve, roc_curve, confusion_matrix
)
import json, os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import METRICS_DIR, STEALTHY_CLASSES, ALL_CLASSES


def compute_all_metrics(y_true, y_pred, y_prob, encoder, model_name,
                         latency_ms=None):
    classes = encoder.classes_
    n_classes = len(classes)

    f1_mac  = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    f1_wgt  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc     = matthews_corrcoef(y_true, y_pred)

    # AUC-PR (macro one-vs-rest)
    try:
        auc_pr = average_precision_score(
            np.eye(n_classes)[y_true], y_prob,
            average='macro'
        )
    except Exception:
        auc_pr = float('nan')

    # AUC-ROC (macro one-vs-rest)
    try:
        auc_roc = roc_auc_score(
            np.eye(n_classes)[y_true], y_prob,
            average='macro', multi_class='ovr'
        )
    except Exception:
        auc_roc = float('nan')

    # Per-class metrics
    report = classification_report(
        y_true, y_pred,
        target_names=classes, output_dict=True, zero_division=0
    )

    # Recall on stealthy classes
    stealthy_recalls = {}
    for cls in STEALTHY_CLASSES:
        if cls in report:
            stealthy_recalls[cls] = report[cls]['recall']

    acc = report.get('accuracy', float('nan'))

    metrics = {
        "model":            model_name,
        "accuracy":         round(acc,    4),
        "f1_macro":         round(f1_mac, 4),
        "f1_weighted":      round(f1_wgt, 4),
        "auc_roc":          round(auc_roc, 4),
        "auc_pr":           round(auc_pr,  4),
        "mcc":              round(mcc,     4),
        "stealthy_recalls": {k: round(v, 4) for k, v in stealthy_recalls.items()},
        "per_class":        {c: {
            "precision": round(report[c]['precision'], 4),
            "recall":    round(report[c]['recall'],    4),
            "f1":        round(report[c]['f1-score'],  4),
            "support":   int(report[c]['support'])
        } for c in classes if c in report},
        "latency_p95_ms":   round(latency_ms, 2) if latency_ms else None,
    }
    return metrics


def bootstrap_ci(scores, n=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    boot = np.array([
        rng.choice(scores, len(scores), replace=True).mean()
        for _ in range(n)
    ])
    return float(np.median(scores)), float(np.percentile(boot, 100*alpha/2)), \
           float(np.percentile(boot, 100*(1-alpha/2)))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

    def encode(self, obj):
        if isinstance(obj, dict):
            obj = {(str(k) if not isinstance(k, str) else k): v
                   for k, v in obj.items()}
        return super().encode(obj)


def save_metrics(metrics, filename):
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, filename)

    def sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sanitize(metrics), f, indent=2)
    print(f"  Saved: {path}")
    return path
