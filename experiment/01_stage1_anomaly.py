"""
Stage 1 — Anomaly Detection (unsupervised).
Models: Isolation Forest, One-Class SVM (RBF), Local Outlier Factor.
Outputs: anomaly scores, optimal threshold θ, enrichment factors.
"""
import sys, os, time, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils.data_utils import load_and_split, get_normal_train, stealthy_enrichment
from utils.metrics_utils import save_metrics

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def calibrate_theta(scores_normal, scores_anomaly, target_recall=STAGE1_TARGET_RECALL):
    """
    Find the largest threshold θ such that Recall(anomaly) >= target_recall.
    Uses a grid over the combined score distribution.
    """
    combined = np.concatenate([scores_normal, scores_anomaly])
    thresholds = np.percentile(combined, np.linspace(1, 99, 500))

    best_theta = None
    best_metrics = {"recall": 0, "fpr": 1, "precision": 0}

    for theta in sorted(thresholds, reverse=True):
        tp = (scores_anomaly >= theta).sum()
        fn = (scores_anomaly  < theta).sum()
        fp = (scores_normal   >= theta).sum()
        tn = (scores_normal    < theta).sum()

        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        if recall >= target_recall:
            best_theta = theta
            best_metrics = {
                "theta":     float(theta),
                "recall":    round(recall, 4),
                "fpr":       round(fpr, 4),
                "precision": round(precision, 4),
                "f1":        round(2*precision*recall/(precision+recall+1e-9), 4)
            }
            break

    if best_theta is None:
        print(f"  WARNING: target recall {target_recall} not achievable; "
              f"using 95th percentile of normal scores.")
        best_theta = float(np.percentile(scores_normal, 95))
        best_metrics["theta"] = best_theta

    return best_theta, best_metrics


def evaluate_stage1(scores_normal, scores_anomaly, theta):
    y_true   = np.concatenate([np.zeros(len(scores_normal)),
                                np.ones(len(scores_anomaly))])
    y_scores = np.concatenate([scores_normal, scores_anomaly])
    y_pred   = (y_scores >= theta).astype(int)

    tp = int(((y_true==1) & (y_pred==1)).sum())
    fp = int(((y_true==0) & (y_pred==1)).sum())
    fn = int(((y_true==1) & (y_pred==0)).sum())
    tn = int(((y_true==0) & (y_pred==0)).sum())

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall":    round(tp/(tp+fn+1e-9), 4),
        "precision": round(tp/(tp+fp+1e-9), 4),
        "fpr":       round(fp/(fp+tn+1e-9), 4),
        "auc_roc":   round(roc_auc_score(y_true, y_scores), 4),
        "auc_pr":    round(average_precision_score(y_true, y_scores), 4),
    }


def run_stage1():
    print("=" * 60)
    print("STAGE 1 — Anomaly Detection")
    print("=" * 60)

    (X_train_raw, X_val_raw, X_test_raw,
     y_train, y_val, y_test,
     scaler_mm, scaler_std,
     encoder, feature_names, split_info) = load_and_split()

    print(f"\nDataset split (temporal):")
    print(f"  Train: {split_info['n_train']:,} | "
          f"Val: {split_info['n_val']:,} | Test: {split_info['n_test']:,}")

    # Stage 1 uses MinMax-scaled data (for AE/VAE compatibility; IF/LOF use raw)
    X_train_mm  = scaler_mm.transform(X_train_raw)
    X_val_mm    = scaler_mm.transform(X_val_raw)
    X_test_mm   = scaler_mm.transform(X_test_raw)

    X_normal_train = get_normal_train(X_train_mm, y_train)
    # For val: use all val as "anomaly candidate" (has both normal and attacks)
    X_val_normal = X_val_mm[y_val == 0]
    X_val_attack = X_val_mm[y_val != 0]

    print(f"\n  Normal train samples:  {len(X_normal_train):,}")
    print(f"  Val normal samples:    {len(X_val_normal):,}")
    print(f"  Val attack samples:    {len(X_val_attack):,}")

    models_s1 = {
        "Isolation Forest": IsolationForest(
            n_estimators=200, contamination=0.01,
            max_features=0.8, random_state=SEEDS[0]
        ),
        "One-Class SVM (RBF)": OneClassSVM(
            kernel='rbf', nu=0.05, gamma='scale'
        ),
        "LOF": LocalOutlierFactor(
            n_neighbors=20, contamination=0.05, novelty=True
        ),
    }

    stage1_results = {}

    for name, model in models_s1.items():
        print(f"\n--- {name} ---")
        t0 = time.time()
        model.fit(X_normal_train)
        t_fit = time.time() - t0

        # decision_function: higher = more normal, lower = more anomalous
        # We negate so that higher score = more anomalous
        t_inf = time.time()
        score_val_n = -model.decision_function(X_val_normal)
        score_val_a = -model.decision_function(X_val_attack)
        latency_ms  = (time.time() - t_inf) / len(X_val_mm) * 1000

        theta, theta_metrics = calibrate_theta(score_val_n, score_val_a)
        eval_metrics = evaluate_stage1(score_val_n, score_val_a, theta)

        print(f"  Theta = {theta:.4f} | Recall = {eval_metrics['recall']:.3f} | "
              f"FPR = {eval_metrics['fpr']:.3f} | AUC-ROC = {eval_metrics['auc_roc']:.3f}")
        print(f"  Fit time: {t_fit:.1f}s | Latency: {latency_ms:.3f} ms/sample")

        # Compute enrichment on test set
        score_test = -model.decision_function(X_test_mm)
        flagged_mask = score_test >= theta
        y_test_flagged = y_test[flagged_mask]
        r_before, r_after, factor = stealthy_enrichment(y_test, y_test_flagged, encoder)

        print(f"  Stealthy enrichment: {r_before:.4%} -> {r_after:.4%} "
              f"(x{factor:.1f})")

        stage1_results[name] = {
            "theta":        float(theta),
            "theta_metrics": theta_metrics,
            "eval_metrics": eval_metrics,
            "enrichment":   {"before": r_before, "after": r_after, "factor": factor},
            "latency_ms":   latency_ms,
            "fit_time_s":   t_fit,
            "n_flagged":    int(flagged_mask.sum()),
            "flagged_ratio": float(flagged_mask.mean()),
        }

    # Find best Stage 1 model (highest recall with FPR <= MAX_FPR)
    valid = {k: v for k, v in stage1_results.items()
             if v["eval_metrics"]["fpr"] <= STAGE1_MAX_FPR}
    if valid:
        best_s1 = max(valid, key=lambda k: valid[k]["eval_metrics"]["recall"])
    else:
        best_s1 = max(stage1_results,
                      key=lambda k: stage1_results[k]["eval_metrics"]["recall"])

    print(f"\n{'='*60}")
    print(f"Best Stage-1 model: {best_s1}")
    print(f"  Recall = {stage1_results[best_s1]['eval_metrics']['recall']:.3f}")
    print(f"  FPR    = {stage1_results[best_s1]['eval_metrics']['fpr']:.3f}")
    print(f"  Enrichment = x{stage1_results[best_s1]['enrichment']['factor']:.1f}")

    # Save everything
    output = {
        "models":    stage1_results,
        "best_model": best_s1,
        "split_info": split_info,
    }
    save_metrics(output, "stage1_results.json")

    # Save flagged indices for Stage 2
    best_model_obj = models_s1[best_s1]
    X_train_best   = X_train_mm
    score_train    = -best_model_obj.decision_function(X_train_best)
    theta_best     = stage1_results[best_s1]["theta"]
    flagged_train  = score_train >= theta_best
    score_test_best = -best_model_obj.decision_function(X_test_mm)
    flagged_test   = score_test_best >= theta_best

    np.save(os.path.join(METRICS_DIR, "s1_score_train.npy"),  score_train)
    np.save(os.path.join(METRICS_DIR, "s1_score_test.npy"),   score_test_best)
    np.save(os.path.join(METRICS_DIR, "s1_flagged_train.npy"), flagged_train)
    np.save(os.path.join(METRICS_DIR, "s1_flagged_test.npy"),  flagged_test)
    np.save(os.path.join(METRICS_DIR, "y_train.npy"),  y_train)
    np.save(os.path.join(METRICS_DIR, "y_val.npy"),    y_val)
    np.save(os.path.join(METRICS_DIR, "y_test.npy"),   y_test)
    np.save(os.path.join(METRICS_DIR, "X_train_mm.npy"), X_train_mm)
    np.save(os.path.join(METRICS_DIR, "X_test_mm.npy"),  X_test_mm)
    np.save(os.path.join(METRICS_DIR, "X_train_raw.npy"), X_train_raw)
    np.save(os.path.join(METRICS_DIR, "X_test_raw.npy"),  X_test_raw)

    print(f"\nStage 1 complete. {flagged_train.sum():,} train samples flagged "
          f"({flagged_train.mean():.2%}) | "
          f"{flagged_test.sum():,} test samples flagged ({flagged_test.mean():.2%})")
    return stage1_results


if __name__ == "__main__":
    run_stage1()
