"""
evaluate.py — Évaluation centralisée de tous les modèles.

Usage :
  python evaluate.py --model xgboost
  python evaluate.py --model all
  python evaluate.py --stage 1
  python evaluate.py --stage 2
"""

import sys, os, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ton_iot_config import *
from sklearn.metrics import (
    f1_score, accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)


def load_test_data(mode="standalone"):
    """Charge les données de test selon le mode."""
    with open(os.path.join(METRICS_DIR, "label_classes.json")) as f:
        class_names = json.load(f)["classes"]

    if mode == "standalone":
        X_ts = np.load(os.path.join(METRICS_DIR, "X_test_std.npy"))
        y_ts = np.load(os.path.join(METRICS_DIR, "y_test.npy"))
    else:
        flag_ts = np.load(os.path.join(METRICS_DIR, "s1_flag_test.npy"))
        sc_ts   = np.load(os.path.join(METRICS_DIR, "s1_score_test.npy"))
        X_raw   = np.load(os.path.join(METRICS_DIR, "X_test_std.npy"))
        y_raw   = np.load(os.path.join(METRICS_DIR, "y_test.npy"))
        X_ts = np.column_stack([X_raw[flag_ts], sc_ts[flag_ts].reshape(-1,1)])
        y_ts = y_raw[flag_ts]

    return X_ts, y_ts, class_names


def evaluate_sklearn_model(model_name, mode="standalone"):
    """Évalue un modèle sklearn sauvegardé."""
    import joblib
    model_path = os.path.join(RESULTS_DIR, "saved_models",
                               f"{model_name}_{mode}.pkl")
    if not os.path.exists(model_path):
        print(f"Modèle non trouvé : {model_path}")
        return None

    data = joblib.load(model_path)
    model       = data["model"]
    class_names = data["class_names"]
    n_cls       = len(class_names)

    X_ts, y_ts, _ = load_test_data(mode)

    y_pred = model.predict(X_ts)
    y_prob = (model.predict_proba(X_ts)
              if hasattr(model, 'predict_proba')
              else np.eye(n_cls)[y_pred])

    return _compute_and_plot(y_ts, y_pred, y_prob, class_names,
                              f"{model_name}_{mode}")


def _compute_and_plot(y_true, y_pred, y_prob, class_names, tag):
    """Calcule toutes les métriques et génère les figures."""
    n_cls  = len(class_names)
    f1m    = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    f1w    = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    acc    = accuracy_score(y_true, y_pred)
    mcc    = matthews_corrcoef(y_true, y_pred)
    y_bin  = np.eye(n_cls)[y_true]

    try:
        auc_roc = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
    except Exception:
        auc_roc = float('nan')
    try:
        auc_pr = average_precision_score(y_bin, y_prob, average='macro')
    except Exception:
        auc_pr = float('nan')

    rpt = classification_report(y_true, y_pred, target_names=class_names,
                                  output_dict=True, zero_division=0)
    stealthy_r = {c: round(rpt[c]['recall'], 4) for c in STEALTHY_CLASSES
                   if c in rpt}

    print(f"\n{'='*50}")
    print(f"ÉVALUATION : {tag}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1-macro  : {f1m:.4f}")
    print(f"  F1-weight : {f1w:.4f}")
    print(f"  AUC-ROC   : {auc_roc:.4f}")
    print(f"  AUC-PR    : {auc_pr:.4f}")
    print(f"  MCC       : {mcc:.4f}")
    print(f"  Recall classes furtives :")
    for c, r in stealthy_r.items():
        print(f"    {c:12s} : {r:.4f}")

    # Matrice de confusion
    cm_norm = confusion_matrix(y_true, y_pred,
                                labels=range(n_cls)).astype(float)
    cm_norm /= (cm_norm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.3, vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Confusion Matrix — {tag}")
    ax.set_ylabel("Vraie classe"); ax.set_xlabel("Prédite")
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"eval_cm_{tag}.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ROC par classe
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        if y_bin[:, i].sum() > 0:
            fpr_c, tpr_c, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            rc = auc(fpr_c, tpr_c)
            sty = '-' if cls in STEALTHY_CLASSES else '--'
            ax.plot(fpr_c, tpr_c, linestyle=sty,
                    color=PALETTE[i % len(PALETTE)], lw=1.5,
                    label=f"{cls} ({rc:.2f})")
    ax.plot([0,1],[0,1],'k--',alpha=0.4)
    ax.set(xlabel="FPR", ylabel="TPR",
           title=f"ROC curves — {tag}", xlim=[0,1], ylim=[0,1.02])
    ax.legend(fontsize=7, loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"eval_roc_{tag}.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    metrics = {
        "tag": tag,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1m, 4),
        "f1_weighted": round(f1w, 4),
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "mcc": round(mcc, 4),
        "stealthy_recalls": stealthy_r,
        "per_class": {
            c: {k: round(v, 4) for k, v in rpt[c].items()
                if k in ('precision','recall','f1-score','support')}
            for c in class_names if c in rpt
        }
    }
    with open(os.path.join(METRICS_DIR, f"eval_{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def compare_all_models():
    """
    Charge tous les résultats sauvegardés et génère un tableau comparatif.
    """
    results_dir = METRICS_DIR
    all_metrics = []

    for fname in os.listdir(results_dir):
        if fname.endswith("_metrics.json") or fname == "stage2_and_stats.json":
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
            if "stage2" in data:
                for name, m in data["stage2"].items():
                    all_metrics.append({
                        "model": name,
                        "f1_macro": m.get("f1_macro", "-"),
                        "auc_pr": m.get("auc_pr", "-"),
                        "mcc": m.get("mcc", "-"),
                        "backdoor_recall": m.get("stealthy_recalls", {}).get("backdoor", "-"),
                        "mitm_recall": m.get("stealthy_recalls", {}).get("mitm", "-"),
                    })

    if not all_metrics:
        print("Aucun résultat trouvé. Exécuter d'abord les scripts d'entraînement.")
        return

    print("\n" + "=" * 100)
    print("TABLEAU COMPARATIF — TOUS LES MODÈLES")
    print("=" * 100)
    print(f"{'Modèle':45s} {'F1-macro':>10} {'AUC-PR':>10} {'MCC':>10} "
          f"{'backdoor':>12} {'mitm':>10}")
    print("-" * 100)
    for m in sorted(all_metrics, key=lambda x: x.get("f1_macro", 0),
                     reverse=True):
        print(f"{str(m['model'])[:45]:45s} "
              f"{str(m['f1_macro']):>10} "
              f"{str(m['auc_pr']):>10} "
              f"{str(m['mcc']):>10} "
              f"{str(m.get('backdoor_recall','-')):>12} "
              f"{str(m.get('mitm_recall','-')):>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="compare")
    parser.add_argument("--mode",  type=str, default="standalone",
                        choices=["standalone", "hybrid"])
    args = parser.parse_args()

    if args.model == "compare":
        compare_all_models()
    else:
        evaluate_sklearn_model(args.model, args.mode)
