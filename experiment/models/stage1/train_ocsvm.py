"""
═══════════════════════════════════════════════════════════════════
ONE-CLASS SVM (OCSVM) — Stage-1 Anomaly Detector
═══════════════════════════════════════════════════════════════════

THÉORIE
────────
L'OCSVM apprend à enfermer les données normales dans une hypersphère
dans un espace de features transformé par un noyau (kernel).

Formulation du problème :
  min   (1/2)||w||² + (1/νn)Σ ξᵢ - ρ
  s.t.  w·φ(xᵢ) ≥ ρ - ξᵢ,  ξᵢ ≥ 0

  où φ(x) = projection dans l'espace du noyau
     ρ    = rayon de la sphère (à optimiser)
     ν    = borne supérieure sur la fraction d'anomalies
     ξᵢ  = variable d'écart (slack)

Avec noyau RBF (Radial Basis Function) :
  K(x, x') = exp(-γ||x - x'||²)

  γ trop grand → overfitting (trop proche des données normales)
  γ trop petit → underfitting (sphère trop large, tout est normal)

Score de décision :
  f(x) = sgn( Σ αᵢ K(xᵢ, x) - ρ )
  score = Σ αᵢ K(xᵢ, x) - ρ
  > 0 : NORMAL (dans la sphère)
  < 0 : ANOMALIE (hors de la sphère)

POURQUOI EN ICS/IIoT
──────────────────────
Le trafic normal ICS (Modbus, DNP3) forme des clusters denses et
prévisibles. Le noyau RBF est idéal pour capturer cette structure
géométrique non-linéaire. L'OCSVM avec γ bien calibré peut apprendre
la "frontière" du comportement normal avec grande précision.

FORCES
  ✓ Excellent avec clusters denses (trafic ICS structuré)
  ✓ Flexible grâce au choix du noyau
  ✓ Théoriquement fondé (maximisation de marge)
  ✓ Robuste au bruit si ν est bien calibré

FAIBLESSES
  ✗ Lent sur de grands datasets (O(n²) pour l'entraînement)
  ✗ Très sensible à γ et ν — tuning difficile
  ✗ Pas de sortie probabiliste directe
  ✗ Mal adapté aux données de très haute dimension sans PCA

COMMANDE D'EXÉCUTION
  python train_ocsvm.py
  Durée : 15-30 secondes sur TON_IoT (données normales seulement)
  RAM : ~1 GB
  GPU : non requis

SORTIES
  results/models/ocsvm.pkl
  results/metrics/ocsvm_metrics.json
  results/figures/ocsvm_roc_pr.png
═══════════════════════════════════════════════════════════════════
"""

import sys, os, json, time, joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ton_iot_config import *

os.makedirs(os.path.join(RESULTS_DIR, "saved_models"), exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMÈTRES — One-Class SVM
# ──────────────────────────────────────────────────────────────────────────────
#
#  kernel (défaut='rbf', alternatives: 'linear', 'poly', 'sigmoid')
#    → 'rbf' : meilleur pour données non-linéaires (recommandé ICS)
#    → 'linear' : plus rapide, données linéairement séparables seulement
#    → IMPACT : capacité à capturer des frontières non-linéaires
#
#  nu (défaut=0.5, intervalle: [0, 1])
#    → Borne supérieure sur le TAUX D'ERREUR d'entraînement
#    → ET borne inférieure sur le TAUX DE VECTEURS DE SUPPORT
#    → nu=0.05 → accepte que 5% des données normales soient mal classées
#    → Petit nu → frontière serrée → moins de faux positifs mais moins
#      robuste aux données normales variées
#    → Grand nu → frontière large → plus de faux positifs mais meilleur recall
#    → IMPACT : principal contrôle sur sensibilité/spécificité
#    → CONSEIL : commencer avec nu=0.01 à 0.1, tuner sur validation
#
#  gamma ('scale', 'auto', ou float)
#    → 'scale' = 1/(n_features * Var(X)) — recommandé par défaut
#    → 'auto'  = 1/n_features
#    → float   = valeur manuelle (ex: 0.001)
#    → Contrôle le RAYON d'influence de chaque point d'entraînement
#    → Petit γ → grande zone d'influence → frontière douce
#    → Grand γ → petite zone d'influence → frontière accidentée (overfitting)
#    → IMPACT : décisif pour la qualité du modèle
#
#  TUNING RECOMMANDÉ :
#    1. Commencer par kernel='rbf', gamma='scale', nu=0.05
#    2. Si FPR trop élevé → réduire nu (0.01)
#    3. Si Recall trop bas → augmenter nu (0.1)
#    4. Si performances insuffisantes → essayer gamma manuel avec grille
#       gamma ∈ {0.001, 0.01, 0.1, 1.0, 10.0}
#
# ──────────────────────────────────────────────────────────────────────────────

HYPERPARAMS = {
    "kernel": "rbf",     # noyau gaussien — meilleur pour ICS
    "nu":     0.05,      # 5% d'anomalies autorisées — frontière modérément serrée
    "gamma":  "scale",   # calibrage automatique basé sur la variance
}


def train():
    print("=" * 60)
    print("ONE-CLASS SVM (RBF) — Stage-1 Anomaly Detection")
    print("=" * 60)

    print("\n1. Chargement des données...")
    X_train_mm = np.load(os.path.join(METRICS_DIR, "X_train_mm.npy"))
    X_test_mm  = np.load(os.path.join(METRICS_DIR, "X_test_mm.npy"))
    X_val_mm   = np.load(os.path.join(METRICS_DIR, "X_val_mm.npy"))
    y_train    = np.load(os.path.join(METRICS_DIR, "y_train.npy"))
    y_test     = np.load(os.path.join(METRICS_DIR, "y_test.npy"))
    y_val      = np.load(os.path.join(METRICS_DIR, "y_val.npy"))

    with open(os.path.join(METRICS_DIR, "label_classes.json")) as f:
        class_names = json.load(f)["classes"]
    normal_id = class_names.index("normal")

    X_normal = X_train_mm[y_train == normal_id]
    print(f"   Normal train : {len(X_normal):,} | Test : {len(y_test):,}")

    # ── NOTE : OCSVM est lent sur de grands datasets ───────────────────────
    # Si X_normal > 50K, sous-échantillonner pour accélérer :
    if len(X_normal) > 50_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_normal), 50_000, replace=False)
        X_train_ocsvm = X_normal[idx]
        print(f"   Sous-échantillonnage OCSVM : {len(X_train_ocsvm):,}")
    else:
        X_train_ocsvm = X_normal

    # ── Entraînement ────────────────────────────────────────────────────────
    print(f"\n2. Entraînement... (hyperparams: {HYPERPARAMS})")
    t0 = time.time()
    model = OneClassSVM(**HYPERPARAMS)
    model.fit(X_train_ocsvm)
    t_fit = time.time() - t0
    print(f"   Durée : {t_fit:.1f}s")

    # ── Scores ──────────────────────────────────────────────────────────────
    # decision_function > 0 = NORMAL, < 0 = ANOMALIE
    # On nège pour avoir : grand score = anomalie
    scores_test = -model.decision_function(X_test_mm)
    y_binary    = (y_test != normal_id).astype(int)

    # ── Calibrage θ ─────────────────────────────────────────────────────────
    print("\n3. Calibrage du seuil θ...")
    sc_val    = -model.decision_function(X_val_mm)
    y_val_bin = (y_val != normal_id).astype(int)

    _, rec_v, thr_v = precision_recall_curve(y_val_bin, sc_val)
    theta = None
    for i in range(len(thr_v) - 1, -1, -1):
        if rec_v[i] >= STAGE1_TARGET_RECALL:
            theta = float(thr_v[i])
            break
    if theta is None:
        theta = float(np.percentile(sc_val[y_val_bin == 0], 95))

    tp = int(((y_binary == 1) & (scores_test >= theta)).sum())
    fp = int(((y_binary == 0) & (scores_test >= theta)).sum())
    fn = int(((y_binary == 1) & (scores_test < theta)).sum())
    tn = int(((y_binary == 0) & (scores_test < theta)).sum())
    recall_t    = tp / (tp + fn + 1e-9)
    fpr_t       = fp / (fp + tn + 1e-9)
    precision_t = tp / (tp + fp + 1e-9)

    print(f"   θ={theta:.4f} | Recall={recall_t:.4f} | FPR={fpr_t:.4f}")

    auc_roc = roc_auc_score(y_binary, scores_test)
    auc_pr  = average_precision_score(y_binary, scores_test)
    print(f"\n4. AUC-ROC={auc_roc:.4f} | AUC-PR={auc_pr:.4f}")

    # ── Enrichissement ────────────────────────────────────────────────────────
    flagged      = scores_test >= theta
    stealthy_ids = [class_names.index(c) for c in STEALTHY_CLASSES
                    if c in class_names]
    r_before = np.isin(y_test, stealthy_ids).mean()
    r_after  = np.isin(y_test[flagged], stealthy_ids).mean()
    factor   = r_after / (r_before + 1e-12)
    print(f"\n5. Enrichissement : {r_before:.4%} → {r_after:.4%} (x{factor:.2f})")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    model_path = os.path.join(RESULTS_DIR, "saved_models", "ocsvm.pkl")
    joblib.dump({"model": model, "theta": theta,
                  "class_names": class_names,
                  "n_support_vectors": model.n_support_vectors_,
                  "hyperparams": HYPERPARAMS}, model_path)

    metrics = {
        "model": "One-Class SVM (RBF)",
        "hyperparams": HYPERPARAMS,
        "theta": theta,
        "recall": round(recall_t, 4),
        "fpr": round(fpr_t, 4),
        "precision": round(precision_t, 4),
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "n_support_vectors": int(model.n_support_vectors_),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "enrichment_factor": round(factor, 4),
        "fit_time_s": round(t_fit, 1),
    }
    with open(os.path.join(METRICS_DIR, "ocsvm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fpr_c, tpr_c, _ = roc_curve(y_binary, scores_test)
    ax1.plot(fpr_c, tpr_c, color='#2ecc71', lw=2,
             label=f"OCSVM (AUC={auc_roc:.3f})")
    ax1.plot([0,1],[0,1],'k--',alpha=0.4)
    ax1.set(xlabel="FPR", ylabel="TPR",
            title="OCSVM — ROC (TON_IoT)", xlim=[0,1], ylim=[0,1.02])
    ax1.legend()
    p_c, r_c, _ = precision_recall_curve(y_binary, scores_test)
    ax2.plot(r_c, p_c, color='#9b59b6', lw=2,
             label=f"OCSVM (AP={auc_pr:.3f})")
    ax2.set(xlabel="Recall", ylabel="Precision",
            title="OCSVM — PR (TON_IoT)")
    ax2.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ocsvm_roc_pr.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nOCSVM terminé. Modèle : {model_path}")
    return metrics


# ══════════════════════════════════════════════════════════════════
# COMPRENDRE LES VECTEURS DE SUPPORT
# ══════════════════════════════════════════════════════════════════
#
# model.n_support_vectors_ indique combien de points d'entraînement
# définissent la frontière de la sphère.
#
# Ratio = n_support_vectors / n_train_normal
#   > 0.5 : beaucoup de vecteurs → frontière complexe → risque d'overfitting
#   < 0.1 : peu de vecteurs → frontière simple → peut-être underfitting
#
# Si model.decision_function(X_new) > 0 : NORMAL
# Si model.decision_function(X_new) < 0 : ANOMALIE
#
# POUR DÉPLOIEMENT EN PRODUCTION :
#   → Sauvegarder le modèle avec joblib
#   → Appliquer le MÊME scaler (MinMax ou Standard) que celui d'entraînement
#   → Utiliser le même θ calibré sur validation
# ══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    m = train()
    print(f"\nOCSVM | Recall={m['recall']} | FPR={m['fpr']} "
          f"| AUC-ROC={m['auc_roc']} | x{m['enrichment_factor']}")
