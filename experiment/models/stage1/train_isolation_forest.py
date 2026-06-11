"""
═══════════════════════════════════════════════════════════════════
ISOLATION FOREST — Stage-1 Anomaly Detector
═══════════════════════════════════════════════════════════════════

THÉORIE
────────
Isolation Forest isole les anomalies en construisant des arbres de
décision aléatoires. L'idée clé : les points anormaux sont plus
FACILES à isoler (moins de coupures nécessaires) que les points
normaux qui sont densément groupés.

Score d'anomalie = hauteur moyenne d'un point dans les arbres.
  → Hauteur faible = anomalie (isolé rapidement)
  → Hauteur élevée = normal (nécessite beaucoup de coupures)

Formule du score :
  s(x, n) = 2^{ -E[h(x)] / c(n) }

  où h(x) = profondeur du point x dans un arbre
     c(n) = profondeur moyenne d'un arbre construit sur n points
     s → 1 : anomalie certaine
     s → 0.5 : frontière
     s → 0 : normal

POURQUOI L'UTILISER EN ICS/IIoT
─────────────────────────────────
1. Ne nécessite PAS de données labellisées (non supervisé)
2. Rapide sur de grands datasets (O(n log n))
3. Robuste aux espaces de grande dimension
4. Efficace pour les attaques de reconnaissance (scanning) qui
   s'isolent facilement des patterns normaux Modbus/TCP

FORCES
  ✓ Très rapide (parallélisable)
  ✓ Pas de normalisation requise
  ✓ Gère bien les données de haute dimension
  ✓ Peu d'hyperparamètres critiques

FAIBLESSES
  ✗ Ne modélise pas la densité locale → moins bon que LOF pour
    les clusters denses de trafic normal
  ✗ Sensible au paramètre contamination
  ✗ Peut rater les anomalies masquées dans des clusters denses

COMMANDE D'EXÉCUTION
  python train_isolation_forest.py
  Durée : ~5-10 secondes sur TON_IoT
  RAM : ~500 MB
  GPU : non requis

SORTIES
  results/models/isolation_forest.pkl
  results/metrics/if_metrics.json
  results/figures/if_roc.png
  results/figures/if_pr.png
  results/figures/if_threshold.png
═══════════════════════════════════════════════════════════════════
"""

import sys, os, json, time, joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve, auc)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ton_iot_config import *

os.makedirs(os.path.join(RESULTS_DIR, "saved_models"), exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMÈTRES — Isolation Forest
# ──────────────────────────────────────────────────────────────────────────────
#
#  n_estimators (défaut=100, recommandé=200-300)
#    → Nombre d'arbres dans la forêt
#    → Plus grand = plus stable mais plus lent
#    → Au-delà de 200, le gain est marginal
#    → IMPACT : précision vs vitesse
#
#  max_samples (défaut='auto' = min(256, n_samples))
#    → Taille du sous-échantillon pour construire chaque arbre
#    → 'auto' = 256 points par arbre
#    → Plus grand = plus précis mais plus lent
#    → IMPACT : vitesse vs qualité de l'isolement
#
#  contamination (défaut='auto' = 0.1)
#    → Proportion estimée d'anomalies dans les données d'entraînement
#    → CRITIQUE : si trop élevé → trop de faux positifs
#    → Pour Stage-1 entraîné sur données normales : mettre très BAS (0.01-0.05)
#    → IMPACT : directement lié au seuil de décision
#
#  max_features (défaut=1.0)
#    → Proportion de features utilisées pour chaque arbre
#    → 0.8 = 80% des features → plus de diversité entre arbres
#    → IMPACT : robustesse vs vitesse
#
#  random_state
#    → Graine aléatoire pour reproductibilité
#    → TOUJOURS fixer pour des résultats reproductibles
#
# ──────────────────────────────────────────────────────────────────────────────

HYPERPARAMS = {
    "n_estimators": 200,      # 200 arbres — bon compromis précision/vitesse
    "max_samples":  256,      # 256 points par arbre — valeur standard
    "contamination": 0.01,    # 1% d'anomalies — faible car entraîné sur Normal
    "max_features":  0.8,     # 80% des features — diversité entre arbres
    "random_state":  42,      # reproductibilité
    "n_jobs": -1,             # tous les CPU
}


def train():
    print("=" * 60)
    print("ISOLATION FOREST — Stage-1 Anomaly Detection")
    print("=" * 60)

    # ── Chargement des données ───────────────────────────────────────────────
    # On charge uniquement les données NORMALES pour l'entraînement.
    # C'est le principe du one-class learning : apprendre le comportement normal,
    # puis détecter tout ce qui s'en écarte.
    print("\n1. Chargement des données...")
    X_train_mm = np.load(os.path.join(METRICS_DIR, "X_train_mm.npy"))
    X_test_mm  = np.load(os.path.join(METRICS_DIR, "X_test_mm.npy"))
    y_train    = np.load(os.path.join(METRICS_DIR, "y_train.npy"))
    y_test     = np.load(os.path.join(METRICS_DIR, "y_test.npy"))

    with open(os.path.join(METRICS_DIR, "label_classes.json")) as f:
        class_names = json.load(f)["classes"]
    normal_id = class_names.index("normal")

    X_normal = X_train_mm[y_train == normal_id]
    print(f"   Données normales d'entraînement : {len(X_normal):,} échantillons")
    print(f"   Features : {X_normal.shape[1]}")

    # ── Préparation des features ─────────────────────────────────────────────
    # Isolation Forest n'a PAS besoin de normalisation (basé sur comparaisons).
    # On utilise MinMaxScaler ici pour la cohérence avec les autres modèles.
    # Pour IF seul, les données brutes fonctionnent aussi bien.

    # ── Entraînement ────────────────────────────────────────────────────────
    print("\n2. Entraînement...")
    print(f"   Hyperparamètres : {HYPERPARAMS}")
    t0 = time.time()
    model = IsolationForest(**HYPERPARAMS)
    model.fit(X_normal)
    t_fit = time.time() - t0
    print(f"   Durée d'entraînement : {t_fit:.1f}s")

    # ── Scores d'anomalie ────────────────────────────────────────────────────
    # decision_function retourne : score élevé = NORMAL, score bas = ANOMALIE
    # On inverse (-) pour avoir : score élevé = ANOMALIE (convention usuelle)
    print("\n3. Calcul des scores d'anomalie...")
    scores_test  = -model.decision_function(X_test_mm)
    y_binary     = (y_test != normal_id).astype(int)   # 1=attaque, 0=normal

    # ── Calibrage du seuil θ ─────────────────────────────────────────────────
    # Le seuil θ détermine à partir de quel score on considère un flux comme
    # anormal. On le calibre pour obtenir Recall ≥ 0.90 sur les attaques.
    #
    # Pourquoi Recall ≥ 0.90 ?
    # En sécurité ICS, manquer une attaque (faux négatif) est plus grave
    # qu'avoir un faux positif. On accepte un certain FPR pour garantir
    # un recall élevé.
    print("\n4. Calibrage du seuil θ...")
    X_val_mm  = np.load(os.path.join(METRICS_DIR, "X_val_mm.npy"))
    y_val     = np.load(os.path.join(METRICS_DIR, "y_val.npy"))
    sc_val    = -model.decision_function(X_val_mm)
    y_val_bin = (y_val != normal_id).astype(int)

    prec_v, rec_v, thr_v = precision_recall_curve(y_val_bin, sc_val)
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
    recall_at_theta    = tp / (tp + fn + 1e-9)
    fpr_at_theta       = fp / (fp + tn + 1e-9)
    precision_at_theta = tp / (tp + fp + 1e-9)

    print(f"   θ = {theta:.4f}")
    print(f"   Recall  = {recall_at_theta:.4f}  (cible ≥ {STAGE1_TARGET_RECALL})")
    print(f"   FPR     = {fpr_at_theta:.4f}  (cible ≤ {STAGE1_MAX_FPR})")
    print(f"   Précision = {precision_at_theta:.4f}")

    # ── Métriques globales ────────────────────────────────────────────────────
    auc_roc = roc_auc_score(y_binary, scores_test)
    auc_pr  = average_precision_score(y_binary, scores_test)
    print(f"\n5. Métriques globales :")
    print(f"   AUC-ROC = {auc_roc:.4f}")
    print(f"   AUC-PR  = {auc_pr:.4f}")

    # ── Enrichissement ────────────────────────────────────────────────────────
    flagged       = scores_test >= theta
    stealthy_ids  = [class_names.index(c) for c in STEALTHY_CLASSES
                     if c in class_names]
    r_before = np.isin(y_test, stealthy_ids).mean()
    r_after  = np.isin(y_test[flagged], stealthy_ids).mean()
    factor   = r_after / (r_before + 1e-12)
    print(f"\n6. Enrichissement des classes furtives :")
    print(f"   Avant filtrage : {r_before:.4%}")
    print(f"   Après filtrage : {r_after:.4%}")
    print(f"   Facteur        : x{factor:.2f}")

    # ── Sauvegarde du modèle ─────────────────────────────────────────────────
    model_path = os.path.join(RESULTS_DIR, "saved_models", "isolation_forest.pkl")
    joblib.dump({"model": model, "theta": theta,
                  "class_names": class_names}, model_path)
    print(f"\nModèle sauvegardé : {model_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    fpr_c, tpr_c, _ = roc_curve(y_binary, scores_test)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    ax1.plot(fpr_c, tpr_c, color='#e74c3c', linewidth=2,
             label=f"IF (AUC={auc_roc:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax1.axvline(fpr_at_theta, color='gray', linestyle=':',
                label=f"θ sélectionné (FPR={fpr_at_theta:.3f})")
    ax1.set(xlabel="FPR", ylabel="TPR",
            title="Isolation Forest — Courbe ROC (TON_IoT)",
            xlim=[0, 1], ylim=[0, 1.02])
    ax1.legend()

    # Precision-Recall
    p_c, r_c, _ = precision_recall_curve(y_binary, scores_test)
    ax2.plot(r_c, p_c, color='#3498db', linewidth=2,
             label=f"IF (AP={auc_pr:.3f})")
    ax2.axvline(recall_at_theta, color='red', linestyle='--',
                label=f"θ sélectionné (Recall={recall_at_theta:.3f})")
    ax2.set(xlabel="Recall", ylabel="Precision",
            title="Isolation Forest — Courbe PR (TON_IoT)")
    ax2.legend()

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "if_roc_pr.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure sauvegardée : {fig_path}")

    # ── JSON métriques ────────────────────────────────────────────────────────
    metrics = {
        "model": "Isolation Forest",
        "hyperparams": HYPERPARAMS,
        "theta": theta,
        "recall": round(recall_at_theta, 4),
        "fpr": round(fpr_at_theta, 4),
        "precision": round(precision_at_theta, 4),
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "enrichment_factor": round(factor, 4),
        "fit_time_s": round(t_fit, 1),
        "n_normal_train": len(X_normal),
        "n_test": len(y_test),
    }
    metrics_path = os.path.join(METRICS_DIR, "if_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Métriques sauvegardées : {metrics_path}")

    return metrics


# ══════════════════════════════════════════════════════════════════
# COMMENT MODIFIER LES HYPERPARAMÈTRES
# ══════════════════════════════════════════════════════════════════
#
# Si le Recall est trop bas (< 0.90) :
#   → Réduire contamination (ex: 0.005)
#   → Augmenter n_estimators (ex: 300)
#
# Si le FPR est trop élevé (> 0.15) :
#   → Augmenter contamination (ex: 0.05)
#   → Utiliser max_features=1.0
#
# Si l'entraînement est trop lent :
#   → Réduire n_estimators (100 suffit pour les données ICS)
#   → Réduire max_samples (128 est souvent suffisant)
#
# Pour tuner automatiquement :
#   >>> import optuna
#   >>> def objective(trial):
#   ...     params = {
#   ...         "n_estimators": trial.suggest_int("n_est", 50, 300),
#   ...         "contamination": trial.suggest_float("cont", 0.001, 0.1),
#   ...         "max_features": trial.suggest_float("mf", 0.5, 1.0),
#   ...     }
#   ...     model = IsolationForest(**params, random_state=42)
#   ...     model.fit(X_normal)
#   ...     scores = -model.decision_function(X_val)
#   ...     return roc_auc_score(y_val_binary, scores)
#   >>> study = optuna.create_study(direction='maximize')
#   >>> study.optimize(objective, n_trials=30)
# ══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    metrics = train()
    print("\n" + "=" * 60)
    print("RÉSUMÉ ISOLATION FOREST")
    print("=" * 60)
    print(f"  Recall  : {metrics['recall']:.4f}  (cible ≥ 0.90)")
    print(f"  FPR     : {metrics['fpr']:.4f}  (cible ≤ 0.15)")
    print(f"  AUC-ROC : {metrics['auc_roc']:.4f}")
    print(f"  Enrichissement : x{metrics['enrichment_factor']:.2f}")
