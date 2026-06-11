"""
═══════════════════════════════════════════════════════════════════
XGBOOST — Stage-2 Supervised Multi-Class Classifier
═══════════════════════════════════════════════════════════════════

THÉORIE
────────
XGBoost (eXtreme Gradient Boosting) construit séquentiellement un
ensemble d'arbres de décision. Chaque nouvel arbre corrige les
erreurs des arbres précédents en minimisant un objectif régularisé.

Principe du Gradient Boosting :
  F₀(x) = constante (moyenne des labels)
  Pour t = 1, 2, ..., T :
    Calculer les gradients gᵢ et hessiennes hᵢ par rapport à F_{t-1}
    Entraîner l'arbre hₜ sur les gradients/hessiennes
    Mettre à jour : Fₜ(x) = F_{t-1}(x) + η · hₜ(x)

  η = learning_rate (taux d'apprentissage)

Objectif XGBoost (avec régularisation L1 + L2) :
  L(t) = Σ l(yᵢ, F_{t-1}(xᵢ) + hₜ(xᵢ)) + Ω(hₜ)

  où Ω(h) = γT + (λ/2)Σwⱼ²  (pénalité sur le nombre de feuilles T
                               et les poids wⱼ des feuilles)

Pour la classification multi-classe :
  Softmax sur les scores de chaque classe
  Loss = Cross-Entropy (log-loss)

POURQUOI EN ICS/IIoT
──────────────────────
XGBoost est le modèle de référence pour les données tabulaires en
intrusion detection. Performant, rapide, interprétable via SHAP,
et robuste au déséquilibre de classes grâce à scale_pos_weight.

FORCES
  ✓ Très précis sur les données tabulaires (souvent meilleur modèle
    sur les benchmarks IoT — voir papier d'Enzo)
  ✓ Rapide grâce à l'implémentation optimisée (histogrammes)
  ✓ Robuste aux features redondantes/inutiles
  ✓ Interprétable via SHAP
  ✓ Gère nativement les valeurs manquantes

FAIBLESSES
  ✗ Moins adapté aux séries temporelles longues que LSTM/Transformer
  ✗ Peut overfitter si max_depth et n_estimators trop grands
  ✗ Temps d'entraînement élevé sur très grands datasets (> 500K)
  ✗ N'exploite pas directement la structure temporelle des flux ICS

COMMANDE D'EXÉCUTION
  python train_xgboost.py
  Durée : ~50-200 secondes selon la taille des données
  RAM : ~1-2 GB
  GPU : optionnel (tree_method='gpu_hist' pour GPU)

SORTIES
  results/models/xgboost.pkl
  results/metrics/xgboost_metrics.json
  results/figures/xgb_cm.png / xgb_roc_pr.png
  results/figures/xgb_shap.png
═══════════════════════════════════════════════════════════════════
"""

import sys, os, json, time, joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ton_iot_config import *

os.makedirs(os.path.join(RESULTS_DIR, "saved_models"), exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMÈTRES — XGBoost
# ──────────────────────────────────────────────────────────────────────────────
#
#  n_estimators (défaut=100, recommandé=100-500)
#    → Nombre d'arbres (= nombre d'itérations de boosting)
#    → Plus d'arbres = meilleur ajustement mais risque d'overfitting
#    → Utiliser avec early_stopping pour trouver le nombre optimal
#    → IMPACT : performance vs vitesse
#
#  max_depth (défaut=6, recommandé=3-8)
#    → Profondeur maximale de chaque arbre
#    → Valeur élevée → arbres plus complexes → risque d'overfitting
#    → Valeur faible → underfitting → modèle trop simple
#    → Pour ICS/IoT : 4-6 suffisent généralement
#    → IMPACT : complexité du modèle
#
#  learning_rate / eta (défaut=0.3, recommandé=0.01-0.3)
#    → Taux d'apprentissage : réduit la contribution de chaque arbre
#    → Petit learning_rate → meilleure généralisation mais besoin de
#      plus d'arbres (n_estimators plus grand)
#    → RÈGLE EMPIRIQUE : learning_rate × n_estimators ≈ constante
#      ex: 0.1 × 300 ≈ 0.01 × 3000 (performances similaires)
#    → IMPACT : stabilité vs vitesse de convergence
#
#  subsample (défaut=1.0, recommandé=0.6-1.0)
#    → Fraction des échantillons utilisés par arbre (stochastic boosting)
#    → Réduit l'overfitting en introduisant de la variance
#    → IMPACT : régularisation vs performance
#
#  colsample_bytree (défaut=1.0, recommandé=0.5-1.0)
#    → Fraction des features utilisées par arbre (comme random forests)
#    → IMPACT : régularisation, robustesse aux features bruitées
#
#  min_child_weight (défaut=1)
#    → Poids minimum des instances dans un nœud enfant
#    → Valeur élevée → arbres plus conservatives → moins d'overfitting
#    → IMPACT : régularisation sur les nœuds de l'arbre
#
#  reg_alpha (L1, défaut=0) et reg_lambda (L2, défaut=1)
#    → Régularisation L1 (Lasso) et L2 (Ridge) sur les poids des feuilles
#    → Utile si beaucoup de features non pertinentes
#    → IMPACT : parcimonie et robustesse
#
#  scale_pos_weight (pour classification binaire avec déséquilibre)
#    → Pour multi-classe avec déséquilibre : utiliser class_weight (non natif)
#    → Alternative : prétraitement ADASYN/SMOTE
#
# ──────────────────────────────────────────────────────────────────────────────

HYPERPARAMS = {
    "n_estimators":    200,
    "max_depth":       6,
    "learning_rate":   0.1,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "min_child_weight":3,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "eval_metric":    "mlogloss",
    "verbosity":       0,
    "random_state":    42,
    "n_jobs":         -1,
}


def train(mode="standalone"):
    """
    mode='standalone' : entraîne sur toutes les données
    mode='hybrid'     : entraîne uniquement sur les données flaggées par Stage-1
    """
    print("=" * 60)
    print(f"XGBOOST — Stage-2 Classification (mode: {mode})")
    print("=" * 60)

    print("\n1. Chargement des données...")
    with open(os.path.join(METRICS_DIR, "label_classes.json")) as f:
        class_names = json.load(f)["classes"]
    n_cls = len(class_names)

    if mode == "standalone":
        X_train = np.load(os.path.join(METRICS_DIR, "X_train_std.npy"))
        y_train = np.load(os.path.join(METRICS_DIR, "y_train.npy"))
        X_test  = np.load(os.path.join(METRICS_DIR, "X_test_std.npy"))
        y_test  = np.load(os.path.join(METRICS_DIR, "y_test.npy"))
    else:  # hybrid
        flag_tr = np.load(os.path.join(METRICS_DIR, "s1_flag_train.npy"))
        flag_ts = np.load(os.path.join(METRICS_DIR, "s1_flag_test.npy"))
        sc_tr   = np.load(os.path.join(METRICS_DIR, "s1_score_train.npy"))
        sc_ts   = np.load(os.path.join(METRICS_DIR, "s1_score_test.npy"))
        X_tr    = np.load(os.path.join(METRICS_DIR, "X_train_std.npy"))
        X_ts    = np.load(os.path.join(METRICS_DIR, "X_test_std.npy"))
        y_tr    = np.load(os.path.join(METRICS_DIR, "y_train.npy"))
        y_ts    = np.load(os.path.join(METRICS_DIR, "y_test.npy"))
        # Append anomaly score as additional feature
        X_train = np.column_stack([X_tr[flag_tr], sc_tr[flag_tr].reshape(-1,1)])
        y_train = y_tr[flag_tr]
        X_test  = np.column_stack([X_ts[flag_ts], sc_ts[flag_ts].reshape(-1,1)])
        y_test  = y_ts[flag_ts]

    print(f"   Train : {X_train.shape} | Test : {X_test.shape}")
    print(f"   Classes : {n_cls} ({class_names})")

    # ── Statistiques des classes ─────────────────────────────────────────────
    # Identifier les classes rares pour surveiller leurs métriques
    unique, counts = np.unique(y_train, return_counts=True)
    print("\n   Distribution des classes (train) :")
    for cls_id, cnt in zip(unique, counts):
        pct = cnt / len(y_train) * 100
        flag = " ← RARE" if pct < 1.0 else ""
        print(f"   {class_names[cls_id]:12s} : {cnt:7,} ({pct:5.2f}%){flag}")

    # ── Entraînement ─────────────────────────────────────────────────────────
    print(f"\n2. Entraînement XGBoost...")
    print(f"   Hyperparamètres : {HYPERPARAMS}")

    # Note sur l'entraînement avec early stopping :
    # Si vous voulez utiliser early stopping (recommandé pour éviter l'overfitting) :
    #   model.fit(X_train, y_train,
    #             eval_set=[(X_val, y_val)],
    #             early_stopping_rounds=20,
    #             verbose=False)
    # Ici on utilise le nombre d'estimateurs fixe pour la reproductibilité.

    t0 = time.time()
    model = XGBClassifier(**HYPERPARAMS)
    model.fit(X_train, y_train)
    t_fit = time.time() - t0
    print(f"   Durée : {t_fit:.1f}s")

    # ── Prédictions ──────────────────────────────────────────────────────────
    t1 = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    lat_ms = (time.time() - t1) / len(X_test) * 1000

    # ── Métriques ────────────────────────────────────────────────────────────
    print("\n3. Métriques d'évaluation...")
    f1m  = f1_score(y_test, y_pred, average='macro',    zero_division=0)
    f1w  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    acc  = accuracy_score(y_test, y_pred)
    mcc  = matthews_corrcoef(y_test, y_pred)

    y_bin = np.eye(n_cls)[y_test]
    auc_roc = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
    auc_pr  = average_precision_score(y_bin, y_prob, average='macro')

    rpt = classification_report(y_test, y_pred,
                                  target_names=class_names, zero_division=0)
    print(rpt)
    print(f"   AUC-ROC   : {auc_roc:.4f}")
    print(f"   AUC-PR    : {auc_pr:.4f}")
    print(f"   MCC       : {mcc:.4f}")
    print(f"   Latence   : {lat_ms:.4f} ms/flux")

    # ── Stealthy class recalls ────────────────────────────────────────────────
    rpt_dict = classification_report(y_test, y_pred, target_names=class_names,
                                      output_dict=True, zero_division=0)
    print("\n   Recall des classes furtives :")
    for cls in STEALTHY_CLASSES:
        if cls in rpt_dict:
            r = rpt_dict[cls]['recall']
            s = rpt_dict[cls]['support']
            print(f"   {cls:12s} : Recall={r:.4f}  (support={s})")

    # ── Matrice de confusion ──────────────────────────────────────────────────
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.3, vmin=0, vmax=1, ax=ax)
    ax.set_title(f"XGBoost ({mode}) — Matrice de confusion normalisée")
    ax.set_ylabel("Vraie classe")
    ax.set_xlabel("Classe prédite")
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"xgb_{mode}_cm.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── ROC curves ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, cls in enumerate(class_names):
        fpr_c, tpr_c, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_a = auc(fpr_c, tpr_c)
        style = '-' if cls in STEALTHY_CLASSES else '--'
        ax.plot(fpr_c, tpr_c, color=PALETTE[i % len(PALETTE)],
                linestyle=style, linewidth=1.5,
                label=f"{cls} (AUC={roc_a:.2f})")
    ax.plot([0,1],[0,1],'k--',alpha=0.4,lw=1)
    ax.set(xlabel="FPR", ylabel="TPR",
           title=f"XGBoost ({mode}) — Courbes ROC par classe",
           xlim=[0,1], ylim=[0,1.02])
    ax.legend(fontsize=7, loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"xgb_{mode}_roc.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    model_path = os.path.join(RESULTS_DIR, "saved_models", f"xgboost_{mode}.pkl")
    joblib.dump({"model": model, "class_names": class_names,
                  "hyperparams": HYPERPARAMS, "mode": mode}, model_path)

    metrics = {
        "model": f"XGBoost ({mode})",
        "hyperparams": HYPERPARAMS,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1m, 4),
        "f1_weighted": round(f1w, 4),
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "mcc": round(mcc, 4),
        "stealthy_recalls": {
            cls: round(rpt_dict[cls]['recall'], 4)
            for cls in STEALTHY_CLASSES if cls in rpt_dict
        },
        "per_class": {
            c: {k: round(v, 4) for k, v in rpt_dict[c].items()
                if k in ('precision','recall','f1-score','support')}
            for c in class_names if c in rpt_dict
        },
        "fit_time_s": round(t_fit, 1),
        "latency_ms": round(lat_ms, 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    with open(os.path.join(METRICS_DIR, f"xgboost_{mode}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nXGBoost ({mode}) terminé.")
    print(f"  F1-macro={f1m:.4f} | AUC-PR={auc_pr:.4f} | MCC={mcc:.4f}")
    return metrics


# ══════════════════════════════════════════════════════════════════
# COMPRENDRE LES RÉSULTATS
# ══════════════════════════════════════════════════════════════════
#
# F1-macro vs F1-weighted :
#   F1-macro = moyenne ÉGALE sur toutes les classes
#     → Pénalise les mauvaises performances sur classes rares (mitm)
#     → C'est la métrique principale pour ce projet
#   F1-weighted = moyenne PONDÉRÉE par le support
#     → Proche de l'accuracy globale
#     → Masque les mauvaises perfs sur classes minoritaires
#
# MCC (Matthews Correlation Coefficient) :
#   → Entre -1 et +1
#   → 0 = classification aléatoire
#   → +1 = classification parfaite
#   → Robuste même si les classes sont déséquilibrées
#   → Meilleure métrique que l'accuracy en cas de déséquilibre
#
# Si une classe a Recall=0 → le modèle n'arrive jamais à la prédire
#   → Vérifier si cette classe est dans l'ensemble d'entraînement
#   → Essayer ADASYN ou augmenter le poids (scale_pos_weight)
# ══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standalone", "hybrid"],
                        default="standalone")
    args = parser.parse_args()
    train(mode=args.mode)
