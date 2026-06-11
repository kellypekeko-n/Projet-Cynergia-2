"""
═══════════════════════════════════════════════════════════════════
LIGHTGBM — Stage-2 Supervised Multi-Class Classifier
═══════════════════════════════════════════════════════════════════

THÉORIE
────────
LightGBM est une implémentation du Gradient Boosting optimisée par
deux innovations majeures :

1. GOSS (Gradient-based One-Side Sampling) :
   → Ne conserve que les instances avec de GRANDS gradients
     (celles qui ont le plus besoin d'être apprises)
   → Sous-échantillonne aléatoirement les instances à petit gradient
   → Réduit le nombre d'instances de ~80% sans perte significative

2. EFB (Exclusive Feature Bundling) :
   → Regroupe les features mutuellement exclusives (un seul non-zéro)
   → Réduit le nombre effectif de features
   → Accélère la construction des arbres

Construction des arbres :
   XGBoost/standard : croissance Level-wise (par niveau)
   LightGBM          : croissance Leaf-wise (par feuille)

   Level-wise :    Leaf-wise :
   ┌───────────┐   ┌─────────────────┐
   │ L1  L1   │   │ L1              │
   │ L2 L2 L2 │   │ L2        L2   │
   │ ...       │   │ L3  ...         │
   └───────────┘   └─────────────────┘

   Leaf-wise → Plus précis mais peut overfitter si mal configuré
   → Contrôler avec min_data_in_leaf et max_depth

POURQUOI EN ICS/IIoT
──────────────────────
LightGBM est très rapide sur les grands datasets grâce à GOSS+EFB.
Recommandé quand le dataset > 500K exemples ou quand la vitesse
d'entraînement est prioritaire.

ATTENTION : LightGBM peut sous-performer si mal configuré sur des
données très déséquilibrées. Toujours utiliser class_weight='balanced'
ou is_unbalance=True.

FORCES
  ✓ Très rapide (5-10× plus rapide que XGBoost sur grands datasets)
  ✓ Faible consommation mémoire
  ✓ Gère les features catégorielles nativement
  ✓ Excellente précision comparable à XGBoost

FAIBLESSES
  ✗ Sensible aux hyperparamètres (num_leaves critique)
  ✗ Risque d'overfitting avec leaf-wise si num_leaves trop grand
  ✗ Instable sur petits datasets (< 10K)
  ✗ Moins de documentation que XGBoost pour les cas edge

COMMANDE D'EXÉCUTION
  python train_lightgbm.py
  Durée : ~30-120 secondes
  RAM : ~500 MB
  GPU : optionnel (device='gpu')

SORTIES
  results/models/lightgbm_standalone.pkl
  results/metrics/lightgbm_standalone_metrics.json
═══════════════════════════════════════════════════════════════════
"""

import sys, os, json, time, joblib, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import (
    f1_score, accuracy_score, matthews_corrcoef,
    classification_report, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ton_iot_config import *

os.makedirs(os.path.join(RESULTS_DIR, "saved_models"), exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMÈTRES — LightGBM
# ──────────────────────────────────────────────────────────────────────────────
#
#  n_estimators (défaut=100)
#    → Nombre d'arbres (itérations de boosting)
#    → Similaire à XGBoost mais LightGBM converge souvent plus vite
#    → Utiliser avec early_stopping
#
#  num_leaves (défaut=31, recommandé=20-63)
#    → PARAMÈTRE LE PLUS IMPORTANT de LightGBM
#    → Contrôle la complexité de chaque arbre (croissance leaf-wise)
#    → num_leaves > 2^max_depth → overfitting garantie
#    → RÈGLE : num_leaves < 2^max_depth
#    → Pour données ICS : 15-31 (conservateur)
#
#  min_data_in_leaf (défaut=20, recommandé=10-200)
#    → Nombre minimum d'instances dans une feuille
#    → Régularisation principale pour leaf-wise growth
#    → Valeur élevée → underfitting, valeur faible → overfitting
#    → CRITICAL pour éviter l'overfitting sur classes rares
#    → Si mitm a 200 instances, min_data_in_leaf=5 au minimum
#
#  learning_rate (défaut=0.1)
#    → Même rôle que dans XGBoost
#    → LightGBM peut utiliser des learning_rate plus grands car
#      la convergence est souvent plus stable
#
#  class_weight='balanced' ou is_unbalance=True
#    → 'balanced' : sklearn calcule automatiquement les poids
#    → is_unbalance=True : paramètre natif LightGBM (similaire)
#    → TOUJOURS activer pour des données déséquilibrées
#
#  feature_name (IMPORTANT)
#    → LightGBM fonctionne mieux avec des noms de features (pandas DF)
#    → Passer X comme DataFrame, pas numpy array, pour class_weight
#    → C'est pourquoi LightGBM peut échouer avec des arrays numpy
#      lorsque class_weight est utilisé
#
# ──────────────────────────────────────────────────────────────────────────────

HYPERPARAMS = {
    "n_estimators":    200,
    "num_leaves":      31,
    "min_data_in_leaf":20,
    "learning_rate":   0.1,
    "class_weight":   "balanced",
    "random_state":    42,
    "verbose":        -1,
    "n_jobs":         -1,
}


def train(mode="standalone"):
    print("=" * 60)
    print(f"LIGHTGBM — Stage-2 Classification (mode: {mode})")
    print("=" * 60)

    print("\n1. Chargement des données...")
    with open(os.path.join(METRICS_DIR, "label_classes.json")) as f:
        class_names = json.load(f)["classes"]
    with open(os.path.join(METRICS_DIR, "dataset_meta.json")) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]

    # IMPORTANT : LightGBM avec class_weight nécessite des noms de features
    # On utilise pandas DataFrame pour garantir le bon fonctionnement
    if mode == "standalone":
        X_np  = np.load(os.path.join(METRICS_DIR, "X_train_std.npy"))
        y_tr  = np.load(os.path.join(METRICS_DIR, "y_train.npy"))
        Xts_np = np.load(os.path.join(METRICS_DIR, "X_test_std.npy"))
        y_ts  = np.load(os.path.join(METRICS_DIR, "y_test.npy"))
        # Utiliser des noms de features pour éviter les avertissements
        fnames = (feature_names if len(feature_names) == X_np.shape[1]
                  else [f"f{i}" for i in range(X_np.shape[1])])
        X_tr_df  = pd.DataFrame(X_np,   columns=fnames)
        X_ts_df  = pd.DataFrame(Xts_np, columns=fnames)
    else:
        flag_tr = np.load(os.path.join(METRICS_DIR, "s1_flag_train.npy"))
        flag_ts = np.load(os.path.join(METRICS_DIR, "s1_flag_test.npy"))
        sc_tr   = np.load(os.path.join(METRICS_DIR, "s1_score_train.npy"))
        sc_ts   = np.load(os.path.join(METRICS_DIR, "s1_score_test.npy"))
        X_tr_std = np.load(os.path.join(METRICS_DIR, "X_train_std.npy"))
        X_ts_std = np.load(os.path.join(METRICS_DIR, "X_test_std.npy"))
        y_all_tr = np.load(os.path.join(METRICS_DIR, "y_train.npy"))
        y_all_ts = np.load(os.path.join(METRICS_DIR, "y_test.npy"))
        X_hyb_tr = np.column_stack([X_tr_std[flag_tr],
                                     sc_tr[flag_tr].reshape(-1,1)])
        X_hyb_ts = np.column_stack([X_ts_std[flag_ts],
                                     sc_ts[flag_ts].reshape(-1,1)])
        y_tr = y_all_tr[flag_tr]
        y_ts = y_all_ts[flag_ts]
        fnames = ([f"f{i}" for i in range(X_hyb_tr.shape[1])])
        X_tr_df = pd.DataFrame(X_hyb_tr, columns=fnames)
        X_ts_df = pd.DataFrame(X_hyb_ts, columns=fnames)

    print(f"   Train : {X_tr_df.shape} | Test : {X_ts_df.shape}")

    print(f"\n2. Entraînement LightGBM (hyperparams: {HYPERPARAMS})...")
    t0 = time.time()
    model = lgb.LGBMClassifier(**HYPERPARAMS)
    model.fit(X_tr_df, y_tr)
    t_fit = time.time() - t0
    print(f"   Durée : {t_fit:.1f}s")

    print("\n3. Évaluation...")
    y_pred = model.predict(X_ts_df)
    y_prob = model.predict_proba(X_ts_df)

    f1m  = f1_score(y_ts, y_pred, average='macro',    zero_division=0)
    f1w  = f1_score(y_ts, y_pred, average='weighted', zero_division=0)
    acc  = accuracy_score(y_ts, y_pred)
    mcc  = matthews_corrcoef(y_ts, y_pred)
    rpt  = classification_report(y_ts, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)

    print(classification_report(y_ts, y_pred, target_names=class_names,
                                  zero_division=0))
    print(f"   MCC : {mcc:.4f}")

    stealthy_r = {c: round(rpt[c]['recall'], 4) for c in STEALTHY_CLASSES
                   if c in rpt}
    print(f"   Recall furtifs : {stealthy_r}")

    # Matrice de confusion
    cm_norm = confusion_matrix(y_ts, y_pred,
                                labels=range(len(class_names))).astype(float)
    cm_norm /= (cm_norm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.3, vmin=0, vmax=1, ax=ax)
    ax.set_title(f"LightGBM ({mode}) — Matrice de confusion")
    ax.set_ylabel("Vraie classe"); ax.set_xlabel("Prédite")
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"lgb_{mode}_cm.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    model_path = os.path.join(RESULTS_DIR, "saved_models",
                               f"lightgbm_{mode}.pkl")
    joblib.dump({"model": model, "class_names": class_names,
                  "hyperparams": HYPERPARAMS, "feature_names": fnames,
                  "mode": mode}, model_path)

    metrics = {
        "model": f"LightGBM ({mode})",
        "hyperparams": HYPERPARAMS,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1m, 4),
        "f1_weighted": round(f1w, 4),
        "mcc": round(mcc, 4),
        "stealthy_recalls": stealthy_r,
        "fit_time_s": round(t_fit, 1),
    }
    with open(os.path.join(METRICS_DIR, f"lgb_{mode}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nLightGBM ({mode}) terminé. F1-macro={f1m:.4f}")
    return metrics


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC : Pourquoi LightGBM peut sous-performer ?
# ══════════════════════════════════════════════════════════════════
#
# Problème fréquent : LightGBM retourne F1-macro très bas (<0.5)
# malgré class_weight='balanced'
#
# CAUSE : LightGBM avec des arrays numpy ignores les class_weight
#   pour la classification multi-classe dans certaines versions.
#   Le modèle prédit tout dans la classe majoritaire.
#
# SOLUTION 1 : Utiliser pandas DataFrame (comme dans ce script)
# SOLUTION 2 : Utiliser is_unbalance=True au lieu de class_weight
# SOLUTION 3 : Passer les poids manuellement via sample_weight :
#
#   >>> from sklearn.utils import compute_sample_weight
#   >>> sw = compute_sample_weight('balanced', y_train)
#   >>> model.fit(X_train, y_train, sample_weight=sw)
#
# DIAGNOSTIC : Si F1-macro ≈ 1/n_classes, le modèle prédit
#   uniformément toutes les classes → problème de déséquilibre
# ══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standalone", "hybrid"],
                        default="standalone")
    args = parser.parse_args()
    train(mode=args.mode)
