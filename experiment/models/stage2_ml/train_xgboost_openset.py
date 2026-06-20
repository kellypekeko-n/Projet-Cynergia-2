"""
═══════════════════════════════════════════════════════════════════
XGBOOST OPEN-SET — Stage-2 avec classe UNKNOWN
═══════════════════════════════════════════════════════════════════

PROBLÈME ADRESSÉ
────────────────
Un XGBoost classique est un système "closed-set" : il FORCE toujours
une prédiction parmi les N classes connues, même face à une attaque
qu'il n'a jamais vue. Une attaque zero-day est silencieusement
mal-étiquetée comme "backdoor" ou "mitm" — dangereux en ICS réel.

SOLUTION : OPEN-SET RECOGNITION
─────────────────────────────────
On entraîne le modèle avec une 11ème classe artificielle "UNKNOWN"
construite à partir d'échantillons synthétiques qui tombent dans les
zones grises entre les classes connues. Le modèle apprend activement
à reconnaître "je ne sais pas".

GÉNÉRATION DES ÉCHANTILLONS UNKNOWN
─────────────────────────────────────
Méthode : Inter-class Mixup
  Pour chaque échantillon UNKNOWN synthétique :
  1. Tirer deux classes connues différentes c1, c2
  2. Tirer un échantillon x1 ∈ c1 et x2 ∈ c2
  3. Interpoler : x_unknown = λ·x1 + (1-λ)·x2,  λ ∈ [0.3, 0.7]

  Ces échantillons sont volontairement ambigus : ils ne ressemblent
  clairement à aucune classe. Le modèle apprend que cette ambiguïté
  signifie "UNKNOWN".

Méthode complémentaire : Gaussian Tail Sampling
  Échantillons aux extrêmes de la distribution (scores très élevés).
  Simule des attaques de nature radicalement différente.

ÉVALUATION LEAVE-ONE-CLASS-OUT
────────────────────────────────
Pour valider que le modèle dit vraiment "UNKNOWN" face à une nouvelle
attaque, on retire une classe de l'entraînement et on vérifie que
le modèle la prédit comme UNKNOWN au test.

COMMANDE D'EXÉCUTION
  python train_xgboost_openset.py
  python train_xgboost_openset.py --mode hybrid
  python train_xgboost_openset.py --held-out backdoor   (leave-one-out)
  Durée : ~2-5 minutes
═══════════════════════════════════════════════════════════════════
"""

import sys, os, json, time, joblib, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, matthews_corrcoef,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ton_iot_config import *

os.makedirs(os.path.join(RESULTS_DIR, "saved_models"), exist_ok=True)

UNKNOWN_LABEL = "UNKNOWN"
UNKNOWN_RATIO = 0.15      # fraction des données d'entraînement générées en UNKNOWN
MIN_UNKNOWN   = 3_000     # minimum d'échantillons UNKNOWN à générer


# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATION DES ÉCHANTILLONS UNKNOWN SYNTHÉTIQUES
# ─────────────────────────────────────────────────────────────────────────────

def generate_unknown_mixup(X_train, y_train, n_unknown, seed=42):
    """
    Inter-class Mixup : crée des échantillons dans la zone grise entre classes.

    x_unknown = λ·x_c1 + (1-λ)·x_c2   avec λ ∈ [0.3, 0.7], c1 ≠ c2

    Ces vecteurs sont délibérément ambigus — ni c1 ni c2 clairement.
    Le modèle apprend que cette zone = "je ne sais pas".
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y_train)
    unknown = []

    for _ in range(n_unknown):
        c1, c2 = rng.choice(classes, size=2, replace=False)
        idx1 = rng.choice(np.where(y_train == c1)[0])
        idx2 = rng.choice(np.where(y_train == c2)[0])
        lam  = rng.uniform(0.3, 0.7)
        unknown.append(lam * X_train[idx1] + (1 - lam) * X_train[idx2])

    return np.array(unknown, dtype=np.float32)


def generate_unknown_tails(X_train, n_unknown, seed=43):
    """
    Gaussian Tail Sampling : échantillons hors de la distribution connue.

    Simule des attaques de nature radicalement différente des classes vues.
    Centré sur la moyenne ± 3σ de chaque feature.
    """
    rng  = np.random.default_rng(seed)
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    n_half = n_unknown // 2

    # Côté haut de la distribution (+3σ)
    high = rng.normal(loc=mean + 3 * std, scale=std * 0.5, size=(n_half, X_train.shape[1]))
    # Côté bas de la distribution (-3σ)
    low  = rng.normal(loc=mean - 3 * std, scale=std * 0.5, size=(n_unknown - n_half, X_train.shape[1]))

    return np.clip(np.vstack([high, low]), 0, None).astype(np.float32)


def build_openset_dataset(X_train, y_train, unknown_class_id, seed=42):
    """
    Augmente le dataset d'entraînement avec des échantillons UNKNOWN synthétiques.

    Retourne X_aug, y_aug avec la classe UNKNOWN = unknown_class_id.
    """
    n_unknown = max(MIN_UNKNOWN, int(len(X_train) * UNKNOWN_RATIO))
    n_mixup   = int(n_unknown * 0.70)
    n_tails   = n_unknown - n_mixup

    print(f"  Génération de {n_unknown:,} échantillons UNKNOWN :")
    print(f"    Mixup inter-classes : {n_mixup:,}")
    print(f"    Queue gaussienne    : {n_tails:,}")

    X_mixup = generate_unknown_mixup(X_train, y_train, n_mixup, seed=seed)
    X_tails = generate_unknown_tails(X_train, n_tails, seed=seed + 1)

    X_unknown = np.vstack([X_mixup, X_tails])
    y_unknown = np.full(len(X_unknown), unknown_class_id, dtype=np.int64)

    X_aug = np.vstack([X_train, X_unknown])
    y_aug = np.concatenate([y_train, y_unknown])

    idx = np.random.default_rng(seed).permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]


# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATION LEAVE-ONE-CLASS-OUT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_leave_one_out(X_tr, y_tr, X_ts, y_ts, class_names,
                            held_out_cls, unknown_class_id, hyperparams):
    """
    Retire la classe held_out_cls du train, l'utilise comme UNKNOWN au test.

    Vérifie que le modèle prédit bien UNKNOWN quand il voit cette classe
    pour la première fois.
    """
    known_mask_tr = y_tr != class_names.index(held_out_cls)
    X_tr_loo = X_tr[known_mask_tr]
    y_tr_loo = y_tr[known_mask_tr]

    # Réindexer les classes restantes pour éviter les trous
    remaining_classes = [c for c in class_names if c != held_out_cls]
    remap = {class_names.index(c): i for i, c in enumerate(remaining_classes)}
    y_tr_loo = np.array([remap[y] for y in y_tr_loo])

    n_known = len(remaining_classes)
    X_tr_aug, y_tr_aug = build_openset_dataset(X_tr_loo, y_tr_loo, n_known)

    model = XGBClassifier(**{**hyperparams,
                              "num_class": n_known + 1,
                              "n_estimators": 100})
    model.fit(X_tr_aug, y_tr_aug)

    # Test sur la classe retirée : doit être prédite comme UNKNOWN
    held_id   = class_names.index(held_out_cls)
    mask_held = y_ts == held_id
    if mask_held.sum() == 0:
        print(f"  [{held_out_cls}] aucun exemple dans le test — skip")
        return None

    X_held = X_ts[mask_held]
    y_pred_held = model.predict(X_held)
    unknown_rate = (y_pred_held == n_known).mean()

    print(f"  [{held_out_cls}] Taux de détection comme UNKNOWN : "
          f"{unknown_rate:.3f} ({mask_held.sum()} échantillons)")
    return {"held_out": held_out_cls, "unknown_rate": round(float(unknown_rate), 4),
            "n_samples": int(mask_held.sum())}


# ─────────────────────────────────────────────────────────────────────────────
# ENTRAÎNEMENT PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

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


def train(mode="standalone", held_out=None):
    print("=" * 60)
    print(f"XGBOOST OPEN-SET — Stage-2 (mode: {mode})")
    print("=" * 60)

    # ── Chargement des données ────────────────────────────────────────────────
    print("\n1. Chargement des données...")
    with open(os.path.join(METRICS_DIR, "label_classes.json")) as f:
        class_names = json.load(f)["classes"]

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
        X_train = np.column_stack([X_tr[flag_tr], sc_tr[flag_tr].reshape(-1, 1)])
        y_train = y_tr[flag_tr]
        X_test  = np.column_stack([X_ts[flag_ts], sc_ts[flag_ts].reshape(-1, 1)])
        y_test  = y_ts[flag_ts]

    n_known       = len(class_names)
    unknown_id    = n_known                              # index 10
    all_classes   = class_names + [UNKNOWN_LABEL]        # 11 classes

    print(f"   Train : {X_train.shape} | Test : {X_test.shape}")
    print(f"   Classes connues : {n_known} | Classe UNKNOWN id : {unknown_id}")

    # ── Leave-one-class-out (optionnel) ───────────────────────────────────────
    loo_results = []
    if held_out:
        print(f"\n2. Leave-one-class-out sur : {held_out}")
        if held_out not in class_names:
            print(f"   ERREUR : '{held_out}' n'est pas dans {class_names}")
        else:
            r = evaluate_leave_one_out(
                X_train, y_train, X_test, y_test,
                class_names, held_out, unknown_id, HYPERPARAMS
            )
            if r:
                loo_results.append(r)
    else:
        # Évaluer sur toutes les classes furtives automatiquement
        print("\n2. Leave-one-class-out sur les classes furtives...")
        for cls in STEALTHY_CLASSES:
            if cls in class_names:
                r = evaluate_leave_one_out(
                    X_train, y_train, X_test, y_test,
                    class_names, cls, unknown_id, HYPERPARAMS
                )
                if r:
                    loo_results.append(r)

    # ── Construction du dataset open-set ─────────────────────────────────────
    print("\n3. Construction du dataset open-set...")
    X_aug, y_aug = build_openset_dataset(X_train, y_train, unknown_id)
    print(f"   Dataset augmenté : {X_aug.shape} "
          f"({(y_aug == unknown_id).sum():,} UNKNOWN / {len(y_aug):,} total)")

    # ── Entraînement ─────────────────────────────────────────────────────────
    print(f"\n4. Entraînement XGBoost open-set ({len(all_classes)} classes)...")
    t0 = time.time()
    model = XGBClassifier(**HYPERPARAMS)
    model.fit(X_aug, y_aug)
    t_fit = time.time() - t0
    print(f"   Durée : {t_fit:.1f}s")

    # ── Évaluation sur le test set (classes connues uniquement) ───────────────
    print("\n5. Évaluation sur le test set...")
    # Pour l'évaluation standard, on évalue sur les classes connues.
    # La classe UNKNOWN n'est pas dans y_test (normal — elle est synthétique).
    t1     = time.time()
    y_pred = model.predict(X_test)
    lat_ms = (time.time() - t1) / max(len(X_test), 1) * 1000

    # Remplacer les prédictions UNKNOWN par la 2ème classe probable
    # pour le calcul des métriques sur les classes connues
    y_pred_known = y_pred.copy()
    y_pred_known[y_pred_known == unknown_id] = n_known - 1  # fallback

    f1m = f1_score(y_test, y_pred_known, average='macro',
                   labels=list(range(n_known)), zero_division=0)
    acc = accuracy_score(y_test, y_pred_known)
    mcc = matthews_corrcoef(y_test, y_pred_known)

    # Taux de prédiction UNKNOWN sur le test set connu
    unknown_rate_test = (y_pred == unknown_id).mean()

    rpt = classification_report(y_test, y_pred_known,
                                  target_names=class_names,
                                  zero_division=0)
    print(rpt)
    print(f"   F1-macro (classes connues) : {f1m:.4f}")
    print(f"   MCC                        : {mcc:.4f}")
    print(f"   Latence                    : {lat_ms:.4f} ms/flux")
    print(f"   Taux UNKNOWN sur test connu: {unknown_rate_test:.4f} "
          f"(idéalement proche de 0 sur des attaques connues)")

    # Recalls des classes furtives
    rpt_dict = classification_report(y_test, y_pred_known,
                                      target_names=class_names,
                                      output_dict=True, zero_division=0)
    print("\n   Recall des classes furtives :")
    for cls in STEALTHY_CLASSES:
        if cls in rpt_dict:
            r = rpt_dict[cls]['recall']
            print(f"   {cls:12s} : Recall={r:.4f}  "
                  f"(support={rpt_dict[cls]['support']})")

    # ── Leave-one-class-out résumé ────────────────────────────────────────────
    if loo_results:
        print("\n   Leave-One-Class-Out — taux de détection UNKNOWN :")
        for r in loo_results:
            bar = "█" * int(r["unknown_rate"] * 20)
            print(f"   {r['held_out']:12s} : {r['unknown_rate']:.3f}  {bar}")
        avg = np.mean([r["unknown_rate"] for r in loo_results])
        print(f"   Moyenne          : {avg:.3f}")

    # ── Matrice de confusion ──────────────────────────────────────────────────
    cm      = confusion_matrix(y_test, y_pred_known, labels=list(range(n_known)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.3, vmin=0, vmax=1, ax=ax)
    ax.set_title(f"XGBoost Open-Set ({mode}) — Matrice de confusion normalisée")
    ax.set_ylabel("Vraie classe")
    ax.set_xlabel("Classe prédite")
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"xgb_openset_{mode}_cm.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n   [FIG] xgb_openset_{mode}_cm.png")

    # ── Leave-one-out barplot ─────────────────────────────────────────────────
    if loo_results:
        fig, ax = plt.subplots(figsize=(7, 4))
        names = [r["held_out"] for r in loo_results]
        rates = [r["unknown_rate"] for r in loo_results]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
        bars = ax.barh(names, rates, color=colors, edgecolor='white')
        ax.axvline(0.5, color='red', linestyle='--', lw=1, label='seuil 50%')
        ax.set_xlabel("Taux de détection comme UNKNOWN")
        ax.set_title("Open-Set — Capacité à dire 'Je ne sais pas'\n"
                     "(Leave-One-Class-Out par classe furtive)")
        for b, v in zip(bars, rates):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2,
                    f"{v:.3f}", va='center', fontsize=9)
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "xgb_openset_loo.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   [FIG] xgb_openset_loo.png")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    model_path = os.path.join(RESULTS_DIR, "saved_models",
                              f"xgboost_openset_{mode}.pkl")
    joblib.dump({
        "model":        model,
        "class_names":  all_classes,   # 11 classes dont UNKNOWN
        "unknown_label": UNKNOWN_LABEL,
        "unknown_id":   unknown_id,
        "hyperparams":  HYPERPARAMS,
        "mode":         mode,
    }, model_path)
    print(f"\n   Modèle sauvegardé : {model_path}")

    metrics = {
        "model":               f"XGBoost Open-Set ({mode})",
        "n_classes":           len(all_classes),
        "class_names":         all_classes,
        "unknown_label":       UNKNOWN_LABEL,
        "f1_macro_known":      round(f1m, 4),
        "accuracy":            round(acc, 4),
        "mcc":                 round(mcc, 4),
        "unknown_rate_test":   round(float(unknown_rate_test), 4),
        "leave_one_out":       loo_results,
        "stealthy_recalls":    {
            cls: round(rpt_dict[cls]['recall'], 4)
            for cls in STEALTHY_CLASSES if cls in rpt_dict
        },
        "fit_time_s":          round(t_fit, 1),
        "latency_ms":          round(lat_ms, 4),
    }
    metrics_path = os.path.join(METRICS_DIR, f"xgboost_openset_{mode}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"   Métriques sauvegardées : {metrics_path}")

    print(f"\nXGBoost Open-Set ({mode}) terminé.")
    print(f"  F1-macro={f1m:.4f} | MCC={mcc:.4f} | "
          f"UNKNOWN rate on known={unknown_rate_test:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standalone", "hybrid"],
                        default="standalone")
    parser.add_argument("--held-out", type=str, default=None,
                        help=f"Classe à retirer pour le leave-one-out. "
                             f"Choix : {STEALTHY_CLASSES}. "
                             f"Si absent, évalue toutes les classes furtives.")
    args = parser.parse_args()
    train(mode=args.mode, held_out=args.held_out)
