"""
Ensemble Parallèle Spécialisé — Stage 1
Trois détecteurs (ou quatre avec --with-ae) entraînés simultanément,
chacun avec un rôle précis et un seuil calibré pour ce rôle.
Leurs décisions binaires sont combinées par vote configurable.

Rôles spécialisés :
  OCSVM  — frontière globale (boundary precision) : FPR ≤ 3%
             "Ce flux sort-il de la zone normale globale ?"
  IF     — outlier structurel (recall maximum)    : recall ≥ 95%
             "Ce flux a-t-il des combinaisons de features inhabituelles ?"
  LOF    — isolement local (stealthy specialist)  : stealthy recall ≥ 90%
             "Ce flux est-il isolé de ses voisins normaux ?"
  AE     — reconstruction neuronale (--with-ae)   : erreur MSE > seuil
             "Ce flux est-il difficile à reconstruire pour le réseau normal ?"

Vote final :
  OR        (≥1/N) : flag si n'importe quel détecteur alerte → recall max
  MAJORITY  (≥N/2) : flag si la majorité des détecteurs alertent → équilibre
  AND       (N/N)  : flag si tous alertent → précision max

Innovation vs score composite :
  - Composite : combine les SCORES (décision douce)
  - Ensemble  : combine les DÉCISIONS binaires après calibration spécialisée
  - Une attaque doit tromper >= vote_threshold détecteurs pour passer

Usage:
  python train_ensemble_stage1.py
  python train_ensemble_stage1.py --vote majority
  python train_ensemble_stage1.py --with-ae --vote majority
  python train_ensemble_stage1.py --vote or --no-retrain
"""
import os, sys, json, time, warnings, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              confusion_matrix)

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ton_iot_config import (METRICS_DIR, FIGURES_DIR, STEALTHY_CLASSES,
                              STAGE1_TARGET_RECALL, STAGE1_MAX_FPR, SEEDS, PALETTE)

SCORES_CACHE = os.path.join(METRICS_DIR, "composite_s1_scores.npz")
METRICS_OUT  = os.path.join(METRICS_DIR, "ensemble_stage1_metrics.json")

ROLES = {
    "OCSVM": {
        "description": "Frontiere globale — precision",
        "target":      "fpr",
        "fpr_budget":  0.03,            # seuil serré → peu de faux positifs
        "recall_min":  STAGE1_TARGET_RECALL,
    },
    "IF": {
        "description": "Outlier structurel — recall maximum",
        "target":      "recall",
        "recall_min":  0.95,            # attrape le max d'attaques
        "fpr_budget":  STAGE1_MAX_FPR,
    },
    "LOF": {
        "description": "Isolement local — specialiste classes furtives",
        "target":      "stealthy_recall",
        "recall_min":  STAGE1_TARGET_RECALL,
        "fpr_budget":  STAGE1_MAX_FPR,
    },
}

VOTE_RULES = {"or": 1, "majority": 2, "and": 3}


# ── calibration spécialisée ────────────────────────────────────────────────────

def calibrate_for_precision(sc_normal, sc_attack, fpr_budget, recall_min):
    """Seuil le plus haut qui maintient recall >= recall_min et minimise FPR."""
    all_sc = np.concatenate([sc_normal, sc_attack])
    thresholds = np.unique(np.percentile(all_sc, np.linspace(0, 100, 3000)))
    best_theta, best = None, {"recall": 0.0, "fpr": 1.0}
    for th in sorted(thresholds, reverse=True):
        tp = (sc_attack >= th).sum();  fn = (sc_attack < th).sum()
        fp = (sc_normal >= th).sum();  tn = (sc_normal < th).sum()
        recall = tp / (tp + fn + 1e-12)
        fpr    = fp / (fp + tn + 1e-12)
        if recall >= recall_min and fpr <= fpr_budget:
            if best_theta is None or fpr < best["fpr"]:
                best_theta = th
                best = {"recall": float(recall), "fpr": float(fpr),
                        "precision": float(tp/(tp+fp+1e-12))}
    if best_theta is None:
        # relax FPR : au moins recall_min
        for th in sorted(thresholds, reverse=True):
            tp = (sc_attack >= th).sum();  fn = (sc_attack < th).sum()
            fp = (sc_normal >= th).sum();  tn = (sc_normal < th).sum()
            recall = tp/(tp+fn+1e-12); fpr = fp/(fp+tn+1e-12)
            if recall >= recall_min:
                return th, {"recall": float(recall), "fpr": float(fpr), "precision": 0.0}
        best_theta = np.median(all_sc)
    return best_theta, best


def calibrate_for_recall(sc_normal, sc_attack, recall_target, fpr_budget):
    """Seuil qui atteint recall_target en minimisant FPR."""
    return calibrate_for_precision(sc_normal, sc_attack, fpr_budget, recall_target)


def calibrate_for_stealthy(sc_normal, sc_attack_stealthy, sc_attack_all,
                            recall_target, fpr_budget):
    """Seuil qui maximise le recall sur les classes furtives."""
    all_sc = np.concatenate([sc_normal, sc_attack_all])
    thresholds = np.unique(np.percentile(all_sc, np.linspace(0, 100, 3000)))
    best_theta, best = None, {"stealthy_recall": 0.0, "fpr": 1.0}
    for th in sorted(thresholds, reverse=True):
        s_recall = (sc_attack_stealthy >= th).mean()       # recall sur furtifs
        fpr      = (sc_normal >= th).mean()                 # FPR sur normaux
        recall   = (sc_attack_all >= th).mean()             # recall global
        if s_recall >= recall_target and fpr <= fpr_budget:
            if best_theta is None or fpr < best["fpr"]:
                best_theta = th
                best = {"stealthy_recall": float(s_recall),
                        "recall":          float(recall),
                        "fpr":             float(fpr)}
    if best_theta is None:
        for th in sorted(thresholds, reverse=True):
            if (sc_attack_stealthy >= th).mean() >= recall_target:
                return th, {"stealthy_recall": float((sc_attack_stealthy>=th).mean()),
                            "fpr": float((sc_normal>=th).mean()), "recall": 0.0}
        best_theta = np.median(all_sc)
    return best_theta, best


# ── métriques ensemble ─────────────────────────────────────────────────────────

def stealthy_enrichment(y_full, y_flagged, stealthy_ids):
    r_before = np.isin(y_full,    stealthy_ids).mean()
    r_after  = np.isin(y_flagged, stealthy_ids).mean()
    return float(r_before), float(r_after), float(r_after / (r_before + 1e-12))


def eval_binary(y_bin, flag, label):
    tp = int(((y_bin==1)&flag).sum()); fn = int(((y_bin==1)&~flag).sum())
    fp = int(((y_bin==0)&flag).sum()); tn = int(((y_bin==0)&~flag).sum())
    recall = tp/(tp+fn+1e-12); fpr = fp/(fp+tn+1e-12)
    prec   = tp/(tp+fp+1e-12); f1  = 2*prec*recall/(prec+recall+1e-12)
    return {"label": label, "recall": round(recall,4), "fpr": round(fpr,4),
            "precision": round(prec,4), "f1": round(f1,4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_flagged": int(flag.sum()), "flagged_pct": round(flag.mean()*100,2)}


# ── training ──────────────────────────────────────────────────────────────────

def train_models(X_normal, seed=SEEDS[0]):
    print("  Training OCSVM  (role: frontiere globale)...", end=" ", flush=True)
    t = time.time()
    ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale').fit(X_normal)
    print(f"{time.time()-t:.1f}s")

    print("  Training IF     (role: outlier structurel)...", end=" ", flush=True)
    t = time.time()
    iforest = IsolationForest(n_estimators=200, contamination=0.01,
                              max_features=0.8, random_state=seed, n_jobs=-1).fit(X_normal)
    print(f"{time.time()-t:.1f}s")

    print("  Training LOF    (role: isolement local furtif)...", end=" ", flush=True)
    t = time.time()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True).fit(X_normal)
    print(f"{time.time()-t:.1f}s")

    return {"OCSVM": ocsvm, "IF": iforest, "LOF": lof}


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    vote_min = VOTE_RULES[args.vote]

    print(f"\n=== Ensemble Parallele Specialise Stage 1 ===")
    print(f"  Vote : {args.vote.upper()} (flagge si >= {vote_min}/3 detecteurs alertent)\n")

    # 1. Charger les données
    print("--- Chargement donnees ---")
    X_train_mm = np.load(os.path.join(METRICS_DIR, "X_train_mm.npy"))
    X_val_mm   = np.load(os.path.join(METRICS_DIR, "X_val_mm.npy"))
    X_test_mm  = np.load(os.path.join(METRICS_DIR, "X_test_mm.npy"))
    y_train    = np.load(os.path.join(METRICS_DIR, "y_train.npy"))
    y_val      = np.load(os.path.join(METRICS_DIR, "y_val.npy"))
    y_test     = np.load(os.path.join(METRICS_DIR, "y_test.npy"))

    with open(os.path.join(METRICS_DIR, "label_classes.json")) as f:
        classes = json.load(f)["classes"]
    normal_idx   = list(classes).index("normal")
    stealthy_ids = [list(classes).index(c) for c in STEALTHY_CLASSES if c in classes]

    X_normal_tr      = X_train_mm[y_train == normal_idx]
    X_val_normal     = X_val_mm[y_val == normal_idx]
    X_val_attack     = X_val_mm[y_val != normal_idx]
    X_val_stealthy   = X_val_mm[np.isin(y_val, stealthy_ids)]
    y_val_bin        = (y_val != normal_idx).astype(int)
    y_test_bin       = (y_test != normal_idx).astype(int)
    print(f"  Normal train: {len(X_normal_tr):,} | Val: {len(X_val_mm):,} | Test: {len(X_test_mm):,}")

    # 2. Entrainer ou charger scores
    if not args.no_retrain or not os.path.exists(SCORES_CACHE):
        print("\n--- Entrainement des 3 detecteurs en parallele ---")
        models = train_models(X_normal_tr)

        print("\n--- Calcul des scores anomalie ---")
        raw = {}
        for split_name, X in [("val_n", X_val_normal), ("val_a", X_val_attack),
                               ("val_all", X_val_mm), ("test", X_test_mm),
                               ("train", X_train_mm)]:
            for m_name, m in models.items():
                raw[f"{split_name}_{m_name}"] = -m.decision_function(X)

        # Scores stealthy sur val (pour calibration LOF)
        for m_name, m in models.items():
            raw[f"val_stealthy_{m_name}"] = -m.decision_function(X_val_stealthy)

        np.savez(SCORES_CACHE, **raw)
        print(f"  Scores mis en cache: {SCORES_CACHE}")
    else:
        print(f"\n--- Chargement scores cache ---")
        data = np.load(SCORES_CACHE)
        raw  = dict(data)
        # Reconstruire val_stealthy si absent du cache (cache cree par ton_01, pas par cet script)
        stealthy_mask = np.isin(y_val, stealthy_ids)
        for m_name in ["OCSVM", "IF", "LOF"]:
            key = f"val_stealthy_{m_name}"
            if key not in raw and f"val_all_{m_name}" in raw:
                raw[key] = raw[f"val_all_{m_name}"][stealthy_mask]

    keys = ["OCSVM", "IF", "LOF"]

    # 2b. Charger scores AE si disponibles et demandés
    ae_scores_path = os.path.join(METRICS_DIR, "ae_s1_scores.npz")
    use_ae = args.with_ae and os.path.exists(ae_scores_path)
    if args.with_ae and not os.path.exists(ae_scores_path):
        print("  [WARN] Scores AE non trouves — executer train_autoencoder.py d'abord.")
        print("  python experiment/models/stage1/train_autoencoder.py")
    if use_ae:
        ae_data = np.load(ae_scores_path)
        raw["val_n_AE"]       = ae_data["val_n"]
        raw["val_a_AE"]       = ae_data["val_a"]
        raw["val_all_AE"]     = ae_data["val_all"]
        raw["val_stealthy_AE"]= ae_data["val_a"]   # approximation; AE pas specialized
        raw["test_AE"]        = ae_data["test"]
        raw["train_AE"]       = ae_data["train"]
        keys.append("AE")
        print(f"  Autoencoder inclus comme 4e detecteur.")

    n_detectors = len(keys)
    # Vote rules dynamiques selon nombre de détecteurs
    VOTE_RULES_DYN = {
        "or":       1,
        "majority": max(2, n_detectors // 2 + 1),
        "and":      n_detectors,
    }
    vote_min = VOTE_RULES_DYN[args.vote]
    print(f"\n  {n_detectors} detecteurs | Vote {args.vote.upper()} = >= {vote_min}/{n_detectors}")

    # 3. Calibration spécialisée par détecteur (sur validation)
    print("\n--- Calibration specialisee par role ---")
    thetas = {}
    calib_stats = {}

    # OCSVM : rôle précision (FPR ≤ 3%)
    th, st = calibrate_for_precision(
        raw["val_n_OCSVM"], raw["val_a_OCSVM"],
        fpr_budget=ROLES["OCSVM"]["fpr_budget"],
        recall_min=ROLES["OCSVM"]["recall_min"])
    thetas["OCSVM"] = th; calib_stats["OCSVM"] = st
    print(f"  OCSVM  [{ROLES['OCSVM']['description']}]")
    print(f"    theta={th:.4f} | recall={st['recall']:.3f} | fpr={st['fpr']:.4f}")

    # IF : rôle recall max (≥95%)
    th, st = calibrate_for_recall(
        raw["val_n_IF"], raw["val_a_IF"],
        recall_target=ROLES["IF"]["recall_min"],
        fpr_budget=ROLES["IF"]["fpr_budget"])
    thetas["IF"] = th; calib_stats["IF"] = st
    print(f"  IF     [{ROLES['IF']['description']}]")
    print(f"    theta={th:.4f} | recall={st['recall']:.3f} | fpr={st['fpr']:.4f}")

    # LOF : rôle classes furtives
    th, st = calibrate_for_stealthy(
        raw["val_n_LOF"], raw["val_stealthy_LOF"], raw["val_a_LOF"],
        recall_target=ROLES["LOF"]["recall_min"],
        fpr_budget=ROLES["LOF"]["fpr_budget"])
    thetas["LOF"] = th; calib_stats["LOF"] = st
    print(f"  LOF    [{ROLES['LOF']['description']}]")
    print(f"    theta={th:.4f} | stealthy_recall={st.get('stealthy_recall', '?'):.3f}"
          f" | fpr={st['fpr']:.4f}")

    # AE : rôle reconstruction non-lineaire (memes contraintes que OCSVM)
    if use_ae:
        th, st = calibrate_for_precision(
            raw["val_n_AE"], raw["val_a_AE"],
            fpr_budget=ROLES["OCSVM"]["fpr_budget"],
            recall_min=ROLES["OCSVM"]["recall_min"])
        thetas["AE"] = th; calib_stats["AE"] = st
        print(f"  AE     [Reconstruction neuronale — erreur MSE]")
        print(f"    theta={th:.4f} | recall={st['recall']:.3f} | fpr={st['fpr']:.4f}")

    # 4. Décisions binaires individuelles sur val et test
    print("\n--- Decisions binaires individuelles ---")
    flags_val  = {k: raw[f"val_all_{k}"]  >= thetas[k] for k in keys}
    flags_test = {k: raw[f"test_{k}"]     >= thetas[k] for k in keys}
    flags_train= {k: raw[f"train_{k}"]    >= thetas[k] for k in keys}

    for k in keys:
        r = eval_binary(y_val_bin, flags_val[k], k)
        print(f"  {k:<8} recall={r['recall']:.3f} | fpr={r['fpr']:.4f} "
              f"| flagge={r['flagged_pct']:.1f}%")

    # 5. Vote ensemble — plusieurs règles
    print("\n--- Vote ensemble ---")
    results_by_vote = {}
    for vote_name, v_min in VOTE_RULES_DYN.items():
        votes_val  = sum(flags_val[k].astype(int)  for k in keys)
        votes_test = sum(flags_test[k].astype(int) for k in keys)
        flag_ens_val  = votes_val  >= v_min
        flag_ens_test = votes_test >= v_min

        r_val  = eval_binary(y_val_bin,  flag_ens_val,  f"Ensemble-{vote_name.upper()}")
        r_test = eval_binary(y_test_bin, flag_ens_test, f"Ensemble-{vote_name.upper()}-test")

        _, _, enr_val  = stealthy_enrichment(y_val,  y_val[flag_ens_val],   stealthy_ids)
        _, _, enr_test = stealthy_enrichment(y_test, y_test[flag_ens_test], stealthy_ids)

        results_by_vote[vote_name] = {
            "val":  {**r_val,  "enrichment": round(enr_val,  4)},
            "test": {**r_test, "enrichment": round(enr_test, 4)},
        }
        print(f"  {vote_name.upper():<9} (>={v_min}/3) | "
              f"recall={r_test['recall']:.3f} | fpr={r_test['fpr']:.4f} | "
              f"enrichissement x{enr_test:.3f} | flagge={r_test['flagged_pct']:.1f}%")

    # 6. Sélection de la règle de vote demandée + sauvegarde
    chosen = results_by_vote[args.vote]
    votes_train = sum(flags_train[k].astype(int) for k in keys)
    votes_test  = sum(flags_test[k].astype(int)  for k in keys)
    flag_final_train = votes_train >= vote_min
    flag_final_test  = votes_test  >= vote_min

    # Score pseudo-continu : nombre de votes (0-3) — utilisé par Stage 2
    score_train = votes_train.astype(np.float32)
    score_test  = votes_test.astype(np.float32)

    np.save(os.path.join(METRICS_DIR, "s1_ensemble_flag_train.npy"), flag_final_train)
    np.save(os.path.join(METRICS_DIR, "s1_ensemble_flag_test.npy"),  flag_final_test)
    np.save(os.path.join(METRICS_DIR, "s1_ensemble_score_train.npy"), score_train)
    np.save(os.path.join(METRICS_DIR, "s1_ensemble_score_test.npy"),  score_test)
    print(f"\n  Flags sauvegardés (vote={args.vote.upper()}, seuil >={vote_min}/3)")

    # 7. Comparaison individuel vs ensemble
    print("\n--- Comparaison individuel vs ensemble (TEST SET) ---")
    print(f"  {'Detecteur':<22} {'Recall':>7} {'FPR':>7} {'Enrichissement':>15} {'Flagge%':>8}")
    print("  " + "-"*65)
    for k in keys:
        r = eval_binary(y_test_bin, flags_test[k], k)
        _, _, enr = stealthy_enrichment(y_test, y_test[flags_test[k]], stealthy_ids)
        print(f"  {k:<22} {r['recall']:>7.3f} {r['fpr']:>7.4f} {enr:>15.3f}x {r['flagged_pct']:>7.1f}%")
    for vn, vr in results_by_vote.items():
        rt = vr["test"]
        print(f"  {'Ensemble-'+vn.upper():<22} {rt['recall']:>7.3f} {rt['fpr']:>7.4f}"
              f" {rt['enrichment']:>15.3f}x {rt['flagged_pct']:>7.1f}%")

    # 8. Sauvegarde métriques
    with open(METRICS_OUT, "w") as f:
        json.dump({
            "roles":            ROLES,
            "thetas":           {k: float(v) for k, v in thetas.items()},
            "calib_stats":      calib_stats,
            "results_by_vote":  results_by_vote,
            "chosen_vote":      args.vote,
            "chosen_vote_min":  vote_min,
            "chosen_result":    chosen,
        }, f, indent=2)
    print(f"\n  Metriques sauvegardees: {os.path.basename(METRICS_OUT)}")

    # 9. Figures
    print("\n--- Figures ---")

    # Figure A — Votes heatmap : combien de votes par flux (val)
    votes_val_all = sum(flags_val[k].astype(int) for k in keys)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Distribution des votes par classe
    ax = axes[0]
    vote_by_class = {}
    for cls_id, cls_name in enumerate(classes):
        mask = y_val == cls_id
        if mask.sum() == 0:
            continue
        vote_by_class[cls_name] = [
            int((votes_val_all[mask] == v).sum()) for v in range(4)
        ]
    cls_names  = list(vote_by_class.keys())
    vote_mat   = np.array(list(vote_by_class.values()), dtype=float)
    vote_mat_n = vote_mat / (vote_mat.sum(axis=1, keepdims=True) + 1e-12)
    im = ax.imshow(vote_mat_n, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(['0 votes\n(normal)','1 vote','2 votes','3 votes\n(tous)'], fontsize=8)
    ax.set_yticks(range(len(cls_names))); ax.set_yticklabels(cls_names, fontsize=8)
    ax.set_title("Distribution des votes par classe (Val)")
    ax.set_xlabel("Nombre de detecteurs qui alertent")
    for i in range(len(cls_names)):
        for j in range(4):
            ax.text(j, i, f"{vote_mat_n[i,j]:.2f}", ha='center', va='center',
                    fontsize=7, color='black' if vote_mat_n[i,j] < 0.6 else 'white')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Comparaison recall/enrichissement selon règle de vote (test)
    ax = axes[1]
    vote_labels = ["OR\n(≥1/3)", "MAJORITY\n(≥2/3)", "AND\n(3/3)"]
    recalls_v   = [results_by_vote[v]["test"]["recall"]     for v in VOTE_RULES]
    enrs_v      = [results_by_vote[v]["test"]["enrichment"] for v in VOTE_RULES]
    fprs_v      = [results_by_vote[v]["test"]["fpr"]        for v in VOTE_RULES]
    x = np.arange(3)
    w = 0.25
    ax.bar(x - w, recalls_v, w, color=PALETTE[0], label='Recall',       alpha=0.85)
    ax.bar(x,     fprs_v,    w, color=PALETTE[1], label='FPR',          alpha=0.85)
    ax.bar(x + w, [e/5 for e in enrs_v], w, color=PALETTE[2],
           label='Enrichissement /5', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(vote_labels, fontsize=9)
    ax.set_ylim([0, 1.2]); ax.set_ylabel("Score (enrichissement divise par 5)")
    ax.set_title("Impact de la regle de vote (Test Set)")
    ax.legend(fontsize=8)
    ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    for i, (r, e, fp_) in enumerate(zip(recalls_v, enrs_v, fprs_v)):
        ax.text(i, r + 0.02, f"R={r:.2f}", ha='center', fontsize=7, color=PALETTE[0])
        ax.text(i + w, e/5 + 0.02, f"x{e:.2f}", ha='center', fontsize=7, color=PALETTE[2])

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_ensemble_s1_votes.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Sauvegarde: fig_ensemble_s1_votes.png")

    # Figure B — Rôles : recall individuel par classe furtive
    fig, ax = plt.subplots(figsize=(9, 4))
    stealthy_names = [c for c in STEALTHY_CLASSES if c in classes]
    n_s = len(stealthy_names)
    x   = np.arange(n_s)
    w   = 0.2
    for i, k in enumerate(keys):
        per_class_recall = []
        for cls in stealthy_names:
            cls_id = list(classes).index(cls)
            mask   = y_test == cls_id
            if mask.sum() == 0:
                per_class_recall.append(0.0)
            else:
                per_class_recall.append(float(flags_test[k][mask].mean()))
        ax.bar(x + (i-1)*w, per_class_recall, w,
               color=PALETTE[i+1], label=f"{k} ({ROLES.get(k, {}).get('description', 'Reconstruction MSE')[:20]})",
               alpha=0.85)
    # Ensemble (vote choisi)
    ens_recall = []
    for cls in stealthy_names:
        cls_id = list(classes).index(cls)
        mask   = y_test == cls_id
        ens_recall.append(float(flag_final_test[mask].mean()) if mask.sum()>0 else 0.0)
    ax.bar(x + w, ens_recall, w, color=PALETTE[0],
           label=f"Ensemble-{args.vote.upper()} (choisi)", alpha=1.0, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(stealthy_names, fontsize=10)
    ax.set_ylabel("Recall (taux de detection)"); ax.set_ylim([0, 1.1])
    ax.set_title("Recall par classe furtive — Detecteurs individuels vs Ensemble (Test)")
    ax.legend(fontsize=8); ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_ensemble_s1_stealthy.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Sauvegarde: fig_ensemble_s1_stealthy.png")

    # Résumé final
    test_r = chosen["test"]
    print("\n" + "="*60)
    print(f"RESULTAT ENSEMBLE ({args.vote.upper()}, >={vote_min}/3 detecteurs)")
    print("="*60)
    print(f"  Recall     : {test_r['recall']:.4f}")
    print(f"  FPR        : {test_r['fpr']:.4f}")
    print(f"  Enrichissement furtif : x{test_r['enrichment']:.4f}")
    enr = test_r['enrichment']
    h2_verdict = 'VALIDE' if enr >= 5 else f'REJETE ({enr:.2f}x)'
    print(f"  H2 (x5 cible) : {h2_verdict}")
    print(f"  Flux flagges : {test_r['n_flagged']:,} / "
          f"{len(y_test):,} ({test_r['flagged_pct']:.1f}%)")
    print("="*60)
    print("\nPour utiliser ces flags dans Stage 2 :")
    print("  Modifier ton_02_stage2_and_stats.py :")
    print("  s1_flag_tr = np.load('s1_ensemble_flag_train.npy')")
    print("  s1_sc_tr   = np.load('s1_ensemble_score_train.npy')")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ensemble Parallele Stage 1 (3 ou 4 detecteurs)")
    p.add_argument("--vote", choices=["or", "majority", "and"], default="majority",
                   help="Regle de vote : or, majority, and (defaut: majority)")
    p.add_argument("--with-ae", action="store_true",
                   help="Ajouter l'Autoencoder comme 4e detecteur (necessite train_autoencoder.py)")
    p.add_argument("--no-retrain", action="store_true",
                   help="Utiliser les scores en cache (evite de re-entrainer OCSVM/IF/LOF)")
    main(p.parse_args())
