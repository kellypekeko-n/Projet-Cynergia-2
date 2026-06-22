"""
Score composite Stage 1 : α·OCSVM + β·IF + γ·LOF
Poids appris par optimisation sur ensemble de validation pour maximiser
l'enrichissement en attaques furtives sous contrainte recall ≥ 90% et FPR ≤ 15%.

Innovation principale : adresse le rejet de H2 (enrichissement 1.22× << 5×)

Usage:
  python experiment/models/stage1/train_composite_stage1.py
  python experiment/models/stage1/train_composite_stage1.py --no-retrain
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
                              roc_curve, precision_recall_curve, auc)
from scipy.optimize import differential_evolution

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ton_iot_config import (METRICS_DIR, FIGURES_DIR, STEALTHY_CLASSES,
                              STAGE1_TARGET_RECALL, STAGE1_MAX_FPR, SEEDS, PALETTE)

SCORES_CACHE  = os.path.join(METRICS_DIR, "composite_s1_scores.npz")
WEIGHTS_OUT   = os.path.join(METRICS_DIR, "composite_stage1_weights.json")
METRICS_OUT   = os.path.join(METRICS_DIR, "composite_s1_metrics.json")


# ── helpers ──────────────────────────────────────────────────────────────────

def calibrate_theta(sc_normal, sc_attack,
                    target_recall=STAGE1_TARGET_RECALL,
                    max_fpr=STAGE1_MAX_FPR):
    """Binary-search: recall ≥ target_recall, minimize FPR."""
    all_sc = np.concatenate([sc_normal, sc_attack])
    thresholds = np.unique(np.percentile(all_sc, np.linspace(0, 100, 3000)))
    best_theta, best = None, {"recall": 0.0, "fpr": 1.0, "precision": 0.0, "f1": 0.0}
    for th in sorted(thresholds, reverse=True):
        tp = int((sc_attack >= th).sum());  fn = int((sc_attack < th).sum())
        fp = int((sc_normal >= th).sum());  tn = int((sc_normal < th).sum())
        recall = tp / (tp + fn + 1e-12);   fpr  = fp / (fp + tn + 1e-12)
        prec   = tp / (tp + fp + 1e-12);   f1   = 2*prec*recall/(prec+recall+1e-12)
        if recall >= target_recall and fpr <= max_fpr:
            if best_theta is None or fpr < best["fpr"]:
                best_theta = th
                best = {"recall": recall, "fpr": fpr, "precision": prec, "f1": f1}
    if best_theta is None:
        # Relax FPR constraint: just hit target recall
        for th in sorted(thresholds, reverse=True):
            tp = int((sc_attack >= th).sum()); fn = int((sc_attack < th).sum())
            fp = int((sc_normal >= th).sum()); tn = int((sc_normal < th).sum())
            recall = tp/(tp+fn+1e-12);         fpr  = fp/(fp+tn+1e-12)
            if recall >= target_recall:
                return th, {"recall": recall, "fpr": fpr, "precision": 0.0, "f1": 0.0}
        best_theta = np.median(all_sc)
    return best_theta, best


def stealthy_enrichment(y_full, y_flagged, stealthy_ids):
    r_before = np.isin(y_full,    stealthy_ids).mean()
    r_after  = np.isin(y_flagged, stealthy_ids).mean()
    factor   = float(r_after / (r_before + 1e-12))
    return float(r_before), float(r_after), factor


def normalize01(sc_ref, *others):
    """Min-max scale using sc_ref statistics; apply same scale to others."""
    lo, hi = sc_ref.min(), sc_ref.max()
    scaled = [(s - lo) / (hi - lo + 1e-12) for s in (sc_ref, *others)]
    return scaled


def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name + ".png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure: {os.path.basename(path)}")


# ── training ─────────────────────────────────────────────────────────────────

def train_models(X_normal, seed=SEEDS[0]):
    print("  Training Isolation Forest...", end=" ", flush=True)
    t = time.time()
    IF = IsolationForest(n_estimators=200, contamination=0.01,
                         max_features=0.8, random_state=seed, n_jobs=-1).fit(X_normal)
    print(f"{time.time()-t:.1f}s")

    print("  Training One-Class SVM...", end=" ", flush=True)
    t = time.time()
    OCSVM = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale').fit(X_normal)
    print(f"{time.time()-t:.1f}s")

    print("  Training LOF...", end=" ", flush=True)
    t = time.time()
    LOF = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True).fit(X_normal)
    print(f"{time.time()-t:.1f}s")

    return {"IF": IF, "OCSVM": OCSVM, "LOF": LOF}


def compute_raw_scores(models, X):
    """Higher score = more anomalous (negate decision_function)."""
    return {name: -m.decision_function(X) for name, m in models.items()}


# ── composite optimization ────────────────────────────────────────────────────

def make_composite(scores_dict, weights, keys=("OCSVM", "IF", "LOF")):
    """Weighted linear combination of normalized scores."""
    return sum(weights[i] * scores_dict[k] for i, k in enumerate(keys))


def objective(weights, val_normal_sc, val_attack_sc, val_all_sc, y_val, stealthy_ids):
    """
    Minimize negative enrichment factor on validation set.
    Returns +1e9 if recall < target or FPR > max.
    """
    w = np.abs(weights)
    w = w / (w.sum() + 1e-12)

    sc_n = make_composite(val_normal_sc, w)
    sc_a = make_composite(val_attack_sc, w)
    sc_all = make_composite(val_all_sc, w)

    theta, stats = calibrate_theta(sc_n, sc_a)
    if stats["recall"] < STAGE1_TARGET_RECALL - 0.01 or stats["fpr"] > STAGE1_MAX_FPR + 0.02:
        return 1e9

    flag_val = sc_all >= theta
    y_flagged = y_val[flag_val]
    _, _, factor = stealthy_enrichment(y_val, y_flagged, stealthy_ids)
    return -factor


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(scores_normal, scores_attack, scores_all, y_all,
                   stealthy_ids, label):
    """Full evaluation: AUC-ROC, AUC-PR, recall/FPR, enrichment."""
    y_bin = (y_all != -1).astype(int)  # not used directly — passed separately
    theta, stats = calibrate_theta(scores_normal, scores_attack)
    flag = scores_all >= theta
    _, _, factor = stealthy_enrichment(y_all, y_all[flag], stealthy_ids)

    y_val_bin = np.concatenate([
        np.zeros(len(scores_normal)), np.ones(len(scores_attack))
    ])
    sc_val_combined = np.concatenate([scores_normal, scores_attack])
    auc_roc = float(roc_auc_score(y_val_bin, sc_val_combined))
    auc_pr  = float(average_precision_score(y_val_bin, sc_val_combined))
    return {
        "label":       label,
        "theta":       float(theta),
        "recall":      round(stats["recall"],    4),
        "fpr":         round(stats["fpr"],       4),
        "precision":   round(stats["precision"], 4),
        "f1":          round(stats["f1"],        4),
        "auc_roc":     round(auc_roc,            4),
        "auc_pr":      round(auc_pr,             4),
        "enrichment":  round(factor,             4),
        "n_flagged":   int(flag.sum()),
        "flagged_pct": round(flag.mean()*100, 2),
    }, (y_val_bin, sc_val_combined)


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\n=== Score Composite Stage 1 : α·OCSVM + β·IF + γ·LOF ===\n")

    # 1. Load preprocessed data
    print("--- Chargement données ---")
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

    X_normal_tr  = X_train_mm[y_train == normal_idx]
    X_val_normal = X_val_mm[y_val == normal_idx]
    X_val_attack = X_val_mm[y_val != normal_idx]
    print(f"  Normal train: {len(X_normal_tr):,}  |  Val: {len(X_val_mm):,}  |  Test: {len(X_test_mm):,}")

    # 2. Train or load models
    if not args.no_retrain or not os.path.exists(SCORES_CACHE):
        print("\n--- Entraînement des modèles Stage 1 ---")
        models = train_models(X_normal_tr)

        print("\n--- Calcul des scores anomalie ---")
        # Raw (unnormalized) per-model scores
        raw_val_n   = compute_raw_scores(models, X_val_normal)
        raw_val_a   = compute_raw_scores(models, X_val_attack)
        raw_val_all = compute_raw_scores(models, X_val_mm)
        raw_test    = compute_raw_scores(models, X_test_mm)
        raw_train   = compute_raw_scores(models, X_train_mm)

        np.savez(SCORES_CACHE,
                 **{f"val_n_{k}":   v for k, v in raw_val_n.items()},
                 **{f"val_a_{k}":   v for k, v in raw_val_a.items()},
                 **{f"val_all_{k}": v for k, v in raw_val_all.items()},
                 **{f"test_{k}":    v for k, v in raw_test.items()},
                 **{f"train_{k}":   v for k, v in raw_train.items()})
        print(f"  Scores sauvegardés: {SCORES_CACHE}")
    else:
        print(f"\n--- Chargement scores en cache ({SCORES_CACHE}) ---")
        data = np.load(SCORES_CACHE)
        keys = ("OCSVM", "IF", "LOF")
        raw_val_n   = {k: data[f"val_n_{k}"]   for k in keys}
        raw_val_a   = {k: data[f"val_a_{k}"]   for k in keys}
        raw_val_all = {k: data[f"val_all_{k}"] for k in keys}
        raw_test    = {k: data[f"test_{k}"]    for k in keys}
        raw_train   = {k: data[f"train_{k}"]   for k in keys}

    # 3. Normalize scores per model [0, 1] using training set statistics
    keys = ("OCSVM", "IF", "LOF")
    val_n_sc, val_a_sc, val_all_sc, test_sc, train_sc = {}, {}, {}, {}, {}
    for k in keys:
        ref = raw_val_n[k]
        lo, hi = ref.min(), ref.max()
        scale = lambda s: (s - lo) / (hi - lo + 1e-12)
        val_n_sc[k]   = scale(raw_val_n[k])
        val_a_sc[k]   = scale(raw_val_a[k])
        val_all_sc[k] = scale(raw_val_all[k])
        test_sc[k]    = scale(raw_test[k])
        train_sc[k]   = scale(raw_train[k])

    # 4. Évaluation des modèles individuels (sur validation → comparaison équitable)
    print("\n--- Évaluation modèles individuels ---")
    individual_results = {}
    roc_data = {}
    for k in keys:
        res, roc = evaluate_model(
            val_n_sc[k], val_a_sc[k], val_all_sc[k], y_val, stealthy_ids, k
        )
        individual_results[k] = res
        roc_data[k] = roc
        print(f"  [{k}] recall={res['recall']:.3f} | fpr={res['fpr']:.4f} | "
              f"AUC-ROC={res['auc_roc']:.4f} | enrichissement x{res['enrichment']:.2f}")

    # 5. Optimisation des poids sur validation
    print("\n--- Optimisation des poids (differential_evolution) ---")
    print("  Objectif : maximiser enrichissement furtif sous recall ≥ 0.90, FPR ≤ 0.15")

    bounds = [(0.0, 1.0)] * 3
    t0 = time.time()
    result = differential_evolution(
        objective,
        bounds,
        args=(val_n_sc, val_a_sc, val_all_sc, y_val, stealthy_ids),
        seed=SEEDS[0],
        maxiter=200,
        popsize=15,
        tol=1e-5,
        mutation=(0.5, 1.5),
        recombination=0.7,
        workers=1,
        disp=False,
    )
    t_opt = time.time() - t0

    raw_w = np.abs(result.x)
    opt_weights = raw_w / raw_w.sum()
    alpha, beta, gamma = opt_weights
    print(f"  Optimisation terminée en {t_opt:.1f}s")
    print(f"  Poids optimaux : α(OCSVM)={alpha:.4f}, β(IF)={beta:.4f}, γ(LOF)={gamma:.4f}")
    print(f"  Score objectif (négatif enrichissement val) : {result.fun:.4f}")

    # 6. Évaluation du composite sur validation
    sc_composite_val_n   = make_composite(val_n_sc,   opt_weights)
    sc_composite_val_a   = make_composite(val_a_sc,   opt_weights)
    sc_composite_val_all = make_composite(val_all_sc, opt_weights)
    sc_composite_test    = make_composite(test_sc,    opt_weights)
    sc_composite_train   = make_composite(train_sc,   opt_weights)

    composite_val, roc_composite = evaluate_model(
        sc_composite_val_n, sc_composite_val_a, sc_composite_val_all,
        y_val, stealthy_ids, "Composite"
    )
    roc_data["Composite"] = roc_composite
    print(f"\n  [Composite] recall={composite_val['recall']:.3f} | "
          f"fpr={composite_val['fpr']:.4f} | "
          f"AUC-ROC={composite_val['auc_roc']:.4f} | "
          f"enrichissement x{composite_val['enrichment']:.2f}")

    # 7. Évaluation finale sur test set (estimation non biaisée)
    print("\n--- Évaluation finale sur test set ---")
    theta_test = composite_val["theta"]
    flag_test  = sc_composite_test >= theta_test
    y_test_flagged = y_test[flag_test]
    r_before, r_after, factor_test = stealthy_enrichment(y_test, y_test_flagged, stealthy_ids)

    y_test_bin = (y_test != normal_idx).astype(int)
    sc_test_combined = np.concatenate([
        make_composite(val_n_sc, opt_weights),  # use val_normal as proxy
        sc_composite_test
    ])
    # Proper AUC on test
    sc_full_test = sc_composite_test
    auc_roc_test = float(roc_auc_score(y_test_bin, sc_full_test))
    auc_pr_test  = float(average_precision_score(y_test_bin, sc_full_test))

    flag_ts_full = sc_full_test >= theta_test
    tp = int(((y_test_bin==1)&flag_ts_full).sum())
    fp = int(((y_test_bin==0)&flag_ts_full).sum())
    fn = int(((y_test_bin==1)&~flag_ts_full).sum())
    tn = int(((y_test_bin==0)&~flag_ts_full).sum())
    recall_test  = tp/(tp+fn+1e-12)
    fpr_test     = fp/(fp+tn+1e-12)

    print(f"  Recall={recall_test:.3f} | FPR={fpr_test:.4f} | "
          f"AUC-ROC={auc_roc_test:.4f} | AUC-PR={auc_pr_test:.4f}")
    print(f"  Enrichissement furtif test : {r_before:.4%} → {r_after:.4%} "
          f"(×{factor_test:.2f})")
    print(f"  Flux flaggés : {flag_ts_full.sum():,}/{len(flag_ts_full):,} "
          f"({flag_ts_full.mean()*100:.1f}%)")

    # 8. Comparaison vs meilleur modèle individuel
    best_ind = max(individual_results.values(), key=lambda x: x["enrichment"])
    print(f"\n  Meilleur modèle individuel : {best_ind['label']} "
          f"(enrichissement x{best_ind['enrichment']:.2f})")
    gain = factor_test - best_ind["enrichment"]
    print(f"  Gain composite : +{gain:+.2f}× enrichissement")

    # 9. Sauvegarde scores composites
    np.save(os.path.join(METRICS_DIR, "s1_composite_score_train.npy"), sc_composite_train)
    np.save(os.path.join(METRICS_DIR, "s1_composite_score_test.npy"),  sc_composite_test)
    np.save(os.path.join(METRICS_DIR, "s1_composite_flag_train.npy"),
            sc_composite_train >= theta_test)
    np.save(os.path.join(METRICS_DIR, "s1_composite_flag_test.npy"),  flag_ts_full)
    print(f"\n  Scores composites sauvegardés (s1_composite_score_{{train,test}}.npy)")

    # 10. Sauvegarde poids + métriques
    weights_out = {
        "alpha_OCSVM": float(alpha),
        "beta_IF":     float(beta),
        "gamma_LOF":   float(gamma),
        "theta":       float(theta_test),
        "opt_time_s":  round(t_opt, 1),
    }
    with open(WEIGHTS_OUT, "w") as f:
        json.dump(weights_out, f, indent=2)

    ind_test_results = {}
    for k in keys:
        sc_k_test = test_sc[k]
        theta_k   = individual_results[k]["theta"]
        flag_k    = sc_k_test >= theta_k
        _, _, fac = stealthy_enrichment(y_test, y_test[flag_k], stealthy_ids)
        tp_k = int(((y_test_bin==1)&flag_k).sum()); fn_k = int(((y_test_bin==1)&~flag_k).sum())
        fp_k = int(((y_test_bin==0)&flag_k).sum()); tn_k = int(((y_test_bin==0)&~flag_k).sum())
        ind_test_results[k] = {
            "recall":     round(tp_k/(tp_k+fn_k+1e-12), 4),
            "fpr":        round(fp_k/(fp_k+tn_k+1e-12), 4),
            "auc_roc":    round(float(roc_auc_score(y_test_bin, sc_k_test)), 4),
            "enrichment": round(fac, 4),
        }

    metrics_out = {
        "individual_val":  {k: v for k, v in individual_results.items()},
        "composite_val":   composite_val,
        "individual_test": ind_test_results,
        "composite_test": {
            "recall":      round(recall_test,    4),
            "fpr":         round(fpr_test,       4),
            "auc_roc":     round(auc_roc_test,   4),
            "auc_pr":      round(auc_pr_test,    4),
            "enrichment":  round(factor_test,    4),
            "n_flagged":   int(flag_ts_full.sum()),
            "flagged_pct": round(flag_ts_full.mean()*100, 2),
        },
        "weights": weights_out,
        "gain_vs_best_individual": round(gain, 4),
        "h2_enrichment_target":    5.0,
        "h2_validated":            factor_test >= 5.0,
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Métriques sauvegardées: {os.path.basename(METRICS_OUT)}")

    # 11. Figures
    print("\n--- Génération des figures ---")

    # Figure A — ROC curves (3 individuels + composite)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors_map = {"OCSVM": PALETTE[1], "IF": PALETTE[2], "LOF": PALETTE[3],
                  "Composite": PALETTE[0]}
    styles     = {"OCSVM": "--", "IF": "-.", "LOF": ":", "Composite": "-"}
    lwidths    = {"OCSVM": 1.4, "IF": 1.4, "LOF": 1.4, "Composite": 2.2}
    for name, (y_b, sc_b) in roc_data.items():
        fpr_c, tpr_c, _ = roc_curve(y_b, sc_b)
        ra = auc(fpr_c, tpr_c)
        ax.plot(fpr_c, tpr_c, color=colors_map[name], linestyle=styles[name],
                linewidth=lwidths[name], label=f"{name} (AUC={ra:.3f})")
    ax.plot([0,1],[0,1],'k--',alpha=0.4,linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Composite Stage 1 vs Modèles Individuels — ROC (Val)")
    ax.legend(fontsize=9); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    plt.tight_layout()
    save_fig(fig, "fig_composite_s1_roc")

    # Figure B — Enrichissement comparatif (val + test)
    model_names = list(keys) + ["Composite"]
    enr_val  = [individual_results[k]["enrichment"] for k in keys] + [composite_val["enrichment"]]
    enr_test = [ind_test_results[k]["enrichment"]   for k in keys] + [factor_test]
    x = np.arange(len(model_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_v = ax.bar(x - w/2, enr_val,  w, color=[colors_map[n] for n in model_names],
                    alpha=0.6, label='Validation', edgecolor='white')
    bars_t = ax.bar(x + w/2, enr_test, w, color=[colors_map[n] for n in model_names],
                    alpha=1.0, label='Test',       edgecolor='white')
    for b, v in zip(bars_v, enr_val):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"×{v:.2f}",
                ha='center', va='bottom', fontsize=8, color='gray')
    for b, v in zip(bars_t, enr_test):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"×{v:.2f}",
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.axhline(y=1.0, color='gray',  linestyle='--', linewidth=1,   label="Pas d'enrichissement (x1.0)")
    ax.axhline(y=5.0, color='green', linestyle=':',  linewidth=1.5, label="Objectif H2 (x5.0)")
    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel("Facteur d'enrichissement furtif"); ax.legend(fontsize=9)
    ax.set_title("Enrichissement furtif — Modèles individuels vs Score Composite")
    plt.tight_layout()
    save_fig(fig, "fig_composite_s1_enrichment")

    # Figure C — Poids optimaux (pie chart)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    pie_labels  = [f"OCSVM\n(α={alpha:.3f})", f"IF\n(β={beta:.3f})", f"LOF\n(γ={gamma:.3f})"]
    pie_colors  = [PALETTE[1], PALETTE[2], PALETTE[3]]
    ax1.pie([alpha, beta, gamma], labels=pie_labels, colors=pie_colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    ax1.set_title("Poids optimaux α·OCSVM + β·IF + γ·LOF")

    metrics_compare = {
        "AUC-ROC": [individual_results[k]["auc_roc"] for k in keys] + [composite_val["auc_roc"]],
        "Recall":  [individual_results[k]["recall"]  for k in keys] + [composite_val["recall"]],
        "FPR":     [individual_results[k]["fpr"]     for k in keys] + [composite_val["fpr"]],
    }
    x2 = np.arange(len(model_names))
    for i, (metric, vals) in enumerate(metrics_compare.items()):
        offset = (i - 1) * 0.25
        bars = ax2.bar(x2 + offset, vals, 0.22,
                       label=metric, alpha=0.85,
                       color=PALETTE[4+i], edgecolor='white')
    ax2.set_xticks(x2); ax2.set_xticklabels(model_names, fontsize=9)
    ax2.set_ylim([0, 1.1]); ax2.set_ylabel("Score")
    ax2.set_title("Métriques comparatives (Validation)")
    ax2.legend(fontsize=9); ax2.axhline(0.9, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    save_fig(fig, "fig_composite_s1_weights")

    # Summary
    print("\n" + "="*60)
    print("RÉSUMÉ SCORE COMPOSITE STAGE 1")
    print("="*60)
    print(f"  Poids : α(OCSVM)={alpha:.3f}  β(IF)={beta:.3f}  γ(LOF)={gamma:.3f}")
    print(f"  Enrichissement test : ×{factor_test:.3f}  "
          f"(vs ×{best_ind['enrichment']:.3f} meilleur individuel)")
    print(f"  H2 (enrichissement ≥5×) : {'VALIDÉ ✓' if factor_test >= 5.0 else f'REJETÉ ({factor_test:.2f}×)'}")
    print(f"  Recall test : {recall_test:.3f}  |  FPR test : {fpr_test:.4f}")
    print(f"  AUC-ROC test : {auc_roc_test:.4f}")
    print("="*60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Score composite Stage 1")
    p.add_argument("--no-retrain", action="store_true",
                   help="Utiliser les scores en cache (évite de ré-entraîner les modèles)")
    main(p.parse_args())
