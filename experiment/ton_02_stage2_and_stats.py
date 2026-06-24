"""
STEPS 4-7 — Stage-2, Hybrid Pipeline, Ablation, Statistical Tests.
Dataset: TON_IoT Network (536K samples, 10 classes).
Optimized for reasonable execution time.
"""
import sys, os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ton_iot_config import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from xgboost import XGBClassifier
import lightgbm as lgb
from scipy.stats import wilcoxon, friedmanchisquare
from statsmodels.stats.multitest import multipletests

try:
    from imblearn.over_sampling import ADASYN
    HAS_ADASYN = True
except ImportError:
    HAS_ADASYN = False

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})


def save_fig(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"),
                    dpi=FIG_DPI if ext == 'png' else None,
                    bbox_inches='tight')
    plt.close(fig)
    print(f"  [FIG] {name}.png/.pdf")


def sanitize(obj):
    if isinstance(obj, dict):            return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):            return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    if isinstance(obj, np.bool_):        return bool(obj)
    return obj


def compute_metrics(y_true, y_pred, y_prob, class_names, model_name,
                     latency_ms=None):
    n_cls = len(class_names)
    f1m   = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    f1w   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    acc   = accuracy_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)

    # Restrict y_prob to classes present in test
    y_bin = np.eye(n_cls)[y_true]
    try:
        auc_roc = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
    except Exception:
        auc_roc = float('nan')
    try:
        auc_pr = average_precision_score(y_bin, y_prob, average='macro')
    except Exception:
        auc_pr = float('nan')

    rpt = classification_report(y_true, y_pred,
                                 target_names=class_names,
                                 output_dict=True, zero_division=0)
    stealthy_recalls = {}
    for cls in STEALTHY_CLASSES:
        if cls in class_names and cls in rpt:
            stealthy_recalls[cls] = round(rpt[cls]['recall'], 4)

    per_class = {}
    for c in class_names:
        if c in rpt:
            per_class[c] = {k: round(v, 4)
                            for k, v in rpt[c].items()
                            if k in ('precision', 'recall', 'f1-score', 'support')}

    return {
        "model":            model_name,
        "accuracy":         round(acc, 4),
        "f1_macro":         round(f1m, 4),
        "f1_weighted":      round(f1w, 4),
        "auc_roc":          round(auc_roc, 4),
        "auc_pr":           round(auc_pr, 4),
        "mcc":              round(mcc, 4),
        "stealthy_recalls": stealthy_recalls,
        "per_class":        per_class,
        "latency_ms":       round(latency_ms, 4) if latency_ms else None,
    }


def train_eval(model, X_tr, y_tr, X_ts, y_ts, class_names, name):
    print(f"  Training {name}...", flush=True)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    t_fit = time.time() - t0
    t1 = time.time()
    yp  = model.predict(X_ts)
    lat = (time.time() - t1) / max(len(X_ts), 1) * 1000
    ypr = (model.predict_proba(X_ts)
           if hasattr(model, 'predict_proba')
           else np.eye(len(class_names))[yp])
    m = compute_metrics(y_ts, yp, ypr, class_names, name, lat)
    m["fit_time_s"] = round(t_fit, 1)
    print(f"    F1-macro={m['f1_macro']:.4f} | AUC-PR={m['auc_pr']:.4f} "
          f"| MCC={m['mcc']:.4f} | fit={t_fit:.0f}s", flush=True)
    return m, yp, ypr


def bootstrap_ci(scores, n=1000, alpha=0.05, seed=42):
    rng  = np.random.default_rng(seed)
    boot = np.array([
        rng.choice(scores, len(scores), replace=True).mean()
        for _ in range(n)
    ])
    return (float(np.median(scores)),
            float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def run(use_composite: bool = False, use_ensemble: str = None):
    print("=" * 60, flush=True)
    print("TON_IoT — Stage-2 + Hybrid + Ablation + Stats", flush=True)
    if use_composite:
        print("  [MODE] Stage 1 composite (α·OCSVM + β·IF + γ·LOF)", flush=True)
    if use_ensemble:
        print(f"  [MODE] Stage 1 ensemble parallèle ({use_ensemble.upper()})", flush=True)
    print("=" * 60, flush=True)

    # ── Load everything ───────────────────────────────────────────────────────
    with open(os.path.join(METRICS_DIR, "eda_and_stage1.json")) as f:
        meta = json.load(f)
    class_names = meta["class_names"]   # 10 classes sorted
    best_s1     = meta["best_s1"]
    n_cls       = len(class_names)
    normal_id   = class_names.index("normal")

    X_tr_std = np.load(os.path.join(METRICS_DIR, "X_train_std.npy"))
    X_ts_std = np.load(os.path.join(METRICS_DIR, "X_test_std.npy"))
    y_train  = np.load(os.path.join(METRICS_DIR, "y_train.npy"))
    y_test   = np.load(os.path.join(METRICS_DIR, "y_test.npy"))

    # Stage 1 scores — 3 modes : OCSVM seul (défaut), composite, ensemble parallèle
    _ens_flag_tr = os.path.join(METRICS_DIR, "s1_ensemble_flag_train.npy")
    _ens_sc_tr   = os.path.join(METRICS_DIR, "s1_ensemble_score_train.npy")
    _ens_flag_ts = os.path.join(METRICS_DIR, "s1_ensemble_flag_test.npy")
    _ens_sc_ts   = os.path.join(METRICS_DIR, "s1_ensemble_score_test.npy")
    _has_ensemble = all(os.path.exists(p) for p in
                        [_ens_flag_tr, _ens_sc_tr, _ens_flag_ts, _ens_sc_ts])

    composite_flag_tr = os.path.join(METRICS_DIR, "s1_composite_flag_train.npy")
    composite_sc_tr   = os.path.join(METRICS_DIR, "s1_composite_score_train.npy")
    composite_flag_ts = os.path.join(METRICS_DIR, "s1_composite_flag_test.npy")
    composite_sc_ts   = os.path.join(METRICS_DIR, "s1_composite_score_test.npy")
    _has_composite = all(os.path.exists(p) for p in
                         [composite_flag_tr, composite_sc_tr,
                          composite_flag_ts, composite_sc_ts])

    if use_ensemble and _has_ensemble:
        s1_flag_tr = np.load(_ens_flag_tr)
        s1_flag_ts = np.load(_ens_flag_ts)
        s1_sc_tr   = np.load(_ens_sc_tr)
        s1_sc_ts   = np.load(_ens_sc_ts)
        best_s1    = f"Ensemble-{use_ensemble.upper()} (OCSVM+IF+LOF vote)"
        print(f"  Stage 1 ensemble ({use_ensemble.upper()}) chargé.", flush=True)
    elif use_ensemble and not _has_ensemble:
        print("  [WARN] Scores ensemble non trouvés — "
              "exécuter train_ensemble_stage1.py d'abord. Fallback OCSVM.", flush=True)
        use_ensemble = None

    if not use_ensemble and use_composite and _has_composite:
        s1_flag_tr = np.load(composite_flag_tr)
        s1_flag_ts = np.load(composite_flag_ts)
        s1_sc_tr   = np.load(composite_sc_tr)
        s1_sc_ts   = np.load(composite_sc_ts)
        best_s1    = "Composite (α·OCSVM+β·IF+γ·LOF)"
        print("  Stage 1 composite chargé.", flush=True)
    elif not use_ensemble and use_composite and not _has_composite:
        print("  [WARN] Scores composites non trouvés — "
              "exécuter train_composite_stage1.py d'abord. Fallback OCSVM.", flush=True)

    if not use_ensemble and not (use_composite and _has_composite):
        s1_flag_tr = np.load(os.path.join(METRICS_DIR, "s1_flag_train.npy"))
        s1_flag_ts = np.load(os.path.join(METRICS_DIR, "s1_flag_test.npy"))
        s1_sc_tr   = np.load(os.path.join(METRICS_DIR, "s1_score_train.npy"))
        s1_sc_ts   = np.load(os.path.join(METRICS_DIR, "s1_score_test.npy"))

    print(f"\nDataset: {len(y_train):,} train | {len(y_test):,} test", flush=True)
    print(f"S1 flagged: {s1_flag_tr.sum():,}/{len(s1_flag_tr):,} train "
          f"({s1_flag_tr.mean()*100:.1f}%)", flush=True)

    # ── Hybrid inputs ─────────────────────────────────────────────────────────
    X_tr_hyb = np.column_stack([
        X_tr_std[s1_flag_tr], s1_sc_tr[s1_flag_tr].reshape(-1, 1)
    ])
    y_tr_hyb = y_train[s1_flag_tr]
    X_ts_hyb = np.column_stack([
        X_ts_std[s1_flag_ts], s1_sc_ts[s1_flag_ts].reshape(-1, 1)
    ])
    y_ts_hyb = y_test[s1_flag_ts]

    print(f"Hybrid train: {len(X_tr_hyb):,} | Hybrid test: {len(X_ts_hyb):,}",
          flush=True)

    # Class distribution in hybrid train — check if ADASYN needed
    from collections import Counter
    hyb_dist = Counter(y_tr_hyb)
    print(f"Hybrid train class distribution:", flush=True)
    for cls_id, cnt in sorted(hyb_dist.items()):
        print(f"  {class_names[cls_id]:12s}: {cnt:7,} ({cnt/len(y_tr_hyb)*100:.1f}%)",
              flush=True)

    # Apply ADASYN only if mitm (most rare) has < 500 samples
    mitm_id  = class_names.index("mitm") if "mitm" in class_names else -1
    need_ada = mitm_id >= 0 and hyb_dist.get(mitm_id, 0) < 500

    if need_ada and HAS_ADASYN:
        print("Applying ADASYN (rare classes detected)...", flush=True)
        t0 = time.time()
        try:
            ada = ADASYN(random_state=SEEDS[0], n_neighbors=3)
            X_tr_hyb_ada, y_tr_hyb_ada = ada.fit_resample(X_tr_hyb, y_tr_hyb)
            print(f"  ADASYN: {len(X_tr_hyb)} -> {len(X_tr_hyb_ada)} ({time.time()-t0:.0f}s)",
                  flush=True)
        except Exception as e:
            print(f"  ADASYN failed: {e} — using original", flush=True)
            X_tr_hyb_ada, y_tr_hyb_ada = X_tr_hyb, y_tr_hyb
    else:
        print("ADASYN skipped (classes well-represented in flagged subset).",
              flush=True)
        X_tr_hyb_ada, y_tr_hyb_ada = X_tr_hyb, y_tr_hyb

    # ADASYN on full train for ablation A2
    if HAS_ADASYN:
        print("Applying ADASYN to full train for ablation A2...", flush=True)
        try:
            ada_full = ADASYN(random_state=SEEDS[0], n_neighbors=3)
            X_ada_full, y_ada_full = ada_full.fit_resample(X_tr_std, y_train)
            print(f"  Full ADASYN: {len(y_train)} -> {len(y_ada_full)}", flush=True)
        except Exception as e:
            print(f"  Full ADASYN failed: {e}", flush=True)
            X_ada_full, y_ada_full = X_tr_std, y_train
    else:
        X_ada_full, y_ada_full = X_tr_std, y_train

    # ── Ablation Study ────────────────────────────────────────────────────────
    print("\n--- ABLATION STUDY (XGBoost backbone, n_est=100) ---", flush=True)
    ablation = []

    XGB_FAST = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=SEEDS[0], eval_metric='mlogloss',
                    verbosity=0, n_jobs=-1)

    abl_configs = [
        ("A0: XGB baseline",        X_tr_std,    y_train,     X_ts_std,  y_test),
        ("A1: XGB + anomaly score",
         np.column_stack([X_tr_std, s1_sc_tr.reshape(-1, 1)]), y_train,
         np.column_stack([X_ts_std, s1_sc_ts.reshape(-1, 1)]), y_test),
        ("A2: XGB + ADASYN",        X_ada_full,  y_ada_full,  X_ts_std,  y_test),
        ("A3: S1 filter only",      X_tr_hyb,    y_tr_hyb,    X_ts_hyb,  y_ts_hyb),
        ("A4: S1 + ADASYN",         X_tr_hyb_ada,y_tr_hyb_ada,X_ts_hyb, y_ts_hyb),
        ("A5: S1 + score + ADASYN", X_tr_hyb_ada,y_tr_hyb_ada,X_ts_hyb, y_ts_hyb),
    ]

    for name, Xtr, ytr, Xts, yts in abl_configs:
        m, _, _ = train_eval(XGBClassifier(**XGB_FAST), Xtr, ytr, Xts, yts,
                              class_names, name)
        row = {"config": name, "f1_macro": m["f1_macro"],
               "auc_pr": m["auc_pr"], "mcc": m["mcc"],
               "stealthy_recalls": m["stealthy_recalls"]}
        ablation.append(row)

    # ── Full evaluation (n_est=200) ───────────────────────────────────────────
    print("\n--- FULL MODEL COMPARISON (n_est=200) ---", flush=True)
    XGB_FULL = dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=SEEDS[0], eval_metric='mlogloss',
                    verbosity=0, n_jobs=-1)
    RF_FULL  = dict(n_estimators=200, max_features='sqrt',
                    class_weight='balanced', random_state=SEEDS[0], n_jobs=-1)
    LGB_FULL = dict(n_estimators=200, num_leaves=31, class_weight='balanced',
                    random_state=SEEDS[0], verbose=-1)

    print("\n[Standalone]", flush=True)
    full_results = {}
    pred_store   = {}

    for name, ModelCls, kw, Xtr, ytr, Xts, yts in [
        ("XGBoost (standalone)",   XGBClassifier, XGB_FULL,
         X_tr_std, y_train, X_ts_std, y_test),
        ("Random Forest (standalone)", RandomForestClassifier, RF_FULL,
         X_tr_std, y_train, X_ts_std, y_test),
        ("LightGBM (standalone)",  lgb.LGBMClassifier, LGB_FULL,
         X_tr_std, y_train, X_ts_std, y_test),
    ]:
        m, yp, ypr = train_eval(ModelCls(**kw), Xtr, ytr, Xts, yts,
                                  class_names, name)
        full_results[name] = {"metrics": m, "pipeline": "standalone"}
        pred_store[name]   = {"y_true": yts, "y_pred": yp, "y_prob": ypr}
        sr = {c: m["stealthy_recalls"].get(c, 0) for c in STEALTHY_CLASSES}
        print(f"  Stealthy recalls: {sr}", flush=True)

    print("\n[Hybrid: {best_s1} + classifier]".replace('{best_s1}', best_s1),
          flush=True)
    for name, ModelCls, kw in [
        (f"{best_s1} + XGBoost",       XGBClassifier,       XGB_FULL),
        (f"{best_s1} + Random Forest",  RandomForestClassifier, RF_FULL),
        (f"{best_s1} + LightGBM",       lgb.LGBMClassifier,  LGB_FULL),
    ]:
        m, yp, ypr = train_eval(
            ModelCls(**kw), X_tr_hyb_ada, y_tr_hyb_ada,
            X_ts_hyb, y_ts_hyb, class_names, name
        )
        full_results[name] = {"metrics": m, "pipeline": "hybrid"}
        pred_store[name]   = {"y_true": y_ts_hyb, "y_pred": yp, "y_prob": ypr}
        sr = {c: m["stealthy_recalls"].get(c, 0) for c in STEALTHY_CLASSES}
        print(f"  Stealthy recalls: {sr}", flush=True)

    # ── Statistical tests — 10 seeds, subsampled (80K) ───────────────────────
    print("\n--- STATISTICAL TESTS (10 seeds, subsample 80K) ---", flush=True)
    N_SUB   = 80_000
    rng_sub = np.random.default_rng(0)

    # Precompute fixed subsample indices (same across seeds for comparability)
    sub_idx = rng_sub.choice(len(y_train), size=min(N_SUB, len(y_train)),
                               replace=False)
    Xs_tr = X_tr_std[sub_idx]
    ys_tr = y_train[sub_idx]

    # Hybrid subsample
    flag_sub = s1_flag_tr[sub_idx]
    Xs_hyb   = np.column_stack([
        Xs_tr[flag_sub], s1_sc_tr[sub_idx][flag_sub].reshape(-1, 1)
    ])
    ys_hyb = ys_tr[flag_sub]

    seed_scores = {n: [] for n in list(full_results.keys())}

    XGB_CI  = dict(n_estimators=50, max_depth=6, learning_rate=0.1,
                   eval_metric='mlogloss', verbosity=0, n_jobs=-1)
    RF_CI   = dict(n_estimators=50, max_features='sqrt',
                   class_weight='balanced', n_jobs=-1)
    LGB_CI  = dict(n_estimators=50, num_leaves=31,
                   class_weight='balanced', verbose=-1)

    for seed_idx, seed in enumerate(SEEDS):
        print(f"  Seed {seed} ({seed_idx+1}/10)...", flush=True)
        rng_seed = np.random.default_rng(seed)

        # Standalone
        for sname, ModelCls, kw in [
            ("XGBoost (standalone)",       XGBClassifier,       {**XGB_CI, "random_state": seed}),
            ("Random Forest (standalone)", RandomForestClassifier, {**RF_CI, "random_state": seed}),
            ("LightGBM (standalone)",      lgb.LGBMClassifier,  {**LGB_CI, "random_state": seed}),
        ]:
            m = ModelCls(**kw)
            m.fit(Xs_tr, ys_tr)
            yp = m.predict(X_ts_std)
            sc = f1_score(y_test, yp, average='macro', zero_division=0)
            seed_scores[sname].append(sc)

        # Hybrid with per-seed ADASYN if needed
        if need_ada and HAS_ADASYN:
            try:
                ada_s = ADASYN(random_state=seed, n_neighbors=3)
                Xs_h_ada, ys_h_ada = ada_s.fit_resample(Xs_hyb, ys_hyb)
            except Exception:
                Xs_h_ada, ys_h_ada = Xs_hyb, ys_hyb
        else:
            Xs_h_ada, ys_h_ada = Xs_hyb, ys_hyb

        for hname, ModelCls, kw in [
            (f"{best_s1} + XGBoost",       XGBClassifier,       {**XGB_CI, "random_state": seed}),
            (f"{best_s1} + Random Forest",  RandomForestClassifier, {**RF_CI, "random_state": seed}),
            (f"{best_s1} + LightGBM",       lgb.LGBMClassifier,  {**LGB_CI, "random_state": seed}),
        ]:
            m = ModelCls(**kw)
            m.fit(Xs_h_ada, ys_h_ada)
            yp = m.predict(X_ts_hyb)
            sc = f1_score(y_ts_hyb, yp, average='macro', zero_division=0)
            seed_scores[hname].append(sc)

    # Bootstrap CI
    print("\nBootstrap CI (n_boot=1000):", flush=True)
    ci_results = {}
    for name, scores in seed_scores.items():
        med, lo, hi = bootstrap_ci(np.array(scores))
        ci_results[name] = {"median": med, "ci_low": lo, "ci_high": hi,
                             "scores": scores}
        print(f"  {name[:48]:48s}: {med:.4f} [{lo:.4f}, {hi:.4f}]", flush=True)

    # Friedman test
    model_names = list(seed_scores.keys())
    scores_mat  = np.array([seed_scores[m] for m in model_names])
    stat_f, p_f = friedmanchisquare(*scores_mat)
    print(f"\nFriedman chi2={stat_f:.3f}, p={p_f:.6f} "
          f"({'SIGNIFICANT' if p_f < 0.05 else 'NOT SIGNIFICANT'})", flush=True)

    # Wilcoxon pairwise + Bonferroni
    wilcoxon_res, p_vals, comps = {}, [], []
    for hname in [f"{best_s1} + XGBoost",
                   f"{best_s1} + Random Forest",
                   f"{best_s1} + LightGBM"]:
        for saname in ["XGBoost (standalone)",
                        "Random Forest (standalone)",
                        "LightGBM (standalone)"]:
            comps.append((hname, saname))
            try:
                st, p = wilcoxon(seed_scores[hname], seed_scores[saname])
                p_vals.append(p)
                wilcoxon_res[f"{hname} vs {saname}"] = {
                    "h_median": float(np.median(seed_scores[hname])),
                    "s_median": float(np.median(seed_scores[saname])),
                    "delta":    float(np.median(seed_scores[hname])
                                       - np.median(seed_scores[saname])),
                    "stat": float(st), "p_value": float(p),
                }
            except Exception as e:
                p_vals.append(1.0)
                wilcoxon_res[f"{hname} vs {saname}"] = {"error": str(e)}

    if p_vals:
        rejected, p_bonf, _, _ = multipletests(p_vals, method='bonferroni')
        for i, (h, s) in enumerate(comps):
            key = f"{h} vs {s}"
            if key in wilcoxon_res and "p_value" in wilcoxon_res[key]:
                wilcoxon_res[key]["p_bonf"] = float(p_bonf[i])
                wilcoxon_res[key]["sig"]    = bool(rejected[i])

    print("\nWilcoxon results (selected):", flush=True)
    for k, v in list(wilcoxon_res.items())[:3]:
        if "p_value" in v:
            print(f"  {k}: delta={v['delta']:+.4f} "
                  f"p={v['p_value']:.5f} p_bonf={v.get('p_bonf', 1):.5f} "
                  f"{'*' if v.get('sig') else 'ns'}", flush=True)

    # ── Hypotheses ────────────────────────────────────────────────────────────
    xgb_m  = full_results["XGBoost (standalone)"]["metrics"]
    lgb_m  = full_results["LightGBM (standalone)"]["metrics"]
    hyb_xm = full_results[f"{best_s1} + XGBoost"]["metrics"]
    hyb_lm = full_results[f"{best_s1} + LightGBM"]["metrics"]

    # Récupérer l'enrichissement depuis la bonne source selon le mode Stage 1
    if best_s1 in meta.get("stage1", {}):
        enr = meta["stage1"][best_s1]["enrichment"]
    elif use_composite and os.path.exists(os.path.join(METRICS_DIR, "composite_s1_metrics.json")):
        with open(os.path.join(METRICS_DIR, "composite_s1_metrics.json")) as _f:
            _cm = json.load(_f)
        _factor = _cm["composite_test"]["enrichment"]
        enr = {"factor": _factor, "before": 1.0, "after": _factor}
    elif use_ensemble and os.path.exists(os.path.join(METRICS_DIR, "ensemble_stage1_metrics.json")):
        with open(os.path.join(METRICS_DIR, "ensemble_stage1_metrics.json")) as _f:
            _em = json.load(_f)
        _vote_key = use_ensemble if use_ensemble in _em.get("results_by_vote", {}) else "majority"
        _factor = _em["results_by_vote"][_vote_key]["test"]["enrichment"]
        enr = {"factor": _factor, "before": 1.0, "after": _factor}
    else:
        # Fallback : OCSVM original
        _orig = meta["best_s1"]
        enr = meta["stage1"][_orig]["enrichment"]

    h1 = {
        "H1_f1_delta": round(hyb_lm["f1_macro"] - lgb_m["f1_macro"], 4),
        "H1_verdict":  "VALIDATED" if hyb_lm["f1_macro"] >= lgb_m["f1_macro"] else "REJECTED",
        "stealthy": {}
    }
    for cls in STEALTHY_CLASSES:
        sa = xgb_m["stealthy_recalls"].get(cls, 0)
        hy = hyb_xm["stealthy_recalls"].get(cls, 0)
        h1["stealthy"][cls] = {
            "standalone": sa, "hybrid": hy, "delta": round(hy - sa, 4),
            "verdict": "VALIDATED" if hy > sa else "REJECTED"
        }

    h2 = {
        "factor": round(enr["factor"], 4),
        "before": round(enr["before"], 4),
        "after":  round(enr["after"],  4),
        "verdict": "VALIDATED" if enr["factor"] >= 5 else "PARTIALLY VALIDATED",
    }

    print(f"\nH1 F1-macro delta: {h1['H1_f1_delta']:+.4f} → {h1['H1_verdict']}",
          flush=True)
    print(f"H2 Enrichment: x{h2['factor']:.2f} → {h2['verdict']}", flush=True)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n--- FIGURES ---", flush=True)

    # Fig 8 — Confusion matrices
    for name, data in pred_store.items():
        yt, yp = data["y_true"], data["y_pred"]
        present = sorted(np.unique(np.concatenate([yt, yp])))
        pnames  = [class_names[i] for i in present]
        cm      = confusion_matrix(yt, yp, labels=present)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=pnames, yticklabels=pnames,
                    linewidths=0.3, vmin=0, vmax=1, ax=ax)
        ax.set_title(f"Normalized Confusion Matrix — {name}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.xticks(rotation=30, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        safe = (name.lower().replace(" ", "_").replace("(", "")
                    .replace(")", "").replace("/", "_")[:42])
        save_fig(fig, f"fig08_cm_{safe}")

    # Fig 9 — ROC curves standalone
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, sname in enumerate(["XGBoost (standalone)",
                                 "Random Forest (standalone)",
                                 "LightGBM (standalone)"]):
        d     = pred_store[sname]
        y_bin = np.eye(n_cls)[d["y_true"]]
        fprs, tprs = [], []
        for c in range(n_cls):
            if y_bin[:, c].sum() > 0:
                f, t, _ = roc_curve(y_bin[:, c], d["y_prob"][:, c])
                fprs.append(f); tprs.append(t)
        mfpr = np.linspace(0, 1, 200)
        mtpr = np.mean([np.interp(mfpr, f, t) for f, t in zip(fprs, tprs)],
                        axis=0)
        ax.plot(mfpr, mtpr, color=PALETTE[i],
                label=f"{sname.replace(' (standalone)', '')} "
                      f"(AUC={auc(mfpr, mtpr):.3f})", linewidth=1.8)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
    ax.set(xlabel="FPR", ylabel="TPR",
           title="ROC Curves — Stage-2 Standalone Classifiers (TON_IoT)",
           xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(fig, "fig09_stage2_roc")

    # Fig 10 — PR curves per stealthy class (standalone)
    for cls in STEALTHY_CLASSES:
        if cls not in class_names: continue
        cls_id = class_names.index(cls)
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, sname in enumerate(["XGBoost (standalone)",
                                    "Random Forest (standalone)",
                                    "LightGBM (standalone)"]):
            d     = pred_store[sname]
            y_bin = (d["y_true"] == cls_id).astype(int)
            if y_bin.sum() == 0:
                continue
            p_, r_, _ = precision_recall_curve(y_bin, d["y_prob"][:, cls_id])
            ap = average_precision_score(y_bin, d["y_prob"][:, cls_id])
            ax.plot(r_, p_, color=PALETTE[i],
                    label=f"{sname.replace(' (standalone)', '')} (AP={ap:.3f})",
                    linewidth=1.8)
        bl = (y_test == cls_id).mean()
        ax.axhline(bl, color='gray', linestyle='--', lw=1,
                   label=f"No-skill ({bl:.4f})")
        ax.set(xlabel="Recall", ylabel="Precision",
               title=f"PR Curve — {cls} | {MITRE_MAP.get(cls, '')}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        save_fig(fig, f"fig10_pr_{cls}")

    # Fig 11 — Ablation barplot
    names_abl = [a["config"] for a in ablation]
    f1s_abl   = [a["f1_macro"] for a in ablation]
    aucs_abl  = [a["auc_pr"] for a in ablation]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names_abl))]
    b1 = ax1.barh(names_abl, f1s_abl, color=colors, edgecolor='white')
    ax1.set_xlabel("F1-macro")
    ax1.set_title("Ablation — F1-macro (TON_IoT)")
    for b, v in zip(b1, f1s_abl):
        ax1.text(v + 0.003, b.get_y() + b.get_height() / 2,
                 f"{v:.4f}", va='center', fontsize=9)
    b2 = ax2.barh(names_abl, aucs_abl, color=colors, edgecolor='white')
    ax2.set_xlabel("AUC-PR (macro)")
    ax2.set_title("Ablation — AUC-PR (TON_IoT)")
    for b, v in zip(b2, aucs_abl):
        ax2.text(v + 0.003, b.get_y() + b.get_height() / 2,
                 f"{v:.4f}", va='center', fontsize=9)
    plt.tight_layout()
    save_fig(fig, "fig11_ablation")

    # Fig 12 — Main comparison barplot
    all_n = list(full_results.keys())
    all_f = [full_results[n]["metrics"]["f1_macro"] for n in all_n]
    cols  = [PALETTE[0]] * 3 + [PALETTE[1]] * 3
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(all_n)), all_f, color=cols, edgecolor='white')
    ax.set_xticks(range(len(all_n)))
    ax.set_xticklabels([n.replace("One-Class SVM (RBF) + ", "OCSVM+")
                        for n in all_n], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel("F1-macro")
    ax.set_title("Performance Comparison — Standalone vs. Hybrid (TON_IoT)")
    for b, v in zip(bars, all_f):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.4f}", ha='center', fontsize=8)
    from matplotlib.patches import Patch
    ax.legend([Patch(color=PALETTE[0]), Patch(color=PALETTE[1])],
              ["Standalone", "Hybrid"], loc="lower right")
    plt.tight_layout()
    save_fig(fig, "fig12_main_comparison")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "stage2":           {n: v["metrics"] for n, v in full_results.items()},
        "ablation":         ablation,
        "bootstrap_ci":     ci_results,
        "statistical_tests": {
            "friedman": {"stat": float(stat_f), "p": float(p_f),
                         "significant": bool(p_f < 0.05)},
            "wilcoxon":  wilcoxon_res,
        },
        "hypotheses":       {"H1": h1, "H2": h2},
        "n_test_full":       int(len(y_test)),
        "n_test_hybrid":     int(len(y_ts_hyb)),
        "class_names":       class_names,
        "best_s1":           best_s1,
    }

    with open(os.path.join(METRICS_DIR, "stage2_and_stats.json"),
              "w", encoding="utf-8") as f:
        json.dump(sanitize(output), f, indent=2)

    # Save predictions
    for name, data in pred_store.items():
        safe = (name.lower().replace(" ", "_").replace("(", "")
                    .replace(")", "").replace("/", "_")[:42])
        np.save(os.path.join(METRICS_DIR, f"yp_{safe}.npy"),  data["y_pred"])
        np.save(os.path.join(METRICS_DIR, f"ypr_{safe}.npy"), data["y_prob"])
    np.save(os.path.join(METRICS_DIR, "y_ts_hyb.npy"), y_ts_hyb)

    print("\nAll results saved. Stage 2 complete.", flush=True)
    return output


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="TON_IoT Stage-2 Pipeline")
    p.add_argument("--use-composite", action="store_true",
                   help="Utiliser les scores composites Stage 1 (α·OCSVM+β·IF+γ·LOF)")
    p.add_argument("--use-ensemble", choices=["or", "majority", "and"], default=None,
                   help="Utiliser l'ensemble parallèle Stage 1 (or/majority/and)")
    args = p.parse_args()
    run(use_composite=args.use_composite, use_ensemble=args.use_ensemble)
