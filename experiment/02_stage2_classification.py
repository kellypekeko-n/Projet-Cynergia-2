"""
Stage 2 — Supervised Classification + Hybrid Pipeline Evaluation.
Models: Random Forest, XGBoost, LightGBM, and standalone baselines.
Also runs ablation study A0-A5.
"""
import sys, os, io, time, json
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils.data_utils import load_and_split, stealthy_enrichment
from utils.metrics_utils import compute_all_metrics, save_metrics, bootstrap_ci

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import lightgbm as lgb

try:
    from imblearn.over_sampling import ADASYN
    HAS_ADASYN = True
except ImportError:
    HAS_ADASYN = False
    print("WARNING: imbalanced-learn not found. ADASYN disabled.")


def train_and_evaluate(model, X_train, y_train, X_test, y_test,
                        encoder, model_name, use_proba=True):
    t0 = time.time()
    model.fit(X_train, y_train)
    t_fit = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(X_test)
    t_inf = time.time()
    latency_ms = (t_inf - t1) / len(X_test) * 1000

    if use_proba and hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = np.eye(len(encoder.classes_))[y_pred]

    metrics = compute_all_metrics(
        y_test, y_pred, y_prob, encoder, model_name, latency_ms
    )
    metrics["fit_time_s"] = round(t_fit, 2)
    return metrics, y_pred, y_prob


def run_stage2():
    print("=" * 60)
    print("STAGE 2 — Classification + Hybrid Pipeline")
    print("=" * 60)

    (X_train_raw, X_val_raw, X_test_raw,
     y_train, y_val, y_test,
     scaler_mm, scaler_std,
     encoder, feature_names, split_info) = load_and_split()

    # Load Stage 1 artifacts
    flagged_train = np.load(os.path.join(METRICS_DIR, "s1_flagged_train.npy"))
    flagged_test  = np.load(os.path.join(METRICS_DIR, "s1_flagged_test.npy"))
    score_train   = np.load(os.path.join(METRICS_DIR, "s1_score_train.npy"))
    score_test    = np.load(os.path.join(METRICS_DIR, "s1_score_test.npy"))

    n_classes = len(encoder.classes_)
    class_names = list(encoder.classes_)

    print(f"\n  Train flagged by S1: {flagged_train.sum():,} / {len(flagged_train):,}")
    print(f"  Test  flagged by S1: {flagged_test.sum():,} / {len(flagged_test):,}")

    # ─── Build input variants ─────────────────────────────────────────────────
    # A0: Raw features, no preprocessing tuning (matches existing notebook style)
    X_tr_raw = X_train_raw  # already float
    X_ts_raw = X_test_raw

    # A2+: scaled (StandardScaler) — same scaler from load_and_split
    X_tr_std = scaler_std.transform(X_train_raw)
    X_ts_std = scaler_std.transform(X_test_raw)

    # Hybrid inputs: flagged + anomaly score appended
    X_tr_hyb = np.column_stack([X_tr_std[flagged_train],
                                  score_train[flagged_train].reshape(-1, 1)])
    y_tr_hyb  = y_train[flagged_train]
    X_ts_hyb  = np.column_stack([X_ts_std[flagged_test],
                                  score_test[flagged_test].reshape(-1, 1)])
    y_ts_hyb  = y_test[flagged_test]

    print(f"\n  Hybrid train: {len(X_tr_hyb):,} samples")
    print(f"  Hybrid test:  {len(X_ts_hyb):,} samples")

    # Rééquilibrage ADASYN sur hybride train
    if HAS_ADASYN:
        try:
            adasyn = ADASYN(random_state=SEEDS[0])
            X_tr_hyb_ada, y_tr_hyb_ada = adasyn.fit_resample(X_tr_hyb, y_tr_hyb)
            print(f"  After ADASYN: {len(X_tr_hyb_ada):,} samples")
        except Exception as e:
            print(f"  ADASYN failed: {e} — using original")
            X_tr_hyb_ada, y_tr_hyb_ada = X_tr_hyb, y_tr_hyb
    else:
        X_tr_hyb_ada, y_tr_hyb_ada = X_tr_hyb, y_tr_hyb

    # ─── Ablation study ──────────────────────────────────────────────────────
    print("\n--- ABLATION STUDY ---")
    ablation_results = []

    # A0 — XGBoost baseline (raw features, random-like split, no S1)
    print("\nA0: XGBoost baseline (raw, no S1)")
    mdl_a0 = XGBClassifier(n_estimators=100, random_state=SEEDS[0],
                             eval_metric='mlogloss', verbosity=0)
    m_a0, _, _ = train_and_evaluate(mdl_a0, X_tr_raw, y_train,
                                     X_ts_raw, y_test, encoder, "A0_XGB_baseline")
    ablation_results.append({"config": "A0: XGB (baseline)", **m_a0})
    print(f"  F1-macro={m_a0['f1_macro']:.4f} | AUC-PR={m_a0['auc_pr']:.4f}")

    # A1 — XGBoost + StandardScaler
    print("\nA1: XGBoost + Preprocessing")
    mdl_a1 = XGBClassifier(n_estimators=100, random_state=SEEDS[0],
                             eval_metric='mlogloss', verbosity=0)
    m_a1, _, _ = train_and_evaluate(mdl_a1, X_tr_std, y_train,
                                     X_ts_std, y_test, encoder, "A1_XGB_scaled")
    ablation_results.append({"config": "A1: XGB + Preprocessing", **m_a1})
    print(f"  F1-macro={m_a1['f1_macro']:.4f} | AUC-PR={m_a1['auc_pr']:.4f}")

    # A2 — ADASYN on full train (no Stage 1)
    print("\nA2: XGBoost + ADASYN (no Stage 1)")
    if HAS_ADASYN:
        try:
            ada_full = ADASYN(random_state=SEEDS[0])
            X_tr_ada_full, y_tr_ada_full = ada_full.fit_resample(X_tr_std, y_train)
            mdl_a2 = XGBClassifier(n_estimators=100, random_state=SEEDS[0],
                                    eval_metric='mlogloss', verbosity=0)
            m_a2, _, _ = train_and_evaluate(mdl_a2, X_tr_ada_full, y_tr_ada_full,
                                             X_ts_std, y_test, encoder, "A2_XGB_ADASYN")
        except Exception as e:
            print(f"  ADASYN failed on full train: {e}")
            m_a2 = m_a1.copy(); m_a2['model'] = "A2_XGB_ADASYN"
    else:
        m_a2 = m_a1.copy(); m_a2['model'] = "A2_XGB_ADASYN"
    ablation_results.append({"config": "A2: XGB + ADASYN (no S1)", **m_a2})
    print(f"  F1-macro={m_a2['f1_macro']:.4f} | AUC-PR={m_a2['auc_pr']:.4f}")

    # A3 — Stage 1 (LOF) + XGBoost (no ADASYN)
    print("\nA3: Stage 1 (LOF) + XGBoost (no ADASYN)")
    mdl_a3 = XGBClassifier(n_estimators=100, random_state=SEEDS[0],
                             eval_metric='mlogloss', verbosity=0)
    m_a3, _, _ = train_and_evaluate(mdl_a3, X_tr_hyb, y_tr_hyb,
                                     X_ts_hyb, y_ts_hyb, encoder,
                                     "A3_S1_XGB_noADASYN")
    ablation_results.append({"config": "A3: S1(LOF) + XGB", **m_a3})
    print(f"  F1-macro={m_a3['f1_macro']:.4f} | AUC-PR={m_a3['auc_pr']:.4f}")

    # A4 — Stage 1 + XGBoost + ADASYN
    print("\nA4: Stage 1 (LOF) + XGBoost + ADASYN")
    mdl_a4 = XGBClassifier(n_estimators=100, random_state=SEEDS[0],
                             eval_metric='mlogloss', verbosity=0)
    m_a4, _, _ = train_and_evaluate(mdl_a4, X_tr_hyb_ada, y_tr_hyb_ada,
                                     X_ts_hyb, y_ts_hyb, encoder,
                                     "A4_S1_XGB_ADASYN")
    ablation_results.append({"config": "A4: S1(LOF) + XGB + ADASYN", **m_a4})
    print(f"  F1-macro={m_a4['f1_macro']:.4f} | AUC-PR={m_a4['auc_pr']:.4f}")

    # ─── Full model comparison ────────────────────────────────────────────────
    print("\n--- FULL MODEL COMPARISON ---")
    stage2_results = {}

    # Standalone classifiers (no S1)
    standalone = {
        "XGBoost (standalone)": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=SEEDS[0], eval_metric='mlogloss', verbosity=0
        ),
        "Random Forest (standalone)": RandomForestClassifier(
            n_estimators=200, max_features='sqrt',
            class_weight='balanced', random_state=SEEDS[0], n_jobs=-1
        ),
        "LightGBM (standalone)": lgb.LGBMClassifier(
            n_estimators=200, num_leaves=31,
            class_weight='balanced', random_state=SEEDS[0],
            verbose=-1
        ),
    }

    for name, model in standalone.items():
        print(f"\n  {name}")
        m, ypred, yprob = train_and_evaluate(
            model, X_tr_std, y_train, X_ts_std, y_test, encoder, name
        )
        stage2_results[name] = {
            "metrics": m, "y_pred": ypred, "y_prob": yprob,
            "pipeline": "standalone"
        }
        stealthy_ids = [i for i, c in enumerate(class_names)
                        if c in STEALTHY_CLASSES]
        sr = {class_names[i]: round(m['stealthy_recalls'].get(class_names[i], 0), 4)
              for i in stealthy_ids}
        print(f"    F1-macro={m['f1_macro']:.4f} | AUC-PR={m['auc_pr']:.4f} | "
              f"MCC={m['mcc']:.4f}")
        print(f"    Stealthy recalls: {sr}")

    # Hybrid pipelines
    hybrid = {
        "Hybrid: LOF + XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=SEEDS[0], eval_metric='mlogloss', verbosity=0
        ),
        "Hybrid: LOF + Random Forest": RandomForestClassifier(
            n_estimators=200, max_features='sqrt',
            class_weight='balanced', random_state=SEEDS[0], n_jobs=-1
        ),
        "Hybrid: LOF + LightGBM": lgb.LGBMClassifier(
            n_estimators=200, num_leaves=31,
            class_weight='balanced', random_state=SEEDS[0], verbose=-1
        ),
    }

    for name, model in hybrid.items():
        print(f"\n  {name}")
        m, ypred, yprob = train_and_evaluate(
            model, X_tr_hyb_ada, y_tr_hyb_ada,
            X_ts_hyb, y_ts_hyb, encoder, name
        )
        stage2_results[name] = {
            "metrics": m, "y_pred": ypred, "y_prob": yprob,
            "pipeline": "hybrid"
        }
        stealthy_ids = [i for i, c in enumerate(class_names)
                        if c in STEALTHY_CLASSES]
        sr = {class_names[i]: round(m['stealthy_recalls'].get(class_names[i], 0), 4)
              for i in stealthy_ids}
        print(f"    F1-macro={m['f1_macro']:.4f} | AUC-PR={m['auc_pr']:.4f} | "
              f"MCC={m['mcc']:.4f}")
        print(f"    Stealthy recalls: {sr}")

    # ─── Bootstrap CI ─────────────────────────────────────────────────────────
    print("\n--- BOOTSTRAP CONFIDENCE INTERVALS ---")
    # Collect scores via repeated train/test on 5 seeds
    seed_scores = {name: [] for name in list(standalone.keys()) + list(hybrid.keys())}

    for seed in SEEDS:
        for name, model_cls in [
            ("XGBoost (standalone)",
             lambda s: XGBClassifier(n_estimators=100, random_state=s,
                                      eval_metric='mlogloss', verbosity=0)),
            ("Random Forest (standalone)",
             lambda s: RandomForestClassifier(n_estimators=100, random_state=s,
                                               class_weight='balanced', n_jobs=-1)),
            ("LightGBM (standalone)",
             lambda s: lgb.LGBMClassifier(n_estimators=100, random_state=s,
                                           class_weight='balanced', verbose=-1)),
            ("Hybrid: LOF + XGBoost",
             lambda s: XGBClassifier(n_estimators=100, random_state=s,
                                      eval_metric='mlogloss', verbosity=0)),
            ("Hybrid: LOF + Random Forest",
             lambda s: RandomForestClassifier(n_estimators=100, random_state=s,
                                               class_weight='balanced', n_jobs=-1)),
            ("Hybrid: LOF + LightGBM",
             lambda s: lgb.LGBMClassifier(n_estimators=100, random_state=s,
                                           class_weight='balanced', verbose=-1)),
        ]:
            model = model_cls(seed)
            if "Hybrid" in name:
                model.fit(X_tr_hyb_ada, y_tr_hyb_ada)
                yp = model.predict(X_ts_hyb)
                sc = f1_score(y_ts_hyb, yp, average='macro', zero_division=0)
            else:
                model.fit(X_tr_std, y_train)
                yp = model.predict(X_ts_std)
                sc = f1_score(y_test, yp, average='macro', zero_division=0)
            seed_scores[name].append(sc)

    ci_results = {}
    for name, scores in seed_scores.items():
        med, lo, hi = bootstrap_ci(np.array(scores))
        ci_results[name] = {"median": med, "ci_low": lo, "ci_high": hi,
                             "scores": scores}
        print(f"  {name[:40]:40s}: {med:.4f} [{lo:.4f}, {hi:.4f}]")

    # ─── Save all results ────────────────────────────────────────────────────
    to_save = {
        "stage2_metrics": {
            k: v["metrics"] for k, v in stage2_results.items()
        },
        "ablation": ablation_results,
        "bootstrap_ci": ci_results,
        "class_names": class_names,
        "n_test_hybrid": int(len(y_ts_hyb)),
        "n_test_full":   int(len(y_test)),
    }
    save_metrics(to_save, "stage2_results.json")

    # Save predictions for plotting
    for name, res in stage2_results.items():
        safe = name.lower().replace(" ", "_").replace(":", "").replace("+", "plus")
        np.save(os.path.join(METRICS_DIR, f"y_pred_{safe}.npy"), res["y_pred"])
        np.save(os.path.join(METRICS_DIR, f"y_prob_{safe}.npy"), res["y_prob"])

    np.save(os.path.join(METRICS_DIR, "y_ts_hyb.npy"),  y_ts_hyb)
    np.save(os.path.join(METRICS_DIR, "y_ts_full.npy"), y_test)

    print("\nStage 2 complete.")
    return stage2_results, ablation_results, ci_results


if __name__ == "__main__":
    run_stage2()
