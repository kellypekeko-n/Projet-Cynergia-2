"""
STEPS 1-3 — EDA + Temporal Split + Stage-1 Anomaly Detection on TON_IoT.
"""
import sys, os, io, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve, auc)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ton_iot_config import *

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                     'axes.titlesize': 11, 'figure.dpi': FIG_DPI})


# ─── Helpers ────────────────────────────────────────────────────────────────
def save_fig(fig, name):
    path_png = os.path.join(FIGURES_DIR, f"{name}.png")
    path_pdf = os.path.join(FIGURES_DIR, f"{name}.pdf")
    fig.savefig(path_png, dpi=FIG_DPI, bbox_inches='tight')
    fig.savefig(path_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.png / .pdf")


def calibrate_theta(scores_normal, scores_attack,
                     target_recall=STAGE1_TARGET_RECALL,
                     max_fpr=STAGE1_MAX_FPR):
    all_s = np.concatenate([scores_normal, scores_attack])
    all_y = np.concatenate([np.zeros(len(scores_normal)),
                             np.ones(len(scores_attack))])
    prec, rec, thresholds = precision_recall_curve(all_y, all_s)
    best_theta = None
    best_stats = {}
    for theta in sorted(np.unique(thresholds), reverse=True):
        tp = (scores_attack  >= theta).sum()
        fn = (scores_attack   < theta).sum()
        fp = (scores_normal  >= theta).sum()
        tn = (scores_normal   < theta).sum()
        recall    = tp / (tp + fn + 1e-12)
        fpr_val   = fp / (fp + tn + 1e-12)
        precision = tp / (tp + fp + 1e-12)
        if recall >= target_recall:
            best_theta = float(theta)
            best_stats = {"theta": best_theta, "recall": round(recall, 4),
                          "fpr": round(fpr_val, 4),
                          "precision": round(precision, 4),
                          "f1": round(2*precision*recall/(precision+recall+1e-9),4)}
            break
    if best_theta is None:
        best_theta = float(np.percentile(scores_normal, 95))
        best_stats = {"theta": best_theta, "recall": 0, "fpr": 0,
                      "precision": 0, "f1": 0,
                      "warning": "target recall not achievable"}
    return best_theta, best_stats


# ─── MAIN ───────────────────────────────────────────────────────────────────
def run():
    # ── Load dataset ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("TON_IoT — EDA + Stage-1 Anomaly Detection")
    print("=" * 60)

    print("\nLoading TON_IoT sample...")
    df = pd.read_csv(CACHE_CSV)
    df = df.sort_values("ts").reset_index(drop=True)

    with open(os.path.join(METRICS_DIR, "dataset_meta.json")) as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    class_dist    = meta["class_dist"]
    n_total       = len(df)

    print(f"  Loaded: {n_total:,} rows x {len(feature_names)} features")

    # ── EDA ──────────────────────────────────────────────────────────────────
    print("\n--- EDA ---")
    le = LabelEncoder()
    le.classes_ = np.array(sorted(df["type"].unique()))
    y_all = le.transform(df["type"])

    # EDA stats
    eda_stats = {
        "n_total":       n_total,
        "n_features":    len(feature_names),
        "class_counts":  {k: int((df["type"]==k).sum()) for k in le.classes_},
        "stealthy_ratio": float((df["type"].isin(STEALTHY_CLASSES)).mean()),
        "mitm_count":    int((df["type"]=="mitm").sum()),
        "missing_before": int(df[feature_names].isnull().sum().sum()),
    }
    eda_stats["class_pcts"] = {
        k: round(v/n_total*100, 2)
        for k, v in eda_stats["class_counts"].items()
    }
    print(f"  Stealthy ratio: {eda_stats['stealthy_ratio']:.4%}")
    print(f"  MitM count: {eda_stats['mitm_count']:,}")

    # Figure 1 — Class distribution (log scale)
    fig, ax = plt.subplots(figsize=(9, 4))
    classes  = list(eda_stats["class_counts"].keys())
    counts   = [eda_stats["class_counts"][c] for c in classes]
    colors   = [PALETTE[i % len(PALETTE)] for i in range(len(classes))]
    bars = ax.bar(classes, counts, color=colors, edgecolor='white', linewidth=0.5)
    for b, c in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.05,
                f'{c:,}', ha='center', va='bottom', fontsize=8, rotation=30)
    ax.set_yscale('log')
    ax.set_ylabel("Sample Count (log scale)")
    ax.set_title("TON_IoT Network Dataset — Class Distribution (sampled subset)")
    ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    save_fig(fig, "fig01_class_distribution")

    # Figure 2 — Imbalance ratio visualization
    total = sum(counts)
    pcts  = [c/total*100 for c in counts]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(classes, pcts, color=colors, edgecolor='white')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1,
               label='1% threshold')
    for b, p in zip(bars, pcts):
        ax.text(p + 0.1, b.get_y()+b.get_height()/2,
                f'{p:.2f}%', va='center', fontsize=9)
    ax.set_xlabel("Proportion of total samples (%)")
    ax.set_title("TON_IoT — Class Imbalance Analysis")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "fig02_class_imbalance")

    # Figure 3 — Feature correlation heatmap (numeric features subset)
    X_df = df[feature_names].copy()
    num_feats = [f for f in feature_names
                 if not any(f.startswith(p) for p in CAT_FEATURES)][:16]
    corr = X_df[num_feats].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True,
                linewidths=0.2, cbar_kws={"shrink": 0.8}, ax=ax,
                xticklabels=[f.replace('_', ' ') for f in num_feats],
                yticklabels=[f.replace('_', ' ') for f in num_feats])
    ax.set_title("Feature Correlation Matrix (numeric features)")
    plt.xticks(rotation=40, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    save_fig(fig, "fig03_feature_correlation")

    # ── Temporal split (per-class to ensure all splits have all classes) ─────
    # Each class is sorted by ts, then split 60/20/20 chronologically.
    # This preserves temporal ordering within each class while ensuring
    # all classes appear in all splits — critical for LOF calibration.
    print("\n--- Per-Class Temporal Split (60/20/20) ---")

    X_all  = df[feature_names].values.astype(np.float32)
    y_all  = le.transform(df["type"])

    train_idx, val_idx, test_idx = [], [], []
    for cls_id, cls_name in enumerate(le.classes_):
        idx = np.where(y_all == cls_id)[0]
        n_c = len(idx)
        i1_c = int(n_c * TRAIN_RATIO)
        i2_c = int(n_c * (TRAIN_RATIO + VAL_RATIO))
        train_idx.extend(idx[:i1_c])
        val_idx.extend(idx[i1_c:i2_c])
        test_idx.extend(idx[i2_c:])

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    test_idx  = np.array(test_idx)

    X_train_raw = X_all[train_idx];  y_train = y_all[train_idx]
    X_val_raw   = X_all[val_idx];    y_val   = y_all[val_idx]
    X_test_raw  = X_all[test_idx];   y_test  = y_all[test_idx]

    # Scalers (fit on train only)
    scaler_mm  = MinMaxScaler().fit(X_train_raw)
    scaler_std = StandardScaler().fit(X_train_raw)

    X_train_mm = scaler_mm.transform(X_train_raw)
    X_val_mm   = scaler_mm.transform(X_val_raw)
    X_test_mm  = scaler_mm.transform(X_test_raw)

    print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")

    # Normal-only train
    normal_idx    = (le.transform(["normal"])[0])
    X_normal_tr   = X_train_mm[y_train == normal_idx]
    X_val_normal  = X_val_mm[y_val == normal_idx]
    X_val_attack  = X_val_mm[y_val != normal_idx]
    print(f"  Normal train: {len(X_normal_tr):,}")
    print(f"  Val normal: {len(X_val_normal):,}  | Val attack: {len(X_val_attack):,}")

    split_info = {
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
        "n_normal_train": int(len(X_normal_tr)),
        "train_dist": {le.classes_[c]: int((y_train==c).sum())
                       for c in range(len(le.classes_))},
        "test_dist":  {le.classes_[c]: int((y_test==c).sum())
                       for c in range(len(le.classes_))},
    }

    # Save scalers info & splits
    np.save(os.path.join(METRICS_DIR, "X_train_mm.npy"),  X_train_mm)
    np.save(os.path.join(METRICS_DIR, "X_val_mm.npy"),    X_val_mm)
    np.save(os.path.join(METRICS_DIR, "X_test_mm.npy"),   X_test_mm)
    np.save(os.path.join(METRICS_DIR, "X_train_raw.npy"), X_train_raw)
    np.save(os.path.join(METRICS_DIR, "X_test_raw.npy"),  X_test_raw)
    np.save(os.path.join(METRICS_DIR, "y_train.npy"),     y_train)
    np.save(os.path.join(METRICS_DIR, "y_val.npy"),       y_val)
    np.save(os.path.join(METRICS_DIR, "y_test.npy"),      y_test)
    np.save(os.path.join(METRICS_DIR, "X_train_std.npy"),
            scaler_std.transform(X_train_raw))
    np.save(os.path.join(METRICS_DIR, "X_test_std.npy"),
            scaler_std.transform(X_test_raw))

    # Save label encoder classes
    with open(os.path.join(METRICS_DIR, "label_classes.json"), "w") as f:
        json.dump({"classes": list(le.classes_)}, f)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    print("\n--- Stage 1: Anomaly Detection ---")

    def stealthy_enrichment(y_full, y_flagged):
        stealthy_ids = [list(le.classes_).index(c)
                        for c in STEALTHY_CLASSES if c in le.classes_]
        r_before = np.isin(y_full, stealthy_ids).mean()
        r_after  = np.isin(y_flagged, stealthy_ids).mean()
        factor   = r_after / (r_before + 1e-12)
        return float(r_before), float(r_after), float(factor)

    models_s1 = {
        "Isolation Forest": IsolationForest(
            n_estimators=200, contamination=0.01,
            max_features=0.8, random_state=SEEDS[0], n_jobs=-1
        ),
        "One-Class SVM (RBF)": OneClassSVM(
            kernel='rbf', nu=0.05, gamma='scale'
        ),
        "LOF": LocalOutlierFactor(
            n_neighbors=20, contamination=0.05, novelty=True
        ),
    }

    stage1_results = {}
    roc_data_s1    = {}  # for plotting

    for name, model in models_s1.items():
        print(f"\n  [{name}]")
        t0 = time.time()
        model.fit(X_normal_tr)
        t_fit = time.time() - t0

        t1 = time.time()
        sc_val_n = -model.decision_function(X_val_normal)
        sc_val_a = -model.decision_function(X_val_attack)
        lat_ms   = (time.time()-t1) / len(X_val_mm) * 1000

        theta, th_stats = calibrate_theta(sc_val_n, sc_val_a)

        # Test set evaluation
        sc_test  = -model.decision_function(X_test_mm)
        flag_test = sc_test >= theta
        y_bin_test = (y_test != normal_idx).astype(int)
        y_pred_s1  = flag_test.astype(int)

        tp = int(((y_bin_test==1)&(y_pred_s1==1)).sum())
        fp = int(((y_bin_test==0)&(y_pred_s1==1)).sum())
        fn = int(((y_bin_test==1)&(y_pred_s1==0)).sum())
        tn = int(((y_bin_test==0)&(y_pred_s1==0)).sum())

        sc_val_all = -model.decision_function(X_val_mm)
        y_val_bin  = (y_val != normal_idx).astype(int)
        auc_roc_val = float(roc_auc_score(y_val_bin, sc_val_all))
        auc_pr_val  = float(average_precision_score(y_val_bin, sc_val_all))

        # Enrichment on test
        y_test_flagged = y_test[flag_test]
        r_before, r_after, factor = stealthy_enrichment(y_test, y_test_flagged)

        print(f"    Theta={theta:.4f} | Recall={th_stats['recall']:.3f} | "
              f"FPR={th_stats['fpr']:.4f}")
        print(f"    AUC-ROC={auc_roc_val:.4f} | AUC-PR={auc_pr_val:.4f}")
        print(f"    Enrichment: {r_before:.4%} -> {r_after:.4%}  (x{factor:.2f})")
        print(f"    Fit: {t_fit:.1f}s | Lat: {lat_ms:.4f}ms/s | "
              f"Flagged: {flag_test.sum():,}/{len(flag_test):,}")

        stage1_results[name] = {
            "theta":      theta,
            "recall":     th_stats["recall"],
            "fpr":        th_stats["fpr"],
            "precision":  th_stats["precision"],
            "f1":         th_stats["f1"],
            "auc_roc":    round(auc_roc_val, 4),
            "auc_pr":     round(auc_pr_val, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "enrichment": {"before": r_before, "after": r_after,
                           "factor": factor},
            "n_flagged":  int(flag_test.sum()),
            "flagged_pct":round(flag_test.mean()*100, 2),
            "latency_ms": round(lat_ms, 4),
            "fit_time_s": round(t_fit, 1),
        }
        roc_data_s1[name] = (y_val_bin, sc_val_all)

    # Select best Stage-1 model (recall >= 0.90 AND min FPR)
    valid = {k: v for k, v in stage1_results.items()
             if v["recall"] >= STAGE1_TARGET_RECALL
             and v["fpr"] <= STAGE1_MAX_FPR}
    best_s1 = (max(valid, key=lambda k: valid[k]["auc_roc"])
               if valid else
               max(stage1_results, key=lambda k: stage1_results[k]["recall"]))

    print(f"\n  Best Stage-1 model: {best_s1}")
    print(f"    Recall={stage1_results[best_s1]['recall']:.3f} | "
          f"FPR={stage1_results[best_s1]['fpr']:.4f} | "
          f"AUC-ROC={stage1_results[best_s1]['auc_roc']:.4f} | "
          f"Enrichment x{stage1_results[best_s1]['enrichment']['factor']:.2f}")

    # Save flagged indices using best model
    best_model = models_s1[best_s1]
    theta_best = stage1_results[best_s1]["theta"]

    sc_train = -best_model.decision_function(X_train_mm)
    sc_test_ = -best_model.decision_function(X_test_mm)
    flag_tr  = (sc_train >= theta_best)
    flag_ts  = (sc_test_ >= theta_best)

    np.save(os.path.join(METRICS_DIR, "s1_score_train.npy"), sc_train)
    np.save(os.path.join(METRICS_DIR, "s1_score_test.npy"),  sc_test_)
    np.save(os.path.join(METRICS_DIR, "s1_flag_train.npy"),  flag_tr)
    np.save(os.path.join(METRICS_DIR, "s1_flag_test.npy"),   flag_ts)

    # ── Stage-1 Figures ──────────────────────────────────────────────────────
    # Figure 4 — ROC curves Stage-1
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (name, (y_b, s)) in enumerate(roc_data_s1.items()):
        fpr_c, tpr_c, _ = roc_curve(y_b, s)
        roc_a = auc(fpr_c, tpr_c)
        ax.plot(fpr_c, tpr_c, color=PALETTE[i],
                label=f"{name} (AUC={roc_a:.3f})", linewidth=1.8)
    ax.plot([0,1],[0,1],'k--',alpha=0.4,linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Stage-1 ROC Curves — TON_IoT (Validation Set)")
    ax.legend(fontsize=9)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    plt.tight_layout()
    save_fig(fig, "fig04_stage1_roc")

    # Figure 5 — PR curves Stage-1
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (name, (y_b, s)) in enumerate(roc_data_s1.items()):
        prec_c, rec_c, _ = precision_recall_curve(y_b, s)
        ap = average_precision_score(y_b, s)
        ax.plot(rec_c, prec_c, color=PALETTE[i],
                label=f"{name} (AP={ap:.3f})", linewidth=1.8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Stage-1 Precision-Recall Curves — TON_IoT (Validation Set)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(fig, "fig05_stage1_pr")

    # Figure 6 — Enrichment bar chart
    s1_names  = list(stage1_results.keys())
    factors   = [stage1_results[n]["enrichment"]["factor"] for n in s1_names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(s1_names, factors,
                  color=[PALETTE[i] for i in range(len(s1_names))],
                  edgecolor='white')
    ax.axhline(y=1.0, color='red',   linestyle='--', linewidth=1.2,
               label='No enrichment (x1.0)')
    ax.axhline(y=5.0, color='green', linestyle=':',  linewidth=1.2,
               label='H2 target (x5.0)')
    for b, v in zip(bars, factors):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                f'x{v:.2f}', ha='center', fontweight='bold')
    ax.set_ylabel("Enrichment Factor")
    ax.set_title("Stage-1 Stealthy-Class Enrichment — TON_IoT")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "fig06_stage1_enrichment")

    # Figure 7 — Threshold calibration (best model)
    sc_val_n_ = -models_s1[best_s1].decision_function(X_val_normal)
    sc_val_a_ = -models_s1[best_s1].decision_function(X_val_attack)
    sc_all = np.concatenate([sc_val_n_, sc_val_a_])
    y_all_b = np.concatenate([np.zeros(len(sc_val_n_)),
                               np.ones(len(sc_val_a_))])
    prec_v, rec_v, thr_v = precision_recall_curve(y_all_b, sc_all)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rec_v[:-1], prec_v[:-1], color=PALETTE[0], linewidth=1.8,
            label='Precision-Recall')
    ax.axvline(x=stage1_results[best_s1]["recall"], color='red',
               linestyle='--', linewidth=1.2,
               label=f"Selected theta (Recall={stage1_results[best_s1]['recall']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Threshold Calibration — {best_s1} (Validation Set)")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "fig07_threshold_calibration")

    # ── Save results ─────────────────────────────────────────────────────────
    def sanitize(obj):
        if isinstance(obj, dict):   return {str(k): sanitize(v) for k,v in obj.items()}
        if isinstance(obj, list):   return [sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        return obj

    output = {
        "eda": eda_stats,
        "split": split_info,
        "stage1": stage1_results,
        "best_s1": best_s1,
        "class_names": list(le.classes_),
        "stealthy_classes": STEALTHY_CLASSES,
        "mitre_map": MITRE_MAP,
    }
    with open(os.path.join(METRICS_DIR, "eda_and_stage1.json"),
              "w", encoding="utf-8") as f:
        json.dump(sanitize(output), f, indent=2)

    print("\nStep 1-3 complete. All figures and metrics saved.")
    return output


if __name__ == "__main__":
    run()
