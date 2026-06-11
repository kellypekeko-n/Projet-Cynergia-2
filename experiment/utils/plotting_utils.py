"""
Figure generation utilities — all paper figures.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix
)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIGURES_DIR, FIG_DPI, PALETTE, ALL_CLASSES

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': FIG_DPI,
})


def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {path}")
    return path


def plot_class_distribution(label_counts, title="Class Distribution"):
    fig, ax = plt.subplots(figsize=(8, 4))
    names  = list(label_counts.keys())
    counts = list(label_counts.values())
    bars = ax.bar(names, counts, color=PALETTE[:len(names)], edgecolor='white')
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{cnt:,}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel("Attack Class")
    ax.set_ylabel("Sample Count")
    ax.set_title(title)
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    return _save(fig, "fig_class_distribution")


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, vmin=0, vmax=1, ax=ax)
    ax.set_ylabel("True Label (MITRE)")
    ax.set_xlabel("Predicted Label")
    ax.set_title(f"Normalized Confusion Matrix — {model_name}")
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    name = model_name.lower().replace(" ", "_").replace("+", "plus")
    return _save(fig, f"fig_cm_{name}")


def plot_roc_curves(results_dict, class_names):
    """results_dict: {model_name: (y_true, y_prob)}"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, (y_true, y_prob)) in enumerate(results_dict.items()):
        n_classes = y_prob.shape[1]
        y_bin = np.eye(n_classes)[y_true]
        fpr_all, tpr_all = [], []
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
            fpr_all.append(fpr); tpr_all.append(tpr)
        mean_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.mean([np.interp(mean_fpr, f, t)
                            for f, t in zip(fpr_all, tpr_all)], axis=0)
        roc_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, color=PALETTE[i % len(PALETTE)],
                label=f"{name} (AUC={roc_auc:.3f})", linewidth=1.8)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Stage-2 Classifiers (Macro OvR)")
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    return _save(fig, "fig_roc_comparison")


def plot_pr_curves(results_dict, stealthy_class_idx, stealthy_name):
    """Precision-Recall for a single stealthy class across all models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, (y_true, y_prob)) in enumerate(results_dict.items()):
        y_bin = (y_true == stealthy_class_idx).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, y_prob[:, stealthy_class_idx])
        ap = average_precision_score(y_bin, y_prob[:, stealthy_class_idx])
        ax.plot(rec, prec, color=PALETTE[i % len(PALETTE)],
                label=f"{name} (AP={ap:.3f})", linewidth=1.8)
    baseline = y_bin.mean()
    ax.axhline(baseline, color='gray', linestyle='--', linewidth=1,
               label=f"No-skill (P={baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {stealthy_name}")
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    safe = stealthy_name.lower().replace("-", "_").replace(" ", "_")
    return _save(fig, f"fig_pr_{safe}")


def plot_ablation(ablation_results):
    """ablation_results: list of {'config': str, 'f1_macro': float, 'auc_pr': float}"""
    names  = [r['config'] for r in ablation_results]
    f1s    = [r['f1_macro'] for r in ablation_results]
    aucs   = [r['auc_pr']   for r in ablation_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    bars1 = ax1.barh(names, f1s, color=PALETTE[:len(names)], edgecolor='white')
    ax1.set_xlabel("F1-macro")
    ax1.set_title("Ablation Study — F1-macro")
    ax1.set_xlim([0, 1.05])
    for bar, v in zip(bars1, f1s):
        ax1.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{v:.3f}', va='center', fontsize=9)

    bars2 = ax2.barh(names, aucs, color=PALETTE[:len(names)], edgecolor='white')
    ax2.set_xlabel("AUC-PR (macro)")
    ax2.set_title("Ablation Study — AUC-PR")
    ax2.set_xlim([0, 1.05])
    for bar, v in zip(bars2, aucs):
        ax2.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    return _save(fig, "fig_ablation")


def plot_enrichment(enrichment_dict):
    """enrichment_dict: {model_name: factor}"""
    names   = list(enrichment_dict.keys())
    factors = list(enrichment_dict.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, factors, color=PALETTE[:len(names)], edgecolor='white')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1,
               label='No enrichment (×1.0)')
    ax.axhline(y=5.0, color='green', linestyle=':', linewidth=1.2,
               label='H2 threshold (×5.0)')
    for bar, v in zip(bars, factors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'×{v:.1f}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel("Enrichment Factor (after/before)")
    ax.set_title("Stage-1 Enrichment of Stealthy Classes")
    ax.legend()
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    return _save(fig, "fig_enrichment")


def plot_impact_curve(impact_data):
    """
    impact_data: {
        'proportions': [0.01, 0.05, ...],
        'standalone':  [f1_1, f1_2, ...],
        'hybrid':      [f1_1, f1_2, ...]
    }
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    props = [p * 100 for p in impact_data['proportions']]
    ax.plot(props, impact_data['standalone'], 'o--', color='#e74c3c',
            label='Best standalone classifier', linewidth=1.8)
    ax.plot(props, impact_data['hybrid'],     's-',  color='#2ecc71',
            label='Best hybrid pipeline',      linewidth=1.8)
    real_prop = impact_data.get('real_proportion', None)
    if real_prop:
        ax.axvline(x=real_prop * 100, color='gray', linestyle=':',
                   label=f'Real dataset proportion ({real_prop*100:.1f}%)')
    ax.set_xlabel("Stealthy Attacks as % of Total Traffic")
    ax.set_ylabel("F1-macro")
    ax.set_title("Impact of Class Imbalance — Hybrid vs. Standalone")
    ax.legend()
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    return _save(fig, "fig_impact_curve")
