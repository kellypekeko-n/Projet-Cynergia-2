"""
Autoencoder (AE) — 4e détecteur Stage 1
Réseau de neurones entraîné à reconstruire le trafic NORMAL.
Erreur de reconstruction élevée = flux anormal.

Architecture :
  Encoder : features → 64 → 32 → 16 → bottleneck(8)
  Decoder : 8        → 16 → 32 → 64 → features

Pourquoi plus puissant que OCSVM/IF/LOF :
  - Capture des relations NON-LINÉAIRES entre features
  - Apprend des représentations compactes du trafic normal
  - L'erreur de reconstruction est plus informative qu'un score de distance
  - Très utilisé dans les papiers ICS/IIoT (IEEE TII, TIFS)

Score d'anomalie : erreur MSE par flux (plus élevé = plus anormal)

Usage:
  python train_autoencoder.py
  python train_autoencoder.py --epochs 100 --bottleneck 8
"""
import os, sys, json, time, warnings, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ton_iot_config import (METRICS_DIR, FIGURES_DIR, STEALTHY_CLASSES,
                              STAGE1_TARGET_RECALL, STAGE1_MAX_FPR, SEEDS, PALETTE)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

SCORES_OUT   = os.path.join(METRICS_DIR, "ae_s1_scores.npz")
METRICS_OUT  = os.path.join(METRICS_DIR, "autoencoder_stage1_metrics.json")
MODEL_OUT    = os.path.join(METRICS_DIR, "..", "saved_models", "autoencoder_stage1.pt")


# ── Architecture ──────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    """
    Autoencoder symétrique pour détection d'anomalies.
    Entraîné sur trafic normal uniquement.
    Score anomalie = MSE(x, reconstruit(x))
    """
    def __init__(self, n_features: int, bottleneck: int = 8,
                 hidden: tuple = (64, 32, 16)):
        super().__init__()
        # Encoder
        enc_layers = []
        in_dim = n_features
        for h in hidden:
            enc_layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        enc_layers += [nn.Linear(in_dim, bottleneck), nn.ReLU()]
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (symétrique)
        dec_layers = []
        in_dim = bottleneck
        for h in reversed(hidden):
            dec_layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        dec_layers += [nn.Linear(in_dim, n_features), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """MSE par flux — score d'anomalie."""
        with torch.no_grad():
            x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)


# ── Numpy fallback (si pas de torch) ─────────────────────────────────────────

class SimpleAENumpyFallback:
    """
    Autoencoder PCA-based (fallback si PyTorch non disponible).
    Reconstruction via PCA truncated → score d'anomalie.
    """
    def __init__(self, n_components=8):
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=n_components)

    def fit(self, X):
        self.pca.fit(X)
        return self

    def reconstruction_error(self, X):
        X_low  = self.pca.transform(X)
        X_hat  = self.pca.inverse_transform(X_low)
        return np.mean((X - X_hat) ** 2, axis=1)

    def decision_function(self, X):
        return -self.reconstruction_error(X)   # négatif pour aligner avec sklearn


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_theta(sc_normal, sc_attack, target_recall=STAGE1_TARGET_RECALL,
                    max_fpr=STAGE1_MAX_FPR):
    all_sc = np.concatenate([sc_normal, sc_attack])
    thresholds = np.unique(np.percentile(all_sc, np.linspace(0, 100, 3000)))
    best_theta, best = None, {"recall": 0.0, "fpr": 1.0, "precision": 0.0, "f1": 0.0}
    for th in sorted(thresholds, reverse=True):
        tp = (sc_attack >= th).sum();  fn = (sc_attack < th).sum()
        fp = (sc_normal >= th).sum();  tn = (sc_normal < th).sum()
        recall = tp/(tp+fn+1e-12); fpr = fp/(fp+tn+1e-12)
        prec   = tp/(tp+fp+1e-12); f1  = 2*prec*recall/(prec+recall+1e-12)
        if recall >= target_recall and fpr <= max_fpr:
            if best_theta is None or fpr < best["fpr"]:
                best_theta = th
                best = {"recall": float(recall), "fpr": float(fpr),
                        "precision": float(prec), "f1": float(f1)}
    if best_theta is None:
        for th in sorted(thresholds, reverse=True):
            if (sc_attack >= th).mean() >= target_recall:
                return th, {"recall": float((sc_attack>=th).mean()),
                            "fpr": float((sc_normal>=th).mean()),
                            "precision": 0.0, "f1": 0.0}
        best_theta = np.median(all_sc)
    return best_theta, best


def stealthy_enrichment(y_full, y_flagged, stealthy_ids):
    r_before = np.isin(y_full,    stealthy_ids).mean()
    r_after  = np.isin(y_flagged, stealthy_ids).mean()
    return float(r_before), float(r_after), float(r_after/(r_before+1e-12))


# ── Entraînement PyTorch ──────────────────────────────────────────────────────

def train_torch_ae(X_normal: np.ndarray, n_features: int,
                   bottleneck: int, epochs: int, batch_size: int,
                   lr: float, patience: int, seed: int) -> Autoencoder:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    X_t = torch.tensor(X_normal, dtype=torch.float32)
    n   = len(X_t)
    n_val = int(n * 0.1)
    idx   = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    X_tr  = X_t[idx[n_val:]]; X_vl = X_t[idx[:n_val]]

    dl_tr = DataLoader(TensorDataset(X_tr), batch_size=batch_size, shuffle=True)
    model = Autoencoder(n_features, bottleneck=bottleneck).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit  = nn.MSELoss()

    best_val, best_state, no_imp = float('inf'), None, 0
    train_losses, val_losses = [], []

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for (xb,) in dl_tr:
            xb = xb.to(device)
            loss = crit(model(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(X_tr)

        model.eval()
        with torch.no_grad():
            vl_out  = model(X_vl.to(device))
            val_loss = crit(vl_out, X_vl.to(device)).item()
        sched.step(val_loss)
        train_losses.append(tr_loss); val_losses.append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            print(f"    Early stop epoch {ep+1}/{epochs}  val_loss={best_val:.6f}")
            break
        if (ep+1) % 10 == 0:
            print(f"    Epoch {ep+1:3d}/{epochs}  train={tr_loss:.6f}  val={val_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    return model, train_losses, val_losses, device


def get_ae_scores_torch(model, X: np.ndarray, device, batch_size=2048) -> np.ndarray:
    model.eval()
    X_t    = torch.tensor(X, dtype=torch.float32)
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb  = X_t[i:i+batch_size].to(device)
            err = model.reconstruction_error(xb).cpu().numpy()
            scores.append(err)
    return np.concatenate(scores)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    print("\n=== Autoencoder Stage 1 — Detection d'anomalies par reconstruction ===\n")

    # 1. Charger données
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
    n_features   = X_train_mm.shape[1]
    y_test_bin   = (y_test != normal_idx).astype(int)

    print(f"  Features : {n_features} | Normal train : {len(X_normal_tr):,}")

    # 2. Entraîner AE
    print(f"\n--- Entraînement Autoencoder (bottleneck={args.bottleneck}) ---")
    t0 = time.time()

    if HAS_TORCH:
        model, tr_losses, vl_losses, device = train_torch_ae(
            X_normal_tr.astype(np.float32), n_features,
            bottleneck=args.bottleneck,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            seed=SEEDS[0],
        )
        print(f"  Entraîné en {time.time()-t0:.1f}s")

        # Scores d'anomalie = erreur de reconstruction (MSE par flux)
        print("  Calcul des scores de reconstruction...")
        sc_val_n   = get_ae_scores_torch(model, X_val_normal.astype(np.float32), device)
        sc_val_a   = get_ae_scores_torch(model, X_val_attack.astype(np.float32), device)
        sc_val_all = get_ae_scores_torch(model, X_val_mm.astype(np.float32), device)
        sc_test    = get_ae_scores_torch(model, X_test_mm.astype(np.float32), device)
        sc_train   = get_ae_scores_torch(model, X_train_mm.astype(np.float32), device)

        # Sauvegarder modèle
        torch.save({"model_state": model.state_dict(),
                    "n_features":  n_features,
                    "bottleneck":  args.bottleneck,
                    "classes":     classes}, MODEL_OUT)
        print(f"  Modèle sauvegardé: {os.path.basename(MODEL_OUT)}")
    else:
        print("  [INFO] PyTorch non disponible — fallback PCA autoencoder")
        pca_ae = SimpleAENumpyFallback(n_components=args.bottleneck)
        pca_ae.fit(X_normal_tr)
        sc_val_n   = pca_ae.reconstruction_error(X_val_normal)
        sc_val_a   = pca_ae.reconstruction_error(X_val_attack)
        sc_val_all = pca_ae.reconstruction_error(X_val_mm)
        sc_test    = pca_ae.reconstruction_error(X_test_mm)
        sc_train   = pca_ae.reconstruction_error(X_train_mm)
        tr_losses, vl_losses = [], []

    # 3. Calibrer seuil
    print("\n--- Calibration du seuil ---")
    theta, calib = calibrate_theta(sc_val_n, sc_val_a)
    print(f"  theta={theta:.6f} | recall={calib['recall']:.3f} | fpr={calib['fpr']:.4f}")

    # 4. Évaluation sur test
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_val_bin  = (y_val != normal_idx).astype(int)

    auc_roc_val = float(roc_auc_score(y_val_bin, sc_val_all))
    auc_pr_val  = float(average_precision_score(y_val_bin, sc_val_all))
    flag_test   = sc_test >= theta
    y_test_flag = y_test[flag_test]

    tp = int(((y_test_bin==1)&flag_test).sum()); fn = int(((y_test_bin==1)&~flag_test).sum())
    fp = int(((y_test_bin==0)&flag_test).sum()); tn = int(((y_test_bin==0)&~flag_test).sum())
    recall_t = tp/(tp+fn+1e-12); fpr_t = fp/(fp+tn+1e-12)
    _, _, enr  = stealthy_enrichment(y_test, y_test_flag, stealthy_ids)

    auc_roc_test = float(roc_auc_score(y_test_bin, sc_test))
    print(f"\n  [TEST] recall={recall_t:.4f} | fpr={fpr_t:.4f}")
    print(f"  AUC-ROC={auc_roc_test:.4f} | AUC-PR val={auc_pr_val:.4f}")
    print(f"  Enrichissement furtif : x{enr:.3f}")

    # 5. Sauvegarder scores (pour l'ensemble)
    np.savez(SCORES_OUT,
             val_n=sc_val_n, val_a=sc_val_a,
             val_all=sc_val_all, test=sc_test, train=sc_train)
    np.save(os.path.join(METRICS_DIR, "s1_ae_score_train.npy"), sc_train)
    np.save(os.path.join(METRICS_DIR, "s1_ae_score_test.npy"),  sc_test)
    print(f"  Scores sauvegardés: {os.path.basename(SCORES_OUT)}")

    # 6. Métriques JSON
    metrics = {
        "model":        "Autoencoder",
        "backend":      "PyTorch" if HAS_TORCH else "PCA-fallback",
        "n_features":   n_features,
        "bottleneck":   args.bottleneck,
        "fit_time_s":   round(time.time()-t0, 1),
        "theta":        float(theta),
        "calib_val":    calib,
        "test": {
            "recall":      round(recall_t,       4),
            "fpr":         round(fpr_t,          4),
            "auc_roc":     round(auc_roc_test,   4),
            "auc_pr_val":  round(auc_pr_val,     4),
            "enrichment":  round(enr,            4),
            "n_flagged":   int(flag_test.sum()),
            "flagged_pct": round(flag_test.mean()*100, 2),
        },
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Métriques: {os.path.basename(METRICS_OUT)}")

    # 7. Figures
    fig, axes = plt.subplots(1, 2 if not tr_losses else 3,
                             figsize=(14 if tr_losses else 9, 4))

    # Distribution des scores (normal vs attaque)
    ax = axes[0]
    ax.hist(sc_val_n,   bins=80, alpha=0.6, color=PALETTE[0], label='Normal',  density=True)
    ax.hist(sc_val_a,   bins=80, alpha=0.6, color=PALETTE[1], label='Attaque', density=True)
    ax.axvline(theta, color='red', linestyle='--', linewidth=1.5, label=f'seuil={theta:.4f}')
    ax.set_xlabel("Erreur de reconstruction (MSE)")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution des scores AE\n(Normal vs Attaque, Validation)")
    ax.legend(fontsize=8); ax.set_xlim(left=0)

    # ROC curve
    from sklearn.metrics import roc_curve
    ax = axes[1]
    fpr_c, tpr_c, _ = roc_curve(y_val_bin, sc_val_all)
    ax.plot(fpr_c, tpr_c, color=PALETTE[2], linewidth=2,
            label=f"AE (AUC={auc_roc_val:.3f})")
    ax.plot([0,1],[0,1],'k--',alpha=0.4,linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("Recall (TPR)")
    ax.set_title("Courbe ROC — Autoencoder Stage 1")
    ax.legend(fontsize=9)

    # Courbe d'apprentissage (si PyTorch)
    if tr_losses and len(axes) > 2:
        ax = axes[2]
        ax.plot(tr_losses, color=PALETTE[0], label='Train', linewidth=1.5)
        ax.plot(vl_losses, color=PALETTE[1], label='Val',   linewidth=1.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.set_title("Courbe d'apprentissage Autoencoder")
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_ae_stage1.png")
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Figure: fig_ae_stage1.png")

    print(f"\n{'='*55}")
    print("RÉSUMÉ AUTOENCODER STAGE 1")
    print(f"{'='*55}")
    print(f"  AUC-ROC test    : {auc_roc_test:.4f}")
    print(f"  Recall          : {recall_t:.4f}")
    print(f"  FPR             : {fpr_t:.4f}")
    print(f"  Enrichissement  : x{enr:.4f}")
    print(f"  H2 (x5 cible)  : {'VALIDE' if enr>=5 else f'REJETE ({enr:.2f}x)'}")
    print(f"  Temps           : {time.time()-t0:.1f}s")
    print(f"{'='*55}")
    print("\n  Prochaine etape : ajouter AE a l'ensemble")
    print("  python train_ensemble_stage1.py --with-ae --vote majority")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Autoencoder Stage 1 — Anomaly Detection")
    p.add_argument("--bottleneck", type=int,   default=8,
                   help="Dimension du bottleneck (defaut: 8)")
    p.add_argument("--epochs",     type=int,   default=80,
                   help="Nombre d'epochs max (defaut: 80)")
    p.add_argument("--batch-size", type=int,   default=512,
                   help="Taille des batches (defaut: 512)")
    p.add_argument("--lr",         type=float, default=1e-3,
                   help="Learning rate (defaut: 0.001)")
    p.add_argument("--patience",   type=int,   default=10,
                   help="Early stopping patience (defaut: 10)")
    main(p.parse_args())
