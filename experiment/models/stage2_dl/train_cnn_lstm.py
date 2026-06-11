"""
═══════════════════════════════════════════════════════════════════
CNN-LSTM — Stage-2 Deep Learning Classifier
═══════════════════════════════════════════════════════════════════

THÉORIE
────────
Le CNN-LSTM combine deux architectures complémentaires :

1. CNN (Convolutional Neural Network) pour l'extraction de patterns
   locaux dans la fenêtre temporelle :
     Entrée : [batch, window_size, n_features]
     Conv1D : filtre de taille k parcourt la dimension temporelle
     → Détecte les patterns locaux (burst de paquets, séquences Modbus)

     Formule Conv1D :
     out[t] = ReLU( Σ_k w[k] · x[t-k] + b )

2. LSTM (Long Short-Term Memory) pour capturer les dépendances
   temporelles à long terme :
     Les LSTMs utilisent 3 "portes" pour contrôler le flux d'information :

     Porte d'oubli (forget gate) :
       fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)

     Porte d'entrée (input gate) :
       iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)
       C̃ₜ = tanh(WC · [hₜ₋₁, xₜ] + bC)

     Mise à jour de la cellule :
       Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ

     Porte de sortie (output gate) :
       oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
       hₜ = oₜ * tanh(Cₜ)

   σ = sigmoid, tanh = tangente hyperbolique

ARCHITECTURE CNN-LSTM
──────────────────────
  Input (batch, window, features)
       ↓
  Conv1D(64, kernel=3) → BatchNorm → ReLU → MaxPool
       ↓
  Conv1D(128, kernel=3) → BatchNorm → ReLU → MaxPool
       ↓
  LSTM(128, return_seq=True) → Dropout(0.3)
       ↓
  LSTM(64) → Dropout(0.3)
       ↓
  Dense(64, ReLU) → Dense(n_classes, Softmax)

POURQUOI EN ICS/IIoT
──────────────────────
Les attaques ICS comme Replay Attack et Modify Parameter ont des
SIGNATURES TEMPORELLES : une série d'échanges Modbus légitimes suivis
d'un pattern anormal. Le CNN capture les patterns locaux (burst de
commandes write), le LSTM capture la séquence temporelle globale.

FORCES
  ✓ Capture les patterns temporels des attaques furtives
  ✓ Adapté aux séries temporelles de trafic réseau
  ✓ Plus robuste que LSTM seul grâce à la hiérarchie de features

FAIBLESSES
  ✗ Nécessite un prétraitement en fenêtres temporelles (windowing)
  ✗ Requiert GPU pour un entraînement rapide
  ✗ Sensible à la taille de la fenêtre w
  ✗ Plus de données nécessaires que les modèles ML classiques

COMMANDE D'EXÉCUTION
  python train_cnn_lstm.py --window_size 30 --epochs 50
  Durée : ~10-30 min sans GPU, ~2-5 min avec GPU
  RAM : ~2-4 GB
  GPU : fortement recommandé (CUDA)

SORTIES
  results/models/cnn_lstm.pt
  results/metrics/cnn_lstm_metrics.json
  results/figures/cnn_lstm_cm.png
═══════════════════════════════════════════════════════════════════
"""

import sys, os, json, time, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    f1_score, accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ton_iot_config import *

os.makedirs(os.path.join(RESULTS_DIR, "saved_models"), exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMÈTRES — CNN-LSTM
# ──────────────────────────────────────────────────────────────────────────────
#
#  window_size (recommandé: 10-120 secondes équivalent)
#    → Nombre de flux consécutifs dans chaque fenêtre temporelle
#    → Plus grand = capture des patterns plus longs mais plus de mémoire
#    → IMPACT : capacité à détecter les attaques lentes (Replay Attack)
#    → CONSEIL : commencer avec window_size=30, tester 10, 60
#
#  cnn_filters (défaut: [64, 128])
#    → Nombre de filtres dans chaque couche Conv1D
#    → Plus de filtres = plus de patterns détectables
#    → IMPACT : expressivité vs coût mémoire
#
#  kernel_size (défaut: 3)
#    → Taille du filtre convolutional (fenêtre locale)
#    → kernel_size=3 → détecte des patterns sur 3 pas de temps
#    → Plus grand → patterns plus globaux
#    → IMPACT : granularité des patterns locaux détectés
#
#  lstm_units (défaut: 128, puis 64)
#    → Dimension de l'espace latent LSTM
#    → Plus grand = plus de capacité mais overfitting
#    → IMPACT : mémoire temporelle à long terme
#
#  dropout_rate (défaut: 0.3)
#    → Probabilité de désactiver aléatoirement un neurone pendant l'entraînement
#    → Régularisation pour éviter l'overfitting
#    → IMPACT : généralisation
#
#  learning_rate (défaut: 1e-3)
#    → Taux d'apprentissage d'Adam
#    → Trop grand → instabilité, trop petit → convergence lente
#    → CONSEIL : utiliser learning rate scheduler (ReduceLROnPlateau)
#
#  epochs (défaut: 50)
#    → Nombre de passes complètes sur les données
#    → Utiliser early stopping pour éviter l'overfitting
#
#  batch_size (défaut: 256)
#    → Nombre d'échantillons par mise à jour des poids
#    → Plus grand = plus stable mais plus de mémoire GPU
#    → IMPACT : stabilité de l'entraînement
#
# ──────────────────────────────────────────────────────────────────────────────

HYPERPARAMS = {
    "window_size":   30,       # 30 flux consécutifs par fenêtre
    "cnn_filters":   [64, 128],
    "kernel_size":   3,
    "lstm_units":    [128, 64],
    "dropout_rate":  0.3,
    "learning_rate": 1e-3,
    "epochs":        50,
    "batch_size":    256,
    "patience":      10,       # early stopping patience
}


# ── Architecture CNN-LSTM ─────────────────────────────────────────────────────
class CNNLSTM(nn.Module):
    def __init__(self, n_features, n_classes, hp):
        super().__init__()

        # Bloc CNN : extraction de patterns locaux
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, hp["cnn_filters"][0],
                      kernel_size=hp["kernel_size"], padding='same'),
            nn.BatchNorm1d(hp["cnn_filters"][0]),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(hp["cnn_filters"][0], hp["cnn_filters"][1],
                      kernel_size=hp["kernel_size"], padding='same'),
            nn.BatchNorm1d(hp["cnn_filters"][1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Taille de sortie après MaxPool x2 sur window_size
        conv_out_len = hp["window_size"] // 4  # 2 MaxPool de kernel=2

        # Bloc LSTM : dépendances temporelles à long terme
        self.lstm1 = nn.LSTM(hp["cnn_filters"][1], hp["lstm_units"][0],
                              batch_first=True)
        self.drop1 = nn.Dropout(hp["dropout_rate"])
        self.lstm2 = nn.LSTM(hp["lstm_units"][0], hp["lstm_units"][1],
                              batch_first=True)
        self.drop2 = nn.Dropout(hp["dropout_rate"])

        # Tête de classification
        self.fc = nn.Sequential(
            nn.Linear(hp["lstm_units"][1], 64),
            nn.ReLU(),
            nn.Dropout(hp["dropout_rate"]),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (batch, window_size, n_features)
        # Conv1d attend (batch, channels, length)
        x = x.permute(0, 2, 1)               # → (batch, features, window)
        x = self.conv_block(x)                # → (batch, filters, window//4)
        x = x.permute(0, 2, 1)               # → (batch, window//4, filters)

        # LSTM
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, (hn, _) = self.lstm2(x)
        x = self.drop2(hn[-1])               # dernier hidden state

        return self.fc(x)


def create_windows(X, y, window_size):
    """
    Transforme les données en fenêtres glissantes.

    Entrée : X (n_samples, n_features), y (n_samples,)
    Sortie : X_win (n_windows, window_size, n_features),
             y_win (n_windows,) — label de la dernière fenêtre
    """
    n_samples = len(X)
    n_windows = n_samples - window_size + 1
    if n_windows <= 0:
        raise ValueError(f"Dataset ({n_samples} samples) trop petit pour "
                         f"window_size={window_size}")

    X_win = np.stack([X[i:i+window_size] for i in range(n_windows)])
    y_win = y[window_size-1:]  # label du dernier pas de temps
    return X_win.astype(np.float32), y_win


def train(hp=None):
    if hp is None:
        hp = HYPERPARAMS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    print("=" * 60)
    print("CNN-LSTM — Stage-2 Deep Learning Classification")
    print("=" * 60)

    print("\n1. Chargement et fenêtrage des données...")
    X_tr = np.load(os.path.join(METRICS_DIR, "X_train_std.npy"))
    X_ts = np.load(os.path.join(METRICS_DIR, "X_test_std.npy"))
    y_tr = np.load(os.path.join(METRICS_DIR, "y_train.npy"))
    y_ts = np.load(os.path.join(METRICS_DIR, "y_test.npy"))

    with open(os.path.join(METRICS_DIR, "label_classes.json")) as f:
        class_names = json.load(f)["classes"]
    n_cls = len(class_names)

    # Création des fenêtres
    X_train_win, y_train_win = create_windows(X_tr, y_tr, hp["window_size"])
    X_test_win,  y_test_win  = create_windows(X_ts, y_ts, hp["window_size"])

    print(f"   Fenêtres train : {X_train_win.shape}")
    print(f"   Fenêtres test  : {X_test_win.shape}")

    # DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train_win),
        torch.from_numpy(y_train_win).long()
    )
    test_ds  = TensorDataset(
        torch.from_numpy(X_test_win),
        torch.from_numpy(y_test_win).long()
    )
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"],
                               shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=hp["batch_size"],
                               shuffle=False, num_workers=0)

    # Poids de classe pour Focal Loss ou CrossEntropy pondérée
    # Compenser le déséquilibre de classes
    unique, counts = np.unique(y_train_win, return_counts=True)
    class_weights = 1.0 / (counts + 1)
    class_weights = class_weights / class_weights.sum() * n_cls
    weights_tensor = torch.zeros(n_cls)
    for i, uid in enumerate(unique):
        weights_tensor[uid] = class_weights[i]
    weights_tensor = weights_tensor.to(device)

    # ── Modèle ────────────────────────────────────────────────────────────────
    print("\n2. Initialisation du modèle...")
    n_features = X_tr.shape[1]
    model      = CNNLSTM(n_features, n_cls, hp).to(device)
    n_params   = sum(p.numel() for p in model.parameters())
    print(f"   Paramètres : {n_params:,}")

    optimizer  = optim.Adam(model.parameters(), lr=hp["learning_rate"])
    criterion  = nn.CrossEntropyLoss(weight=weights_tensor)

    # Learning rate scheduler : réduit le LR si la perte stagne
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # ── Entraînement ─────────────────────────────────────────────────────────
    print("\n3. Entraînement...")
    best_f1   = 0.0
    patience  = 0
    train_losses, val_f1s = [], []

    t_start = time.time()
    for epoch in range(hp["epochs"]):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        avg_loss = total_loss / len(train_ds)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        # Évaluation rapide sur test (pour early stopping)
        if (epoch + 1) % 5 == 0:
            model.eval()
            all_pred = []
            with torch.no_grad():
                for X_b, _ in test_loader:
                    out    = model(X_b.to(device))
                    preds  = out.argmax(dim=1).cpu().numpy()
                    all_pred.extend(preds)
            f1 = f1_score(y_test_win, all_pred, average='macro',
                          zero_division=0)
            val_f1s.append(f1)
            print(f"   Epoch {epoch+1:3d}/{hp['epochs']} | "
                  f"Loss={avg_loss:.4f} | Val F1-mac={f1:.4f}",
                  flush=True)

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in
                               model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= hp["patience"] // 5:
                    print(f"   Early stopping à epoch {epoch+1}")
                    break

    t_fit = time.time() - t_start
    print(f"   Durée totale : {t_fit:.0f}s")

    # Charger le meilleur modèle
    model.load_state_dict(best_state)

    # ── Évaluation finale ─────────────────────────────────────────────────────
    print("\n4. Évaluation finale...")
    model.eval()
    all_pred, all_prob = [], []
    with torch.no_grad():
        for X_b, _ in test_loader:
            out   = model(X_b.to(device))
            prob  = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            all_pred.extend(preds)
            all_prob.extend(prob)

    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)

    f1m    = f1_score(y_test_win, y_pred, average='macro',    zero_division=0)
    f1w    = f1_score(y_test_win, y_pred, average='weighted', zero_division=0)
    acc    = accuracy_score(y_test_win, y_pred)
    mcc    = matthews_corrcoef(y_test_win, y_pred)

    y_bin  = np.eye(n_cls)[y_test_win]
    try:
        auc_roc = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
    except Exception:
        auc_roc = float('nan')
    auc_pr  = average_precision_score(y_bin, y_prob, average='macro')

    print(f"\n   F1-macro  = {f1m:.4f}")
    print(f"   AUC-PR    = {auc_pr:.4f}")
    print(f"   MCC       = {mcc:.4f}")
    print(classification_report(y_test_win, y_pred,
                                  target_names=class_names,
                                  zero_division=0))

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    model_path = os.path.join(RESULTS_DIR, "saved_models", "cnn_lstm.pt")
    torch.save({"model_state": model.state_dict(),
                 "class_names": class_names,
                 "hyperparams":  hp,
                 "n_features":   n_features,
                 "n_classes":    n_cls}, model_path)

    metrics = {
        "model": "CNN-LSTM",
        "f1_macro": round(f1m, 4),
        "f1_weighted": round(f1w, 4),
        "accuracy": round(acc, 4),
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "mcc": round(mcc, 4),
        "hyperparams": hp,
        "fit_time_s": round(t_fit, 0),
        "device": str(device),
        "n_params": n_params,
    }
    with open(os.path.join(METRICS_DIR, "cnn_lstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Courbe d'apprentissage
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label='Train Loss', color='#e74c3c')
    ax.set(xlabel="Epoch", ylabel="Cross-Entropy Loss",
           title="CNN-LSTM — Courbe d'apprentissage")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "cnn_lstm_loss.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nCNN-LSTM sauvegardé : {model_path}")
    return metrics


# ══════════════════════════════════════════════════════════════════
# COMMENT DÉBOGUER UN CNN-LSTM
# ══════════════════════════════════════════════════════════════════
#
# Problème : La loss ne diminue pas
#   → Réduire learning_rate (1e-4 au lieu de 1e-3)
#   → Vérifier que les données sont bien normalisées (StandardScaler)
#   → Vérifier que les labels sont des entiers 0..n_classes-1
#
# Problème : Overfitting (loss train ↓ mais val F1 ↓)
#   → Augmenter dropout_rate (0.5)
#   → Réduire les unités LSTM (64, 32)
#   → Augmenter weight_decay dans Adam
#
# Problème : Out of Memory GPU
#   → Réduire batch_size (128, 64)
#   → Réduire window_size (10)
#   → Réduire les filtres CNN (32, 64)
#
# Problème : Recall=0 sur mitm/ransomware
#   → Ces classes sont très rares (0.1%) dans les fenêtres
#   → Utiliser class_weights encore plus aggressifs
#   → Envisager Focal Loss au lieu de CrossEntropy standard
# ══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    hp = {**HYPERPARAMS, "window_size": args.window_size,
          "epochs": args.epochs, "batch_size": args.batch_size}
    train(hp)
