"""
═══════════════════════════════════════════════════════════════════
TRANSFORMER (PatchTST variant) — Stage-2 Deep Learning Classifier
═══════════════════════════════════════════════════════════════════

THÉORIE — Transformer Standard
────────────────────────────────
Le Transformer utilise le mécanisme d'ATTENTION MULTI-TÊTE pour
pondérer l'importance de chaque pas de temps par rapport à tous les
autres — sans récurrence, contrairement au LSTM.

Mécanisme d'attention :
  Q = XW_Q   (Query  : "quelle information je cherche ?")
  K = XW_K   (Key    : "quelle information je contiens ?")
  V = XW_V   (Value  : "quelle information je fournis ?")

  Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

  d_k = dimension des vecteurs Query/Key
  √d_k = facteur d'échelle (évite les gradients trop petits)

Multi-Head Attention :
  MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · W_O
  headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)

  → Chaque "tête" apprend à s'intéresser à différentes relations
  → h=4 têtes : une pour les patterns locaux, une pour les patterns
    globaux, etc.

Encodage positionnel :
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  → Injecte l'information de position temporelle (absent par défaut
    car l'attention est permutation-invariante)

THÉORIE — PatchTST
───────────────────
PatchTST découpe la série temporelle en "patches" (segments) avant
de les passer au Transformer — analogue aux patches dans Vision
Transformer (ViT) pour les images.

Avantages des patches :
  1. Réduit la complexité de O(L²) à O((L/p)²) où p = patch_size
  2. Capture des patterns à plus grande échelle
  3. Plus efficace sur les séries temporelles courtes

Architecture PatchTST :
  Input (batch, window, features)
       ↓
  Découpage en patches de taille p avec stride s
  → n_patches = (window - patch_size) / stride + 1
       ↓
  Projection linéaire : chaque patch → vecteur de d_model dimensions
       ↓
  + Encodage positionnel
       ↓
  Transformer Encoder (N couches)
       ↓
  Flatten + Dense → classification

POURQUOI EN ICS/IIoT
──────────────────────
Les Transformers excellent pour les dépendances à long terme dans
les séries temporelles — utile pour les attaques Replay qui ont des
patterns répétés sur plusieurs minutes.

FORCES
  ✓ Capture des dépendances à très longue portée
  ✓ Parallélisable (contrairement au LSTM séquentiel)
  ✓ PatchTST : efficace sur séries temporelles courtes
  ✓ Interprétable via les poids d'attention

FAIBLESSES
  ✗ Requiert beaucoup de données pour converger
  ✗ Coûteux en mémoire (O(n²) pour l'attention standard)
  ✗ Plus difficile à entraîner que LSTM
  ✗ Risque d'overfitting avec peu de données ICS

COMMANDE D'EXÉCUTION
  python train_transformer.py --model patchtst --window_size 60
  Durée : ~20-60 min sans GPU, ~5-10 min avec GPU
  RAM : ~4-8 GB
  GPU : fortement recommandé

SORTIES
  results/models/transformer.pt / patchtst.pt
  results/metrics/transformer_metrics.json
═══════════════════════════════════════════════════════════════════
"""

import sys, os, json, time, math, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (f1_score, accuracy_score, matthews_corrcoef,
                              roc_auc_score, average_precision_score,
                              classification_report)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ton_iot_config import *

os.makedirs(os.path.join(RESULTS_DIR, "saved_models"), exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMÈTRES — Transformer / PatchTST
# ──────────────────────────────────────────────────────────────────────────────
#
#  window_size (défaut=60)
#    → Longueur de la séquence temporelle en entrée
#    → Plus grand = plus de contexte mais O(window²) pour l'attention
#    → Pour ICS : 30-120 secondes équivalent
#
#  d_model (défaut=64)
#    → Dimension des embeddings et des couches internes
#    → Doit être divisible par n_heads
#    → Plus grand = plus de capacité mais plus lent
#    → IMPACT : expressivité du modèle
#
#  n_heads (défaut=4)
#    → Nombre de "têtes" dans la Multi-Head Attention
#    → Chaque tête apprend des patterns différents
#    → Contrainte : d_model % n_heads == 0
#    → IMPACT : diversité des patterns appris
#
#  n_layers (défaut=2)
#    → Nombre de couches Transformer empilées
#    → 2 couches suffisent pour des séries temporelles courtes
#    → Plus de couches → plus de capacité mais risque d'overfitting
#    → Pour des petits datasets ICS, GARDER n_layers ≤ 3
#    → IMPACT : profondeur de la hiérarchie de représentations
#
#  d_ff (défaut=256)
#    → Dimension de la couche Feed-Forward dans chaque bloc Transformer
#    → Généralement 4 × d_model
#    → IMPACT : capacité de transformation non-linéaire
#
#  dropout (défaut=0.2)
#    → Régularisation dans l'attention et les couches FC
#    → Crucial pour éviter l'overfitting sur ICS
#
#  patch_size (PatchTST uniquement, défaut=8)
#    → Taille de chaque patch temporel
#    → window_size doit être divisible par patch_size (avec stride)
#    → Plus grand = moins de patches = plus rapide
#    → IMPACT : granularité temporelle
#
#  stride (PatchTST uniquement, défaut=4)
#    → Pas entre deux patches consécutifs
#    → stride < patch_size → overlap entre patches
#    → IMPACT : couverture temporelle
#
# ──────────────────────────────────────────────────────────────────────────────

HYPERPARAMS_VANILLA = {
    "model_type":  "vanilla",
    "window_size": 60,
    "d_model":     64,
    "n_heads":     4,
    "n_layers":    2,
    "d_ff":        256,
    "dropout":     0.2,
    "learning_rate": 1e-3,
    "epochs":      50,
    "batch_size":  128,
    "patience":    10,
}

HYPERPARAMS_PATCH = {
    "model_type":  "patchtst",
    "window_size": 60,
    "patch_size":  8,
    "stride":      4,
    "d_model":     64,
    "n_heads":     4,
    "n_layers":    2,
    "d_ff":        256,
    "dropout":     0.2,
    "learning_rate": 1e-3,
    "epochs":      50,
    "batch_size":  128,
    "patience":    10,
}


class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoïdal."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class VanillaTransformer(nn.Module):
    """Transformer standard pour classification de séries temporelles."""
    def __init__(self, n_features, n_classes, hp):
        super().__init__()
        self.input_proj  = nn.Linear(n_features, hp["d_model"])
        self.pos_enc     = PositionalEncoding(hp["d_model"],
                                               hp["window_size"],
                                               hp["dropout"])
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=hp["d_model"], nhead=hp["n_heads"],
            dim_feedforward=hp["d_ff"], dropout=hp["dropout"],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                   num_layers=hp["n_layers"])
        self.fc = nn.Sequential(
            nn.Linear(hp["d_model"], 64), nn.ReLU(),
            nn.Dropout(hp["dropout"]),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (batch, window, features)
        x = self.input_proj(x)   # → (batch, window, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)  # → (batch, window, d_model)
        x = x.mean(dim=1)        # Global Average Pooling sur la dimension temporelle
        return self.fc(x)


class PatchTST(nn.Module):
    """PatchTST : Transformer avec découpage en patches temporels."""
    def __init__(self, n_features, n_classes, hp):
        super().__init__()
        # Calculer le nombre de patches
        self.patch_size = hp["patch_size"]
        self.stride     = hp["stride"]
        n_patches = (hp["window_size"] - hp["patch_size"]) // hp["stride"] + 1

        # Projection linéaire de chaque patch
        self.patch_proj  = nn.Linear(hp["patch_size"] * n_features, hp["d_model"])
        self.pos_enc     = PositionalEncoding(hp["d_model"], n_patches, hp["dropout"])

        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=hp["d_model"], nhead=hp["n_heads"],
            dim_feedforward=hp["d_ff"], dropout=hp["dropout"],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                   num_layers=hp["n_layers"])
        self.fc = nn.Sequential(
            nn.Linear(hp["d_model"], 64), nn.ReLU(),
            nn.Dropout(hp["dropout"]),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (batch, window, features)
        batch, window, feats = x.shape
        # Extraire les patches
        patches = x.unfold(1, self.patch_size, self.stride)
        # patches: (batch, n_patches, features, patch_size)
        patches = patches.contiguous().view(batch, -1,
                                             self.patch_size * feats)
        x = self.patch_proj(patches)   # → (batch, n_patches, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


def create_windows_ts(X, y, window_size):
    n = len(X) - window_size + 1
    Xw = np.stack([X[i:i+window_size] for i in range(n)])
    yw = y[window_size-1:]
    return Xw.astype(np.float32), yw


def train(hp=None):
    if hp is None:
        hp = HYPERPARAMS_PATCH
    model_type = hp.get("model_type", "patchtst")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modèle : {model_type} | Device : {device}")

    X_tr = np.load(os.path.join(METRICS_DIR, "X_train_std.npy"))
    X_ts = np.load(os.path.join(METRICS_DIR, "X_test_std.npy"))
    y_tr = np.load(os.path.join(METRICS_DIR, "y_train.npy"))
    y_ts = np.load(os.path.join(METRICS_DIR, "y_test.npy"))
    with open(os.path.join(METRICS_DIR, "label_classes.json")) as f:
        class_names = json.load(f)["classes"]
    n_cls = len(class_names)

    Xw_tr, yw_tr = create_windows_ts(X_tr, y_tr, hp["window_size"])
    Xw_ts, yw_ts = create_windows_ts(X_ts, y_ts, hp["window_size"])

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xw_tr),
                       torch.from_numpy(yw_tr).long()),
        batch_size=hp["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xw_ts),
                       torch.from_numpy(yw_ts).long()),
        batch_size=hp["batch_size"], shuffle=False
    )

    n_features = X_tr.shape[1]
    model = (PatchTST(n_features, n_cls, hp)
             if model_type == "patchtst"
             else VanillaTransformer(n_features, n_cls, hp)).to(device)

    # Poids de classes
    unique, counts = np.unique(yw_tr, return_counts=True)
    w = 1.0 / (counts + 1)
    w = w / w.sum() * n_cls
    wt = torch.zeros(n_cls)
    for i, uid in enumerate(unique):
        wt[uid] = w[i]
    criterion = nn.CrossEntropyLoss(weight=wt.to(device))
    optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"],
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=hp["epochs"])

    best_f1, patience_cnt, best_state = 0, 0, None
    t0 = time.time()

    for epoch in range(hp["epochs"]):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            out  = model(Xb.to(device))
            loss = criterion(out, yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            preds = []
            with torch.no_grad():
                for Xb, _ in test_loader:
                    preds.extend(model(Xb.to(device)).argmax(1).cpu().numpy())
            f1 = f1_score(yw_ts, preds, average='macro', zero_division=0)
            print(f"  Epoch {epoch+1:3d} | F1-macro={f1:.4f}", flush=True)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in
                               model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 2:
                    print(f"  Early stopping epoch {epoch+1}")
                    break

    if best_state:
        model.load_state_dict(best_state)
    t_fit = time.time() - t0

    # Évaluation finale
    model.eval()
    all_pred, all_prob = [], []
    with torch.no_grad():
        for Xb, _ in test_loader:
            out  = model(Xb.to(device))
            all_prob.extend(torch.softmax(out, 1).cpu().numpy())
            all_pred.extend(out.argmax(1).cpu().numpy())
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)

    f1m  = f1_score(yw_ts, y_pred, average='macro', zero_division=0)
    mcc  = matthews_corrcoef(yw_ts, y_pred)
    y_bin = np.eye(n_cls)[yw_ts]
    try:
        auc_pr = average_precision_score(y_bin, y_prob, average='macro')
    except Exception:
        auc_pr = float('nan')

    print(f"\n{model_type.upper()} | F1-macro={f1m:.4f} | AUC-PR={auc_pr:.4f} "
          f"| MCC={mcc:.4f} | fit={t_fit:.0f}s", flush=True)

    torch.save({"model_state": model.state_dict(),
                 "class_names": class_names,
                 "hyperparams": hp,
                 "n_features": n_features,
                 "n_classes": n_cls},
               os.path.join(RESULTS_DIR, "saved_models",
                             f"{model_type}.pt"))

    metrics = {"model": model_type, "f1_macro": round(f1m, 4),
               "auc_pr": round(auc_pr, 4), "mcc": round(mcc, 4),
               "fit_time_s": round(t_fit, 0), "hyperparams": hp}
    with open(os.path.join(METRICS_DIR, f"{model_type}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vanilla", "patchtst"],
                        default="patchtst")
    parser.add_argument("--window_size", type=int, default=60)
    args = parser.parse_args()
    hp = ({**HYPERPARAMS_PATCH} if args.model == "patchtst"
          else {**HYPERPARAMS_VANILLA})
    hp["window_size"] = args.window_size
    train(hp)
