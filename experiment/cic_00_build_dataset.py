"""
Construit les .npy pour CIC-IDS2018.
Utilisation :
  python experiment/cic_00_build_dataset.py
  python experiment/cic_00_build_dataset.py --max-rows 200000  # test rapide
  python experiment/cic_00_build_dataset.py --cache-only       # n'utilise que le CSV déjà préparé

Produit (dans results_cic/metrics/) :
  X_train.npy, X_val.npy, X_test.npy
  y_train.npy, y_val.npy, y_test.npy
  X_train_normal.npy   (pour Stage 1 unsupervised)
  scaler.pkl           (StandardScaler ajusté sur train)
  label_encoder.pkl
  feature_names.json
"""
import os, glob, json, argparse, warnings, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── importer la config CIC ──────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cic_ids2018_config import (
    CIC_DIR, METRICS_DIR, FIGURES_DIR, LATEX_DIR,
    CACHE_CSV, LABEL_COL, NUMERIC_FEATURES,
    ALL_CLASSES, NORMAL_CLASS, STEALTHY_CLASSES, SAMPLE_TARGET,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LATEX_DIR,   exist_ok=True)


# ── 1. Chargement des CSV ────────────────────────────────────────────────────

def _normalize_label(s: str) -> str:
    """Normalise les variantes de noms dans CIC-IDS2018."""
    mapping = {
        "benign":                  "Benign",
        "dos attacks-goldeneye":   "DoS-GoldenEye",
        "dos attacks-hulk":        "DoS-Hulk",
        "dos attacks-slowhttptest":"DoS-Slowhttptest",
        "dos attacks-slowloris":   "DoS-Slowloris",
        "ddos attacks-loic-http":  "DDoS-LOIC-HTTP",
        "ddos attack-loic-http":   "DDoS-LOIC-HTTP",
        "botnet":                  "BotNet",
        "bot":                     "BotNet",
        "ftp-bruteforce":          "FTP-BruteForce",
        "ssh-bruteforce":          "SSH-BruteForce",
        "infilteration":           "Infiltration",
        "infiltration":            "Infiltration",
        "brute force -web":        "Web-BruteForce",
        "brute force -xss":        "Web-XSS",
        "sql injection":           "Web-Sql-Injection",
        "ddos attack-hoic":        "DDOS-HOIC",
        "heartbleed":              "Heartbleed",
    }
    return mapping.get(s.strip().lower(), s.strip())


def load_raw(max_rows: int = None) -> pd.DataFrame:
    """Lit tous les CSV du dossier CIC_DIR, filtre et normalise les labels."""
    if not os.path.isdir(CIC_DIR):
        raise FileNotFoundError(
            f"Dossier introuvable : {CIC_DIR}\n"
            "Télécharge CIC-IDS2018 depuis https://www.unb.ca/cic/datasets/ids-2018.html\n"
            "et adapte CIC_DIR dans cic_ids2018_config.py"
        )

    files = sorted(glob.glob(os.path.join(CIC_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {CIC_DIR}")

    print(f"[CIC] {len(files)} fichier(s) détecté(s)")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="latin-1", low_memory=False)
            # Normaliser les noms de colonnes (espaces en début/fin)
            df.columns = df.columns.str.strip()
            if LABEL_COL not in df.columns:
                # Essayer "label" en minuscules
                alt = [c for c in df.columns if c.lower() == "label"]
                if alt:
                    df.rename(columns={alt[0]: LABEL_COL}, inplace=True)
                else:
                    print(f"  [SKIP] {os.path.basename(f)} — colonne '{LABEL_COL}' absente")
                    continue
            df[LABEL_COL] = df[LABEL_COL].astype(str).apply(_normalize_label)
            frames.append(df)
            print(f"  [OK] {os.path.basename(f)} — {len(df):,} lignes")
        except Exception as e:
            print(f"  [WARN] {os.path.basename(f)} : {e}")

    if not frames:
        raise ValueError("Aucun fichier CSV valide chargé.")

    df_all = pd.concat(frames, ignore_index=True)
    if max_rows:
        df_all = df_all.sample(min(max_rows, len(df_all)), random_state=42)
    print(f"[CIC] Total : {len(df_all):,} flux | classes : {df_all[LABEL_COL].unique()}")
    return df_all


# ── 2. Nettoyage et feature engineering ─────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Garder uniquement les colonnes numériques pertinentes + label
    available = [c for c in NUMERIC_FEATURES if c in df.columns]
    missing   = [c for c in NUMERIC_FEATURES if c not in df.columns]
    if missing:
        print(f"[WARN] Features absentes ({len(missing)}) : {missing[:5]} ...")

    df = df[available + [LABEL_COL]].copy()

    # Remplacer inf et NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Filtrer les labels inconnus
    df = df[df[LABEL_COL].isin(ALL_CLASSES)]

    # Feature engineering simple (ratios si disponibles)
    if "Flow Bytes/s" in df.columns and "Flow Packets/s" in df.columns:
        df["bytes_per_pkt"] = (
            df["Flow Bytes/s"] / (df["Flow Packets/s"] + 1e-9)
        ).clip(0, 1e7)

    print(f"[CIC] Après nettoyage : {len(df):,} flux valides")
    print(df[LABEL_COL].value_counts().to_string())
    return df


# ── 3. Sous-échantillonnage stratifié ───────────────────────────────────────

def stratified_sample(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for cls, target in SAMPLE_TARGET.items():
        sub = df[df[LABEL_COL] == cls]
        if len(sub) == 0:
            print(f"  [WARN] classe absente : {cls}")
            continue
        n = min(target, len(sub))
        frames.append(sub.sample(n, random_state=42))
        print(f"  {cls:30s}: {len(sub):>8,} → {n:>8,}")
    return pd.concat(frames, ignore_index=True)


# ── 4. Split train/val/test ──────────────────────────────────────────────────

def split(df: pd.DataFrame):
    X = df.drop(columns=[LABEL_COL]).values
    y = df[LABEL_COL].values

    le = LabelEncoder()
    le.fit(ALL_CLASSES)
    y_enc = le.transform(y)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y_enc, test_size=TEST_RATIO, random_state=42, stratify=y_enc
    )
    val_frac = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=42, stratify=y_tv
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    normal_idx = le.transform([NORMAL_CLASS])[0]
    X_train_normal = X_train[y_train == normal_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, le, X_train_normal, df.drop(columns=[LABEL_COL]).columns.tolist()


# ── 5. Sauvegarde ────────────────────────────────────────────────────────────

def save(X_train, X_val, X_test, y_train, y_val, y_test,
         scaler, le, X_train_normal, feature_names):
    np.save(os.path.join(METRICS_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(METRICS_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(METRICS_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(METRICS_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(METRICS_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(METRICS_DIR, "y_test.npy"),  y_test)
    np.save(os.path.join(METRICS_DIR, "X_train_normal.npy"), X_train_normal)

    with open(os.path.join(METRICS_DIR, "scaler.pkl"),        "wb") as f: pickle.dump(scaler, f)
    with open(os.path.join(METRICS_DIR, "label_encoder.pkl"), "wb") as f: pickle.dump(le, f)
    with open(os.path.join(METRICS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"\n[CIC] Sauvegardé dans {METRICS_DIR}")
    print(f"  Train : {X_train.shape} | Val : {X_val.shape} | Test : {X_test.shape}")
    print(f"  Normal train : {X_train_normal.shape}")
    print(f"  Classes : {list(le.classes_)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build CIC-IDS2018 .npy dataset")
    ap.add_argument("--max-rows",   type=int, default=None,
                    help="Limite le nb de lignes chargées (test rapide)")
    ap.add_argument("--cache-only", action="store_true",
                    help="Repart du CSV préparé (CACHE_CSV) au lieu des fichiers bruts")
    args = ap.parse_args()

    if args.cache_only and os.path.exists(CACHE_CSV):
        print(f"[CIC] Chargement depuis le cache : {CACHE_CSV}")
        df = pd.read_csv(CACHE_CSV)
    else:
        df_raw = load_raw(max_rows=args.max_rows)
        df     = clean(df_raw)
        os.makedirs(os.path.dirname(CACHE_CSV), exist_ok=True)
        df.to_csv(CACHE_CSV, index=False)
        print(f"[CIC] Cache CSV sauvegardé : {CACHE_CSV}")

    df_s  = stratified_sample(df)
    out   = split(df_s)
    save(*out)
    print("\n[CIC] Done. Lance maintenant les scripts ton_01+ avec DATASET=cic dans les configs.")


if __name__ == "__main__":
    main()
