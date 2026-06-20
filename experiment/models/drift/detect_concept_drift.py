"""
Concept Drift Detection for ICS/IIoT Traffic (Cynergia Framework)
Détecte quand la distribution des flux réseau dérive significativement
par rapport à la baseline d'entraînement, signalant un changement de
l'environnement ICS (nouveau équipement, nouveau protocole, reconfiguration).

Méthode : Maximum Mean Discrepancy (MMD) avec kernel RBF sur fenêtre glissante.
  - MMD ≈ 0 : distribution courante ≈ baseline
  - MMD > seuil : dérive détectée → re-calibration ou re-entraînement recommandé

Références :
  - Gretton et al. (2012) "A Kernel Two-Sample Test", JMLR 13
  - Lu et al. (2019) "Learning under Concept Drift: A Review", TKDE

Usage:
  python detect_concept_drift.py --calibrate     # Calibrer le seuil sur données de référence
  python detect_concept_drift.py --stream flows.jsonl  # Surveillance temps réel
  python detect_concept_drift.py --simulate      # Injection de dérive synthétique pour test
"""
import os, sys, json, argparse, time, warnings
from datetime import datetime, timezone
from collections import deque
from typing import Optional, List, Tuple

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ton_iot_config import (METRICS_DIR, FIGURES_DIR, NUMERIC_FEATURES,
                              SEEDS, PALETTE)

DRIFT_CONFIG_PATH = os.path.join(METRICS_DIR, "drift_config.json")
DRIFT_LOG_PATH    = os.path.join(METRICS_DIR, "drift_events.jsonl")

# ── MMD ───────────────────────────────────────────────────────────────────────

def rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Compute RBF kernel matrix K(X, Y) with bandwidth σ."""
    # ||x - y||² via expansion: ||x||² + ||y||² - 2x·y
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    dist2 = XX + YY.T - 2.0 * (X @ Y.T)
    return np.exp(-dist2 / (2.0 * sigma ** 2))


def mmd_squared(X: np.ndarray, Y: np.ndarray,
                sigma: Optional[float] = None) -> float:
    """
    Biased MMD² estimator between samples X and Y.
    MMD²(X,Y) = E[k(x,x')] - 2·E[k(x,y)] + E[k(y,y')]
    """
    if sigma is None:
        # Median heuristic for bandwidth
        all_data = np.vstack([X, Y])
        dists = np.sum((all_data[:, None] - all_data[None, :]) ** 2, axis=-1)
        sigma = float(np.sqrt(np.median(dists[dists > 0]) / 2.0))
        sigma = max(sigma, 1e-6)

    Kxx = rbf_kernel_matrix(X, X, sigma)
    Kyy = rbf_kernel_matrix(Y, Y, sigma)
    Kxy = rbf_kernel_matrix(X, Y, sigma)

    n, m = len(X), len(Y)
    # Unbiased estimators: exclude diagonal
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = (Kxx.sum() / (n * (n-1)) +
            Kyy.sum() / (m * (m-1)) -
            2.0 * Kxy.mean())
    return float(mmd2)


def bootstrap_mmd_threshold(X_ref: np.ndarray, n_bootstrap: int = 200,
                             window_size: int = 500, alpha: float = 0.01,
                             seed: int = SEEDS[0]) -> Tuple[float, float, float]:
    """
    Permutation bootstrap to estimate the MMD threshold under H₀ (no drift).
    Returns (threshold_α, mean_H0, std_H0).
    """
    rng = np.random.RandomState(seed)
    n   = len(X_ref)
    # Estimate sigma once on the reference
    dists = np.sum((X_ref[:1000, None] - X_ref[None, :1000]) ** 2, axis=-1)
    sigma = float(np.sqrt(np.median(dists[dists > 0]) / 2.0))
    sigma = max(sigma, 1e-6)

    null_mmds = []
    for _ in range(n_bootstrap):
        idx1 = rng.choice(n, size=min(window_size, n//2), replace=False)
        idx2 = rng.choice(n, size=min(window_size, n//2), replace=False)
        X1 = X_ref[idx1];  X2 = X_ref[idx2]
        null_mmds.append(mmd_squared(X1, X2, sigma=sigma))

    null_mmds = np.array(null_mmds)
    threshold = float(np.percentile(null_mmds, (1 - alpha) * 100))
    return threshold, float(null_mmds.mean()), float(null_mmds.std())


# ── Drift detector ────────────────────────────────────────────────────────────

class ConceptDriftDetector:
    """
    Sliding window MMD-based concept drift detector.

    Architecture :
      - Reference window Xref : normal traffic from training set
      - Current window Xcurr  : last `window_size` observed flows
      - Every `check_every` flows : compute MMD(Xref_sample, Xcurr)
      - Alert if MMD² > threshold
    """

    def __init__(self, X_reference: np.ndarray, threshold: float,
                 sigma: float, window_size: int = 500,
                 ref_sample_size: int = 500, check_every: int = 100,
                 verbose: bool = True):
        self.threshold      = threshold
        self.sigma          = sigma
        self.window_size    = window_size
        self.ref_sample_size= ref_sample_size
        self.check_every    = check_every
        self.verbose        = verbose

        # Fixed reference sample (from training normal traffic)
        rng = np.random.RandomState(SEEDS[0])
        idx = rng.choice(len(X_reference),
                         size=min(ref_sample_size, len(X_reference)), replace=False)
        self._X_ref = X_reference[idx].astype(np.float32)

        self._buffer = deque(maxlen=window_size)
        self._n_seen = 0
        self._drift_events: List[dict] = []
        self._mmd_history: List[Tuple[int, float]] = []

    def update(self, x: np.ndarray) -> Optional[dict]:
        """Feed one flow vector; return drift event dict if drift detected, else None."""
        self._buffer.append(x.astype(np.float32))
        self._n_seen += 1

        if (self._n_seen % self.check_every != 0 or
                len(self._buffer) < self.window_size // 2):
            return None

        X_curr = np.array(self._buffer)
        mmd2   = mmd_squared(self._X_ref, X_curr, sigma=self.sigma)
        self._mmd_history.append((self._n_seen, mmd2))

        if mmd2 > self.threshold:
            evt = {
                "type":        "CONCEPT_DRIFT",
                "severity":    "HIGH" if mmd2 > 2 * self.threshold else "MEDIUM",
                "detected_at": datetime.now(tz=timezone.utc).isoformat(),
                "n_flows_seen": self._n_seen,
                "mmd_squared": round(mmd2,   6),
                "threshold":   round(self.threshold, 6),
                "ratio":       round(mmd2 / (self.threshold + 1e-12), 3),
                "recommendation": (
                    "Re-calibration immédiate du modèle Stage 1 recommandée. "
                    "Nouveau équipement ICS ou changement de protocole probable."
                ),
            }
            self._drift_events.append(evt)
            if self.verbose:
                print(f"\n  [DRIFT ALERT] MMD²={mmd2:.6f} > seuil={self.threshold:.6f} "
                      f"(ratio×{evt['ratio']:.2f}) — après {self._n_seen} flux")
            return evt
        return None

    @property
    def drift_events(self):
        return self._drift_events

    @property
    def mmd_history(self):
        return self._mmd_history

    def status(self) -> dict:
        recent_mmd = self._mmd_history[-1][1] if self._mmd_history else 0.0
        return {
            "n_flows_seen":   self._n_seen,
            "buffer_size":    len(self._buffer),
            "n_drift_events": len(self._drift_events),
            "last_mmd":       round(recent_mmd, 6),
            "threshold":      round(self.threshold, 6),
            "drift_detected": recent_mmd > self.threshold,
        }


# ── Calibration ────────────────────────────────────────────────────────────────

def calibrate(args):
    """Estimate MMD threshold from reference (training normal) data."""
    print("\n=== Calibration du seuil MMD (dérive conceptuelle) ===\n")

    X_ref_path = os.path.join(METRICS_DIR, "X_train_mm.npy")
    y_ref_path = os.path.join(METRICS_DIR, "y_train.npy")
    if not os.path.exists(X_ref_path):
        print("[ERROR] Données d'entraînement introuvables. Exécuter ton_01_eda_and_stage1.py d'abord.")
        sys.exit(1)

    X_train = np.load(X_ref_path)
    y_train = np.load(y_ref_path)

    with open(os.path.join(METRICS_DIR, "label_classes.json")) as f:
        classes = json.load(f)["classes"]
    normal_idx  = list(classes).index("normal")
    X_normal    = X_train[y_train == normal_idx]
    print(f"  Flux normaux d'entraînement : {len(X_normal):,}")
    print(f"  Bootstrap MMD (n={args.n_bootstrap} permutations, α={args.alpha})...")

    t0 = time.time()
    threshold, mu_h0, std_h0 = bootstrap_mmd_threshold(
        X_normal, n_bootstrap=args.n_bootstrap,
        window_size=args.window_size, alpha=args.alpha, seed=SEEDS[0]
    )
    t_cal = time.time() - t0

    # Estimate sigma for detector
    rng   = np.random.RandomState(SEEDS[0])
    idx   = rng.choice(len(X_normal), size=min(1000, len(X_normal)), replace=False)
    X_sub = X_normal[idx]
    dists = np.sum((X_sub[:, None] - X_sub[None, :]) ** 2, axis=-1)
    sigma = float(np.sqrt(np.median(dists[dists > 0]) / 2.0))

    config = {
        "threshold":    threshold,
        "sigma":        sigma,
        "mu_h0":        mu_h0,
        "std_h0":       std_h0,
        "alpha":        args.alpha,
        "window_size":  args.window_size,
        "n_bootstrap":  args.n_bootstrap,
        "calibrated_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_ref_samples": len(X_normal),
        "ref_sample_size": min(500, len(X_normal)),
        "check_every":  100,
    }
    with open(DRIFT_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Seuil MMD² (α={args.alpha}) : {threshold:.6f}")
    print(f"  Distribution H₀ : μ={mu_h0:.6f} ± {std_h0:.6f}")
    print(f"  Sigma RBF        : {sigma:.4f}")
    print(f"  Calibration en   : {t_cal:.1f}s")
    print(f"  Config sauvegardée : {DRIFT_CONFIG_PATH}")


# ── Simulation de dérive ───────────────────────────────────────────────────────

def simulate_drift(args):
    """Test avec dérive synthétique injectée."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n=== Simulation de dérive conceptuelle ===\n")
    if not os.path.exists(DRIFT_CONFIG_PATH):
        print("[ERROR] Configuration MMD non trouvée. Exécuter d'abord --calibrate.")
        sys.exit(1)

    with open(DRIFT_CONFIG_PATH) as f:
        cfg = json.load(f)

    X_train = np.load(os.path.join(METRICS_DIR, "X_train_mm.npy"))
    y_train = np.load(os.path.join(METRICS_DIR, "y_train.npy"))
    with open(os.path.join(METRICS_DIR, "label_classes.json")) as f:
        classes = json.load(f)["classes"]
    normal_idx = list(classes).index("normal")
    X_normal   = X_train[y_train == normal_idx]

    detector = ConceptDriftDetector(
        X_reference=X_normal,
        threshold=cfg["threshold"],
        sigma=cfg["sigma"],
        window_size=cfg["window_size"],
        check_every=cfg["check_every"],
        verbose=True,
    )

    rng = np.random.RandomState(SEEDS[0])
    n_total     = 3000
    drift_start = 1500     # inject drift at flow #1500
    drift_shift = 0.5      # shift mean by 0.5 (in [0,1] scaled space)

    print(f"  Simulation {n_total} flux | dérive injectée à flux #{drift_start}")
    print(f"  Décalage de distribution : +{drift_shift} sur {X_normal.shape[1]} features\n")

    idx_pool = rng.choice(len(X_normal), size=n_total, replace=True)
    mmd_trace = []
    n_trace   = []

    for i, idx in enumerate(idx_pool):
        x = X_normal[idx].copy()
        if i >= drift_start:
            # Simulate new ICS device: shift + small noise
            x = np.clip(x + drift_shift * rng.randn(len(x)) * 0.1 + drift_shift * 0.3, 0, 1)
        detector.update(x)
        if detector.mmd_history and detector.mmd_history[-1][0] == detector._n_seen:
            mmd_trace.append(detector.mmd_history[-1][1])
            n_trace.append(i)

    # Plot
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(n_trace, mmd_trace, color=PALETTE[2], linewidth=1.5, label="MMD²")
    ax.axhline(cfg["threshold"], color='red', linestyle='--', linewidth=1.5,
               label=f"Seuil α={cfg['alpha']} ({cfg['threshold']:.4f})")
    ax.axvline(drift_start, color='orange', linestyle=':', linewidth=2,
               label=f"Dérive injectée (flux #{drift_start})")
    for evt in detector.drift_events:
        ax.axvline(evt["n_flows_seen"], color='red', alpha=0.3)
    ax.set_xlabel("Nombre de flux observés")
    ax.set_ylabel("MMD² (distance de distribution)")
    ax.set_title("Détection de dérive conceptuelle — Simulation ICS\n"
                 "Fenêtre glissante MMD² (kernel RBF) vs seuil bootstrap")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig_drift_simulation.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  Dérives détectées : {len(detector.drift_events)}")
    for evt in detector.drift_events:
        print(f"    Flux #{evt['n_flows_seen']} | MMD²={evt['mmd_squared']:.4f} "
              f"| ratio×{evt['ratio']:.2f} | {evt['severity']}")
    print(f"  Figure sauvegardée : {fig_path}")

    # Save log
    with open(DRIFT_LOG_PATH, "a") as f:
        for evt in detector.drift_events:
            f.write(json.dumps(evt) + "\n")
    print(f"  Événements sauvegardés : {DRIFT_LOG_PATH}")


# ── Stream monitoring ─────────────────────────────────────────────────────────

def monitor_stream(args):
    """Monitor a live flow stream for concept drift."""
    if not os.path.exists(DRIFT_CONFIG_PATH):
        print("[ERROR] Configuration MMD non trouvée. Exécuter d'abord --calibrate.")
        sys.exit(1)

    with open(DRIFT_CONFIG_PATH) as f:
        cfg = json.load(f)

    X_train    = np.load(os.path.join(METRICS_DIR, "X_train_mm.npy"))
    y_train    = np.load(os.path.join(METRICS_DIR, "y_train.npy"))
    with open(os.path.join(METRICS_DIR, "label_classes.json")) as f:
        classes = json.load(f)["classes"]
    normal_idx = list(classes).index("normal")
    X_normal   = X_train[y_train == normal_idx]

    with open(os.path.join(METRICS_DIR, "dataset_meta.json")) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    n_features = len(feature_names)

    detector = ConceptDriftDetector(
        X_reference=X_normal,
        threshold=cfg["threshold"],
        sigma=cfg["sigma"],
        window_size=cfg["window_size"],
        check_every=cfg["check_every"],
    )

    print(f"\n=== Surveillance dérive conceptuelle — flux en direct ===")
    print(f"  Seuil MMD² : {cfg['threshold']:.6f}  |  Fenêtre : {cfg['window_size']} flux")
    print(f"  Lecture depuis : {args.stream}\n")

    with open(args.stream) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            features = []
            for feat in feature_names:
                v = obj.get(feat, 0.0)
                try:
                    features.append(float(v))
                except (TypeError, ValueError):
                    features.append(0.0)
            if len(features) != n_features:
                continue
            x = np.array(features, dtype=np.float32)
            evt = detector.update(x)
            if evt:
                with open(DRIFT_LOG_PATH, "a") as lf:
                    lf.write(json.dumps(evt) + "\n")

    status = detector.status()
    print(f"\nFin du flux. {status['n_flows_seen']} flux analysés.")
    print(f"  Dérives détectées : {status['n_drift_events']}")
    print(f"  Dernier MMD²      : {status['last_mmd']:.6f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    if args.calibrate:
        calibrate(args)
    elif args.simulate:
        simulate_drift(args)
    elif args.stream:
        monitor_stream(args)
    else:
        print("Usage: --calibrate | --simulate | --stream flows.jsonl")
        print("  --calibrate  : Calibrer le seuil MMD sur données de référence")
        print("  --simulate   : Test avec dérive synthétique injectée")
        print("  --stream F   : Surveiller un flux en direct")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Concept Drift Detector (MMD) — Cynergia")
    p.add_argument("--calibrate", action="store_true",
                   help="Calibrer le seuil MMD sur données normales d'entraînement")
    p.add_argument("--simulate",  action="store_true",
                   help="Simuler une dérive synthétique pour valider le détecteur")
    p.add_argument("--stream",    type=str, default=None,
                   help="Fichier de flux JSON (un objet par ligne) à surveiller")
    p.add_argument("--n-bootstrap", type=int, default=200,
                   help="Nombre de permutations bootstrap (défaut: 200)")
    p.add_argument("--alpha", type=float, default=0.01,
                   help="Niveau de signification (défaut: 0.01)")
    p.add_argument("--window-size", type=int, default=500,
                   help="Taille fenêtre glissante en flux (défaut: 500)")
    main(p.parse_args())
