"""
inference.py — Pipeline d'inférence pour de nouveaux flux réseau.

Usage :
  # Mode démo (test sur le dataset existant)
  python inference.py --input interactive

  # Inférence sur un seul flux (JSON de features)
  python inference.py --input single --features feature_vector.json

  # Inférence batch sur un fichier CSV
  python inference.py --input csv --file new_traffic.csv

  # Flux continu depuis stdin (JSON une ligne par flux)
  zeek_parser | python inference.py --input stream

  # Flux continu en surveillant un fichier de log (tail -f)
  python inference.py --input stream --file /var/log/zeek/conn.log

  # Rejouer un fichier existant depuis le début (test / simulation)
  python inference.py --input stream --file traffic.log --replay

  # Ajuster le seuil de confiance (défaut=0.70)
  python inference.py --input stream --file traffic.log --threshold 0.80

  # N'afficher que les alertes (pas les flux normaux)
  python inference.py --input stream --alerts-only

  # Écrire les alertes dans un fichier JSON
  python inference.py --input stream --output-log alerts.jsonl

Description du seuil de confiance :
  Si Stage-2 prédit une classe avec probabilité < threshold,
  le flux est étiqueté UNKNOWN_THREAT au lieu de forcer une classe
  erronée. Permet de détecter les attaques zero-day.
"""

import sys, os, json, time, signal, argparse
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ton_iot_config import *


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class HybridICSPipeline:
    """Pipeline hybride complet : OCSVM (Stage-1) + XGBoost (Stage-2)."""

    DEFAULT_THRESHOLD = 0.70

    def __init__(self, confidence_threshold=None):
        self.stage1 = None
        self.stage2 = None
        self.class_names = None
        self.scaler_mm  = None
        self.scaler_std = None
        self.feature_names = None
        self.theta = None
        self._unknown_id = None
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.DEFAULT_THRESHOLD
        )

    def load(self, stage1_path=None, stage2_path=None):
        # Stage 1
        s1_path = stage1_path or os.path.join(RESULTS_DIR, "saved_models", "ocsvm.pkl")
        if os.path.exists(s1_path):
            data = joblib.load(s1_path)
            self.stage1      = data["model"]
            self.theta       = data["theta"]
            self.class_names = data["class_names"]
            print(f"Stage-1 chargé : {s1_path}")
        else:
            print(f"Stage-1 non trouvé : {s1_path}")

        # Stage 2 — open-set en priorité, sinon standard
        s2_openset  = os.path.join(RESULTS_DIR, "saved_models", "xgboost_openset_hybrid.pkl")
        s2_standard = os.path.join(RESULTS_DIR, "saved_models", "xgboost_hybrid.pkl")
        s2_path = stage2_path or (s2_openset if os.path.exists(s2_openset) else s2_standard)
        if os.path.exists(s2_path):
            data = joblib.load(s2_path)
            self.stage2      = data["model"]
            self.class_names = data["class_names"]
            self._unknown_id = data.get("unknown_id", None)
            tag = "open-set" if "openset" in s2_path else "standard"
            print(f"Stage-2 chargé ({tag}) : {s2_path}")
        else:
            print(f"Stage-2 non trouvé : {s2_path}")

        self._fit_scalers()

    def _fit_scalers(self):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        X_train = np.load(os.path.join(METRICS_DIR, "X_train_raw.npy"))
        self.scaler_mm  = MinMaxScaler().fit(X_train)
        self.scaler_std = StandardScaler().fit(X_train)
        with open(os.path.join(METRICS_DIR, "dataset_meta.json")) as f:
            meta = json.load(f)
        self.feature_names = meta["feature_names"]
        print("Scalers recréés depuis les données d'entraînement.")

    def preprocess(self, X_raw):
        return (self.scaler_mm.transform(X_raw),
                self.scaler_std.transform(X_raw))

    def predict(self, X_raw, confidence_threshold=None):
        """
        Prédit la classe MITRE ATT&CK pour un batch de flux.

        Retourne une liste de dicts :
            class          — classe prédite ou "UNKNOWN_THREAT"
            mitre_code     — technique MITRE ATT&CK
            confidence     — probabilité max de Stage-2
            unknown        — True si attaque inconnue
            top_candidates — top-3 classes probables
            stage1_score   — score d'anomalie OCSVM
            flagged        — True si flaggé par Stage-1
        """
        threshold = confidence_threshold or self.confidence_threshold
        X_mm, X_std = self.preprocess(X_raw)
        results = []

        for i in range(len(X_raw)):
            xi_mm  = X_mm[i:i+1]
            xi_std = X_std[i:i+1]

            if self.stage1 is not None:
                s1_score = float(-self.stage1.decision_function(xi_mm)[0])
                flagged  = s1_score >= self.theta
            else:
                s1_score = 0.0
                flagged  = True

            if flagged and self.stage2 is not None:
                xi_aug = np.column_stack([xi_std, [[s1_score]]])
                proba  = self.stage2.predict_proba(xi_aug)[0]

                top3_idx = np.argsort(proba)[::-1][:3]
                top_candidates = [
                    {"class": self.class_names[j], "proba": round(float(proba[j]), 4)}
                    for j in top3_idx
                ]

                cls_id     = int(proba.argmax())
                confidence = float(proba[cls_id])

                model_says_unknown    = (self._unknown_id is not None
                                         and cls_id == self._unknown_id)
                threshold_says_unknown = confidence < threshold

                if model_says_unknown or threshold_says_unknown:
                    cls_name = "UNKNOWN_THREAT"
                    unknown  = True
                else:
                    cls_name = self.class_names[cls_id]
                    unknown  = False
            else:
                cls_name       = "normal"
                confidence     = 1.0 - float(min(s1_score, 1.0))
                unknown        = False
                top_candidates = [{"class": "normal", "proba": round(confidence, 4)}]

            results.append({
                "class":          cls_name,
                "mitre_code":     MITRE_MAP.get(cls_name,
                                                "ALERTE — menace inconnue, analyste requis"),
                "confidence":     round(confidence, 4),
                "unknown":        unknown,
                "top_candidates": top_candidates,
                "stage1_score":   round(s1_score, 4),
                "flagged":        bool(flagged),
            })

        return results

    def predict_from_csv(self, csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            print(f"Features manquantes dans le CSV : {missing}")
            return None
        X_raw = df[self.feature_names].values.astype(np.float32)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        return self.predict(X_raw)


# ─────────────────────────────────────────────────────────────────────────────
# AFFICHAGE BATCH
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(i, r):
    if r["class"] == "normal":
        return
    tag = "[!!! INCONNUE]" if r["unknown"] else "[ATTAQUE]    "
    print(f"  Flux {i:4d} {tag} : {r['class']:16s} | "
          f"Conf: {r['confidence']:.3f} | S1: {r['stage1_score']:.3f}")
    if r["unknown"]:
        print(f"           MITRE : {r['mitre_code']}")
        print(f"           Candidats :")
        for c in r["top_candidates"]:
            print(f"             - {c['class']:12s} ({c['proba']:.3f}) | "
                  f"{MITRE_MAP.get(c['class'], 'N/A')}")
    else:
        print(f"           MITRE : {r['mitre_code']}")


# ─────────────────────────────────────────────────────────────────────────────
# MODE STREAM
# ─────────────────────────────────────────────────────────────────────────────

class StreamStats:
    """Compteurs temps réel pour le mode stream."""

    def __init__(self):
        self.total    = 0
        self.normal   = 0
        self.attack   = 0
        self.unknown  = 0
        self.errors   = 0
        self.t_start  = time.time()
        self.attack_counts = {}   # {class_name: count}

    def record(self, r):
        self.total += 1
        if r["class"] == "normal":
            self.normal += 1
        elif r["unknown"]:
            self.unknown += 1
            self.attack_counts["UNKNOWN_THREAT"] = (
                self.attack_counts.get("UNKNOWN_THREAT", 0) + 1
            )
        else:
            self.attack += 1
            self.attack_counts[r["class"]] = (
                self.attack_counts.get(r["class"], 0) + 1
            )

    def summary(self):
        elapsed = time.time() - self.t_start
        rate    = self.total / max(elapsed, 1)
        lines   = [
            f"\n{'─'*55}",
            f"  STATISTIQUES STREAM — {time.strftime('%H:%M:%S')}",
            f"{'─'*55}",
            f"  Flux traités     : {self.total:>8,}  ({rate:.1f} flux/s)",
            f"  Normaux          : {self.normal:>8,}  ({self.normal/max(self.total,1)*100:.1f}%)",
            f"  Attaques connues : {self.attack:>8,}  ({self.attack/max(self.total,1)*100:.1f}%)",
            f"  Menaces inconnues: {self.unknown:>8,}  ({self.unknown/max(self.total,1)*100:.1f}%)  ← analyste",
            f"  Erreurs parsing  : {self.errors:>8,}",
        ]
        if self.attack_counts:
            lines.append(f"  Détail attaques :")
            for cls, cnt in sorted(self.attack_counts.items(),
                                   key=lambda x: -x[1]):
                lines.append(f"    {cls:18s}: {cnt:,}")
        lines.append(f"{'─'*55}")
        return "\n".join(lines)


def _parse_flow_json(line, feature_names):
    """Parse une ligne JSON → vecteur numpy."""
    data = json.loads(line.strip())
    x = np.array(
        [[float(data.get(f, 0.0)) for f in feature_names]],
        dtype=np.float32
    )
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _parse_flow_csv(line, header, feature_names):
    """Parse une ligne CSV (avec header déjà connu) → vecteur numpy."""
    values = line.strip().split(",")
    row    = dict(zip(header, values))
    x = np.array(
        [[float(row.get(f, 0.0)) for f in feature_names]],
        dtype=np.float32
    )
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _print_stream_alert(flux_id, r, timestamp=None):
    """Affiche une alerte temps réel avec horodatage."""
    ts  = timestamp or time.strftime("%H:%M:%S")
    sep = "!!!" if r["unknown"] else "---"

    if r["unknown"]:
        print(f"\n[{ts}] {sep} MENACE INCONNUE #{flux_id} {sep}")
        print(f"  Classe     : UNKNOWN_THREAT")
        print(f"  Confiance  : {r['confidence']:.3f}  (sous le seuil)")
        print(f"  Score S1   : {r['stage1_score']:.4f}")
        print(f"  Action     : {r['mitre_code']}")
        print(f"  Candidats  :")
        for c in r["top_candidates"]:
            print(f"    - {c['class']:12s} p={c['proba']:.3f}  "
                  f"{MITRE_MAP.get(c['class'], '')}")
    else:
        print(f"[{ts}] {sep} ATTAQUE #{flux_id:6d} : {r['class']:12s} | "
              f"Conf={r['confidence']:.3f} | S1={r['stage1_score']:.4f} | "
              f"{r['mitre_code']}")


def run_stream(pipeline, source, fmt="json", stats_every=100,
               alerts_only=False, output_log=None):
    """
    Boucle de traitement temps réel.

    source : itérable de lignes (stdin ou file generator)
    fmt    : 'json' ou 'csv'
    """
    stats   = StreamStats()
    log_fh  = open(output_log, "a", encoding="utf-8") if output_log else None
    header  = None   # pour le mode CSV

    def shutdown(sig, frame):
        print(stats.summary())
        print("\nStream arrêté.")
        if log_fh:
            log_fh.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    print(f"Stream démarré — format={fmt} | seuil={pipeline.confidence_threshold:.2f}")
    print("Ctrl+C pour arrêter et afficher les statistiques finales.\n")

    flux_id = 0
    for raw_line in source:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Gestion de l'en-tête CSV
        if fmt == "csv" and header is None:
            header = line.split(",")
            continue

        try:
            if fmt == "json":
                X_raw = _parse_flow_json(line, pipeline.feature_names)
            else:
                X_raw = _parse_flow_csv(line, header, pipeline.feature_names)
        except Exception as e:
            stats.errors += 1
            continue

        result = pipeline.predict(X_raw)[0]
        flux_id += 1
        stats.record(result)

        is_alert = result["class"] != "normal"

        if is_alert:
            _print_stream_alert(flux_id, result)
            if log_fh:
                log_fh.write(json.dumps({
                    "flux_id":   flux_id,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    **result
                }) + "\n")
                log_fh.flush()
        elif not alerts_only:
            # Afficher un point par flux normal pour montrer l'activité
            print(".", end="", flush=True)
            if flux_id % 80 == 0:
                print()

        if flux_id % stats_every == 0:
            print(stats.summary())

    # Fin du fichier
    print(stats.summary())
    if log_fh:
        log_fh.close()


def _file_tail_generator(path):
    """Génère les nouvelles lignes ajoutées à un fichier (tail -f)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, 2)   # aller à la fin du fichier
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                time.sleep(0.05)


def _file_replay_generator(path):
    """Rejoue un fichier depuis le début (pour les tests)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline d'inférence ICS/IIoT")
    parser.add_argument("--input",
                        choices=["single", "csv", "stream", "interactive"],
                        default="interactive",
                        help="Source des données")
    parser.add_argument("--features", type=str,
                        help="[single] JSON avec les valeurs des features")
    parser.add_argument("--file", type=str,
                        help="[csv/stream] Fichier à lire")
    parser.add_argument("--threshold", type=float,
                        default=HybridICSPipeline.DEFAULT_THRESHOLD,
                        help=f"Seuil de confiance Stage-2 "
                             f"(défaut={HybridICSPipeline.DEFAULT_THRESHOLD})")
    # Options spécifiques au mode stream
    parser.add_argument("--format", choices=["json", "csv"], default="json",
                        help="[stream] Format des lignes d'entrée (défaut: json)")
    parser.add_argument("--replay", action="store_true",
                        help="[stream] Rejouer un fichier depuis le début "
                             "au lieu de surveiller les nouvelles lignes")
    parser.add_argument("--stats-every", type=int, default=100,
                        help="[stream] Afficher les stats tous les N flux (défaut: 100)")
    parser.add_argument("--alerts-only", action="store_true",
                        help="[stream] N'afficher que les alertes, pas les flux normaux")
    parser.add_argument("--output-log", type=str, default=None,
                        help="[stream] Fichier .jsonl pour enregistrer les alertes")
    args = parser.parse_args()

    pipeline = HybridICSPipeline(confidence_threshold=args.threshold)
    pipeline.load()
    print(f"Seuil de confiance actif : {args.threshold:.2f}\n")

    # ── Mode stream ───────────────────────────────────────────────────────────
    if args.input == "stream":
        if args.file:
            if args.replay:
                print(f"Mode REPLAY depuis : {args.file}")
                source = _file_replay_generator(args.file)
            else:
                print(f"Mode TAIL (surveillance) : {args.file}")
                source = _file_tail_generator(args.file)
        else:
            print("Mode STDIN — en attente de flux JSON (une ligne par flux)...")
            source = sys.stdin

        run_stream(
            pipeline    = pipeline,
            source      = source,
            fmt         = args.format,
            stats_every = args.stats_every,
            alerts_only = args.alerts_only,
            output_log  = args.output_log,
        )

    # ── Mode CSV batch ────────────────────────────────────────────────────────
    elif args.input == "csv" and args.file:
        results   = pipeline.predict_from_csv(args.file)
        if results:
            n_unknown = sum(1 for r in results if r["unknown"])
            n_attack  = sum(1 for r in results if r["class"] not in ("normal", "UNKNOWN_THREAT"))
            print(f"Prédictions pour {len(results):,} flux :")
            print(f"  Attaques connues     : {n_attack}")
            print(f"  Menaces inconnues    : {n_unknown}  ← analyste requis")
            print()
            for i, r in enumerate(results):
                _print_result(i, r)

    # ── Mode single flux ──────────────────────────────────────────────────────
    elif args.input == "single" and args.features:
        with open(args.features) as f:
            features_dict = json.load(f)
        X_raw = np.array(
            [[features_dict.get(f, 0.0) for f in pipeline.feature_names]],
            dtype=np.float32
        )
        result = pipeline.predict(X_raw)[0]
        print(json.dumps(result, indent=2))

    # ── Mode démo ─────────────────────────────────────────────────────────────
    else:
        print("=== MODE DÉMO : test sur quelques flux réels du dataset ===\n")
        X_test  = np.load(os.path.join(METRICS_DIR, "X_test_raw.npy"))
        y_test  = np.load(os.path.join(METRICS_DIR, "y_test.npy"))
        with open(os.path.join(METRICS_DIR, "label_classes.json")) as f:
            class_names = json.load(f)["classes"]

        for target_cls in STEALTHY_CLASSES:
            if target_cls not in class_names:
                continue
            cls_id = class_names.index(target_cls)
            idx    = np.where(y_test == cls_id)[0]
            if len(idx) == 0:
                continue
            sample   = X_test[idx[0]:idx[0]+1]
            result   = pipeline.predict(sample)[0]
            true_cls = class_names[y_test[idx[0]]]
            print(f"  Vrai    : {true_cls}")
            print(f"  Prédit  : {result['class']}"
                  + (" [INCONNUE — alerte analyste]" if result["unknown"] else ""))
            print(f"  MITRE   : {result['mitre_code']}")
            print(f"  Score S1: {result['stage1_score']:.4f} "
                  f"({'FLAGGÉ' if result['flagged'] else 'normal'})")
            print(f"  Conf.   : {result['confidence']:.4f}  "
                  f"(seuil={args.threshold:.2f})")
            if result["unknown"]:
                print("  Candidats : "
                      + ", ".join(f"{c['class']}({c['proba']:.2f})"
                                  for c in result["top_candidates"]))
            print()


if __name__ == "__main__":
    main()
