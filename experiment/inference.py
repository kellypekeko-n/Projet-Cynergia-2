"""
inference.py — Pipeline d'inférence pour de nouveaux flux réseau.

Usage :
  # Inférence sur un seul flux
  python inference.py --input single --features feature_vector.json

  # Inférence sur un fichier CSV
  python inference.py --input csv --file new_traffic.csv

  # Mode interactif
  python inference.py --input interactive

Description :
  Ce script charge le pipeline complet (Stage-1 + Stage-2) sauvegardé
  et prédit la classe MITRE ATT&CK pour chaque nouveau flux réseau.
"""

import sys, os, json, argparse
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ton_iot_config import *


class HybridICSPipeline:
    """
    Pipeline hybride complet : OCSVM (Stage-1) + XGBoost (Stage-2).
    Chargez et utilisez ce pipeline pour prédire sur de nouveaux flux.
    """

    def __init__(self):
        self.stage1 = None
        self.stage2 = None
        self.class_names = None
        self.scaler_mm  = None
        self.scaler_std = None
        self.feature_names = None
        self.theta = None

    def load(self, stage1_path=None, stage2_path=None,
              scalers_path=None):
        """Charge le pipeline depuis les fichiers sauvegardés."""

        # Stage 1
        s1_path = stage1_path or os.path.join(
            RESULTS_DIR, "saved_models", "ocsvm.pkl")
        if os.path.exists(s1_path):
            data = joblib.load(s1_path)
            self.stage1      = data["model"]
            self.theta       = data["theta"]
            self.class_names = data["class_names"]
            print(f"Stage-1 chargé : {s1_path}")
        else:
            print(f"Stage-1 non trouvé : {s1_path}")

        # Stage 2
        s2_path = stage2_path or os.path.join(
            RESULTS_DIR, "saved_models", "xgboost_hybrid.pkl")
        if os.path.exists(s2_path):
            data = joblib.load(s2_path)
            self.stage2 = data["model"]
            print(f"Stage-2 chargé : {s2_path}")
        else:
            print(f"Stage-2 non trouvé : {s2_path}")

        # Scalers (recreate from saved train data)
        if not scalers_path:
            self._fit_scalers()

    def _fit_scalers(self):
        """Recrée les scalers à partir des données d'entraînement."""
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        X_train = np.load(os.path.join(METRICS_DIR, "X_train_raw.npy"))
        self.scaler_mm  = MinMaxScaler().fit(X_train)
        self.scaler_std = StandardScaler().fit(X_train)
        with open(os.path.join(METRICS_DIR, "dataset_meta.json")) as f:
            meta = json.load(f)
        self.feature_names = meta["feature_names"]
        print("Scalers recréés depuis les données d'entraînement.")

    def preprocess(self, X_raw):
        """
        Applique le même prétraitement que pendant l'entraînement.

        IMPORTANT : Utiliser EXACTEMENT le même scaler que pendant
        l'entraînement. Ne pas re-fit le scaler sur les nouvelles données !
        """
        if self.scaler_mm is None:
            self._fit_scalers()
        return (self.scaler_mm.transform(X_raw),
                self.scaler_std.transform(X_raw))

    def predict(self, X_raw):
        """
        Prédit la classe MITRE ATT&CK pour de nouveaux flux.

        Retourne :
            predictions : liste de dictionnaires avec :
                'class'        : nom de la classe prédite
                'mitre_code'   : code MITRE ATT&CK
                'confidence'   : probabilité de la classe prédite
                'stage1_score' : score d'anomalie Stage-1
                'flagged'      : True si flaggé par Stage-1
        """
        X_mm, X_std = self.preprocess(X_raw)
        results = []

        for i in range(len(X_raw)):
            xi_mm  = X_mm[i:i+1]
            xi_std = X_std[i:i+1]

            # Stage 1 : détecter l'anomalie
            if self.stage1 is not None:
                s1_score = float(-self.stage1.decision_function(xi_mm)[0])
                flagged  = s1_score >= self.theta
            else:
                s1_score = 0.0
                flagged  = True  # sans Stage-1, tout passe à Stage-2

            if flagged and self.stage2 is not None:
                # Stage 2 : classifier la technique d'attaque
                xi_aug = np.column_stack([xi_std, [[s1_score]]])
                proba  = self.stage2.predict_proba(xi_aug)[0]
                cls_id = proba.argmax()
                cls_name = self.class_names[cls_id]
                confidence = float(proba[cls_id])
            else:
                cls_name   = "normal"
                confidence = 1.0 - float(min(s1_score, 1.0))

            results.append({
                "class":        cls_name,
                "mitre_code":   MITRE_MAP.get(cls_name, "N/A"),
                "confidence":   round(confidence, 4),
                "stage1_score": round(s1_score, 4),
                "flagged":      bool(flagged),
            })

        return results

    def predict_from_csv(self, csv_path):
        """Charge un CSV et prédit sur chaque ligne."""
        import pandas as pd
        df = pd.read_csv(csv_path)

        # Vérifier que les features sont présentes
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            print(f"Features manquantes dans le CSV : {missing}")
            return None

        X_raw = df[self.feature_names].values.astype(np.float32)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        return self.predict(X_raw)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline d'inférence ICS/IIoT"
    )
    parser.add_argument("--input", choices=["single", "csv", "interactive"],
                        default="interactive")
    parser.add_argument("--features", type=str,
                        help="JSON avec les valeurs des features")
    parser.add_argument("--file", type=str,
                        help="CSV avec les nouvelles données")
    args = parser.parse_args()

    pipeline = HybridICSPipeline()
    pipeline.load()

    if args.input == "csv" and args.file:
        results = pipeline.predict_from_csv(args.file)
        if results:
            print(f"\nPrédictions pour {len(results)} flux :")
            for i, r in enumerate(results):
                if r["class"] != "normal":
                    print(f"  Flux {i:4d} : {r['class']:12s} | "
                          f"MITRE: {r['mitre_code']:45s} | "
                          f"Conf: {r['confidence']:.3f}")

    elif args.input == "single" and args.features:
        with open(args.features) as f:
            features_dict = json.load(f)
        X_raw = np.array([[features_dict.get(f, 0.0)
                            for f in pipeline.feature_names]],
                          dtype=np.float32)
        results = pipeline.predict(X_raw)
        print(f"\nRésultat :")
        print(json.dumps(results[0], indent=2))

    else:
        # Mode démo : tester sur quelques flux du test set
        print("\n=== MODE DÉMO : test sur quelques flux réels ===")
        X_test  = np.load(os.path.join(METRICS_DIR, "X_test_raw.npy"))
        y_test  = np.load(os.path.join(METRICS_DIR, "y_test.npy"))
        with open(os.path.join(METRICS_DIR, "label_classes.json")) as f:
            class_names = json.load(f)["classes"]

        # Sélectionner 5 exemples de classes stealthy
        for target_cls in STEALTHY_CLASSES:
            if target_cls not in class_names:
                continue
            cls_id = class_names.index(target_cls)
            idx    = np.where(y_test == cls_id)[0]
            if len(idx) == 0:
                continue
            sample = X_test[idx[0]:idx[0]+1]
            result = pipeline.predict(sample)[0]
            true_cls = class_names[y_test[idx[0]]]
            print(f"\n  Vrai    : {true_cls}")
            print(f"  Prédit  : {result['class']}")
            print(f"  MITRE   : {result['mitre_code']}")
            print(f"  Score S1: {result['stage1_score']:.4f} "
                  f"({'FLAGGÉ' if result['flagged'] else 'normal'})")
            print(f"  Conf.   : {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
