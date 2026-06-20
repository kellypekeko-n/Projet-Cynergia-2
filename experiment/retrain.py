"""
retrain.py — Pipeline de ré-entraînement continu.

SCÉNARIO D'UTILISATION
──────────────────────
1. inference.py détecte des flux UNKNOWN_THREAT et les écrit dans alerts.jsonl
2. Un analyste humain examine ces alertes et les labélise
3. L'analyste lance ce script pour intégrer les nouveaux exemples au modèle
4. Le modèle est ré-entraîné et les métriques avant/après sont comparées

Usage :
  # Ajouter des nouvelles attaques labélisées et ré-entraîner
  python retrain.py --new-data alerts.jsonl --label supply_chain_attack

  # Ajouter plusieurs alertes avec des labels différents depuis un JSON
  python retrain.py --labeled-file new_attacks_labeled.json

  # Voir les classes actuelles du modèle
  python retrain.py --info

  # Simuler un ré-entraînement sans sauvegarder (dry run)
  python retrain.py --new-data alerts.jsonl --label backdoor --dry-run

Format de --labeled-file (JSON) :
  [
    {"features": {...}, "label": "supply_chain_attack"},
    {"features": {...}, "label": "backdoor"},
    ...
  ]
"""

import sys, os, json, time, joblib, argparse, shutil
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ton_iot_config import *

RETRAIN_LOG  = os.path.join(METRICS_DIR, "retrain_history.json")
BACKUP_DIR   = os.path.join(RESULTS_DIR, "saved_models", "backups")


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE ACTUEL
# ─────────────────────────────────────────────────────────────────────────────

def load_current_model(mode="hybrid"):
    """Charge le meilleur modèle Stage-2 disponible (open-set > standard)."""
    candidates = [
        os.path.join(RESULTS_DIR, "saved_models", f"xgboost_openset_{mode}.pkl"),
        os.path.join(RESULTS_DIR, "saved_models", f"xgboost_{mode}.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            data = joblib.load(path)
            print(f"Modèle actuel chargé : {path}")
            return data, path
    raise FileNotFoundError(
        f"Aucun modèle Stage-2 trouvé. Lancez d'abord train_xgboost.py."
    )


def load_scalers():
    """Charge les scalers pour prétraiter les nouvelles données."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    X_train = np.load(os.path.join(METRICS_DIR, "X_train_raw.npy"))
    scaler_mm  = MinMaxScaler().fit(X_train)
    scaler_std = StandardScaler().fit(X_train)
    with open(os.path.join(METRICS_DIR, "dataset_meta.json")) as f:
        feature_names = json.load(f)["features"]
    return scaler_mm, scaler_std, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# PRÉPARATION DES NOUVELLES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

def parse_new_samples(source, label, feature_names, scaler_mm, scaler_std):
    """
    Parse les nouvelles alertes labélisées.

    source : chemin vers alerts.jsonl (sortie de inference.py)
             ou liste de dicts {"features": {...}, "label": "..."}
    label  : étiquette à donner aux alertes (None si déjà dans le JSON)
    """
    samples = []

    if isinstance(source, str) and os.path.exists(source):
        with open(source, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)

                # Format : sortie directe de inference.py (alerte JSONL)
                if "top_candidates" in entry:
                    # On ne peut pas récupérer les features brutes depuis
                    # une alerte — on a besoin du flux original.
                    # Format attendu : l'analyste a joint les features brutes.
                    if "features" not in entry:
                        print(f"  [SKIP] Alerte sans features brutes (flux #{entry.get('flux_id','?')})")
                        continue
                    feat_dict = entry["features"]
                    lbl = label or entry.get("analyst_label")
                # Format : {"features": {...}, "label": "..."}
                elif "features" in entry:
                    feat_dict = entry["features"]
                    lbl = label or entry.get("label")
                else:
                    continue

                if not lbl:
                    print(f"  [SKIP] Pas de label pour l'entrée : {list(entry.keys())}")
                    continue

                x_raw = np.array(
                    [[float(feat_dict.get(f, 0.0)) for f in feature_names]],
                    dtype=np.float32
                )
                x_raw = np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)
                x_std = scaler_std.transform(x_raw)[0]
                samples.append((x_std, lbl))

    return samples


def integrate_new_class(X_train, y_train, class_names,
                         new_samples, new_label):
    """
    Ajoute les nouveaux échantillons au dataset d'entraînement.

    Si new_label est déjà une classe connue → renforce cette classe.
    Si new_label est nouveau → crée une nouvelle classe.
    """
    if new_label in class_names:
        cls_id = class_names.index(new_label)
        print(f"  Classe existante '{new_label}' (id={cls_id}) — "
              f"ajout de {len(new_samples)} exemples supplémentaires.")
    else:
        cls_id = len(class_names)
        class_names = class_names + [new_label]
        print(f"  Nouvelle classe '{new_label}' créée (id={cls_id}).")

    X_new = np.array([x for x, _ in new_samples], dtype=np.float32)
    y_new = np.full(len(X_new), cls_id, dtype=np.int64)

    X_aug = np.vstack([X_train, X_new])
    y_aug = np.concatenate([y_train, y_new])

    return X_aug, y_aug, class_names


# ─────────────────────────────────────────────────────────────────────────────
# RÉ-ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────

def retrain(X_train, y_train, class_names, hyperparams):
    """Ré-entraîne XGBoost sur le dataset augmenté."""
    from xgboost import XGBClassifier
    print(f"\n  Ré-entraînement sur {len(y_train):,} flux "
          f"({len(class_names)} classes)...")
    t0    = time.time()
    model = XGBClassifier(**hyperparams)
    model.fit(X_train, y_train)
    t_fit = time.time() - t0
    print(f"  Terminé en {t_fit:.1f}s")
    return model, t_fit


def evaluate(model, X_test, y_test, class_names, label=""):
    """Évalue le modèle et retourne les métriques clés."""
    from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
    n_known = len(class_names)
    y_pred  = model.predict(X_test)

    # Ignorer les classes absentes du test set
    labels_present = list(np.unique(np.concatenate([y_test, y_pred])))
    labels_present = [l for l in labels_present if l < n_known]

    f1m = f1_score(y_test, y_pred, average='macro',
                   labels=labels_present, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    stealthy_recalls = {}
    rpt = classification_report(y_test, y_pred,
                                  target_names=class_names[:n_known],
                                  output_dict=True, zero_division=0)
    for cls in STEALTHY_CLASSES:
        if cls in rpt:
            stealthy_recalls[cls] = round(rpt[cls]['recall'], 4)

    print(f"\n  [{label}] F1-macro={f1m:.4f} | MCC={mcc:.4f}")
    print(f"  Recalls furtives : {stealthy_recalls}")
    return {"f1_macro": round(f1m, 4), "mcc": round(mcc, 4),
            "stealthy_recalls": stealthy_recalls}


# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE ET HISTORIQUE
# ─────────────────────────────────────────────────────────────────────────────

def backup_model(model_path):
    """Crée une copie de sauvegarde avant d'écraser le modèle."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = os.path.basename(model_path).replace(".pkl", f"_{ts}.pkl")
    dst  = os.path.join(BACKUP_DIR, name)
    shutil.copy2(model_path, dst)
    print(f"  Backup : {dst}")
    return dst


def save_retrain_log(entry):
    """Ajoute une entrée à l'historique de ré-entraînement."""
    history = []
    if os.path.exists(RETRAIN_LOG):
        with open(RETRAIN_LOG) as f:
            history = json.load(f)
    history.append(entry)
    with open(RETRAIN_LOG, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Historique mis à jour : {RETRAIN_LOG}")


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    print("=" * 60)
    print("CYNERGIA — Pipeline de ré-entraînement continu")
    print("=" * 60)

    # ── Info seulement ────────────────────────────────────────────────────────
    if args.info:
        model_data, path = load_current_model(args.mode)
        print(f"\nModèle : {path}")
        print(f"Classes ({len(model_data['class_names'])}) :")
        for i, c in enumerate(model_data["class_names"]):
            print(f"  {i:2d}. {c}")
        if os.path.exists(RETRAIN_LOG):
            with open(RETRAIN_LOG) as f:
                history = json.load(f)
            print(f"\nHistorique ré-entraînements : {len(history)} session(s)")
            for h in history[-3:]:
                delta = h.get("delta_f1_macro", 0)
                sign  = "+" if delta >= 0 else ""
                print(f"  {h['timestamp']} | {h['new_label']:20s} | "
                      f"F1-macro {sign}{delta:.4f} | "
                      f"{h['n_new_samples']} nouveaux exemples")
        return

    # ── Chargement ────────────────────────────────────────────────────────────
    print("\n1. Chargement du modèle actuel...")
    model_data, model_path = load_current_model(args.mode)

    print("\n2. Chargement des scalers...")
    try:
        scaler_mm, scaler_std, feature_names = load_scalers()
    except Exception as e:
        print(f"  Erreur chargement scalers : {e}")
        print("  Vérifiez que le pipeline a été exécuté (ton_00 → ton_01).")
        return

    # ── Parsing des nouvelles données ─────────────────────────────────────────
    print("\n3. Parsing des nouvelles données labélisées...")
    source = args.new_data or args.labeled_file
    if not source:
        print("  Aucune source fournie. Utilisez --new-data ou --labeled-file.")
        return

    new_samples = parse_new_samples(
        source, args.label, feature_names, scaler_mm, scaler_std
    )
    if not new_samples:
        print("  Aucun échantillon valide trouvé. Vérifiez le format du fichier.")
        return
    print(f"  {len(new_samples)} nouveaux échantillons chargés.")

    # ── Chargement du dataset d'entraînement ──────────────────────────────────
    print("\n4. Chargement du dataset d'entraînement existant...")
    if args.mode == "hybrid":
        flag_tr = np.load(os.path.join(METRICS_DIR, "s1_flag_train.npy"))
        sc_tr   = np.load(os.path.join(METRICS_DIR, "s1_score_train.npy"))
        X_tr    = np.load(os.path.join(METRICS_DIR, "X_train_std.npy"))
        y_tr    = np.load(os.path.join(METRICS_DIR, "y_train.npy"))
        X_train = np.column_stack([X_tr[flag_tr], sc_tr[flag_tr].reshape(-1, 1)])
        y_train = y_tr[flag_tr]
    else:
        X_train = np.load(os.path.join(METRICS_DIR, "X_train_std.npy"))
        y_train = np.load(os.path.join(METRICS_DIR, "y_train.npy"))
    print(f"  Dataset actuel : {X_train.shape}")

    X_test = np.load(os.path.join(METRICS_DIR, "X_test_std.npy"))
    y_test = np.load(os.path.join(METRICS_DIR, "y_test.npy"))

    class_names = model_data["class_names"]

    # ── Évaluation AVANT ──────────────────────────────────────────────────────
    print("\n5. Métriques AVANT ré-entraînement...")
    metrics_before = evaluate(
        model_data["model"], X_test, y_test, class_names, label="AVANT"
    )

    # ── Intégration des nouvelles données ─────────────────────────────────────
    print(f"\n6. Intégration de '{args.label or 'labels mixtes'}'...")
    new_label = args.label or "mixed_new"
    X_aug, y_aug, class_names_new = integrate_new_class(
        X_train, y_train, list(class_names), new_samples, new_label
    )
    print(f"  Dataset augmenté : {X_aug.shape} "
          f"(+{len(new_samples)} → {len(y_aug):,} total)")

    if args.dry_run:
        print("\n[DRY RUN] Ré-entraînement simulé — aucun fichier modifié.")
        return

    # ── Ré-entraînement ───────────────────────────────────────────────────────
    print("\n7. Ré-entraînement...")
    hyperparams = model_data.get("hyperparams", {
        "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "eval_metric": "mlogloss",
        "verbosity": 0, "random_state": 42, "n_jobs": -1,
    })
    new_model, t_fit = retrain(X_aug, y_aug, class_names_new, hyperparams)

    # ── Évaluation APRÈS ──────────────────────────────────────────────────────
    print("\n8. Métriques APRÈS ré-entraînement...")
    metrics_after = evaluate(new_model, X_test, y_test, class_names_new, label="APRÈS")

    delta_f1 = metrics_after["f1_macro"] - metrics_before["f1_macro"]
    delta_mcc = metrics_after["mcc"] - metrics_before["mcc"]
    print(f"\n  ΔF1-macro : {delta_f1:+.4f}")
    print(f"  ΔMCC      : {delta_mcc:+.4f}")
    if new_label not in class_names:
        print(f"  Nouvelle classe '{new_label}' intégrée au modèle.")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    print("\n9. Sauvegarde...")
    backup_model(model_path)

    joblib.dump({
        "model":        new_model,
        "class_names":  class_names_new,
        "unknown_id":   model_data.get("unknown_id"),
        "hyperparams":  hyperparams,
        "mode":         args.mode,
        "retrained_at": datetime.now().isoformat(),
    }, model_path)
    print(f"  Nouveau modèle sauvegardé : {model_path}")

    log_entry = {
        "timestamp":      datetime.now().isoformat(),
        "new_label":      new_label,
        "n_new_samples":  len(new_samples),
        "source":         source,
        "mode":           args.mode,
        "f1_before":      metrics_before["f1_macro"],
        "f1_after":       metrics_after["f1_macro"],
        "delta_f1_macro": round(delta_f1, 4),
        "mcc_before":     metrics_before["mcc"],
        "mcc_after":      metrics_after["mcc"],
        "fit_time_s":     round(t_fit, 1),
        "class_names":    class_names_new,
    }
    save_retrain_log(log_entry)

    print("\nRé-entraînement terminé. Le modèle est prêt pour de nouvelles inférences.")
    print("Lancez inference.py pour utiliser le modèle mis à jour.")


def main():
    parser = argparse.ArgumentParser(
        description="Ré-entraînement continu du pipeline Cynergia"
    )
    parser.add_argument("--new-data", type=str,
                        help="Fichier .jsonl d'alertes UNKNOWN_THREAT "
                             "avec features brutes (sortie de inference.py)")
    parser.add_argument("--labeled-file", type=str,
                        help="Fichier JSON : [{\"features\":{...}, \"label\":\"...\"}]")
    parser.add_argument("--label", type=str,
                        help="Label à assigner à tous les nouveaux exemples")
    parser.add_argument("--mode", choices=["standalone", "hybrid"],
                        default="hybrid",
                        help="Mode du modèle Stage-2 (défaut: hybrid)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simuler le ré-entraînement sans sauvegarder")
    parser.add_argument("--info", action="store_true",
                        help="Afficher les infos du modèle actuel et l'historique")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
