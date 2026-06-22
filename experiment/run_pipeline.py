"""
run_pipeline.py — Lance le pipeline Cynergia complet sur un dataset donné.

Utilisation :
  # TON_IoT (dataset principal)
  python experiment/run_pipeline.py --dataset ton_iot

  # CIC-IDS2018 (dataset de validation croisée)
  python experiment/run_pipeline.py --dataset cic_ids2018

  # Étapes individuelles
  python experiment/run_pipeline.py --dataset ton_iot --stages build,ocsvm,composite

  # Ensemble avec Autoencoder
  python experiment/run_pipeline.py --dataset ton_iot --stages ensemble --vote majority --with-ae

Étapes disponibles (dans l'ordre) :
  build       → ton_00 / cic_00    — construit les .npy
  eda         → ton_01_eda         — EDA + figures classe
  ocsvm       → train_ocsvm_if_lof — Stage 1 : 3 détecteurs individuels
  composite   → train_composite    — Stage 1 : score composite α+β+γ
  autoencoder → train_autoencoder  — Stage 1 : 4e détecteur AE
  ensemble    → train_ensemble     — Stage 1 : vote parallèle
  xgb         → train_xgboost      — Stage 2 standalone + hybride
  stage2      → ton_02_stage2      — pipeline complet Stage1→Stage2
  drift       → detect_drift       — calibration + simulation MMD
  killchain   → detect_killchain   — détection séquences temporelles
  adaptive    → adaptive_threshold — simulation PI controller
  report      → generate_report    — PDF final
"""
import argparse
import os
import subprocess
import sys

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
MODELS  = os.path.join(HERE, "models")
STAGE1  = os.path.join(MODELS, "stage1")
STAGE2  = os.path.join(MODELS, "stage2_dl")
KCHAIN  = os.path.join(MODELS, "killchain")
DRIFT   = os.path.join(MODELS, "drift")


# ── Registre des étapes par dataset ─────────────────────────────────────────

STEPS = {
    "ton_iot": {
        "build":       [sys.executable, os.path.join(HERE, "ton_00_build_dataset.py")],
        "eda":         [sys.executable, os.path.join(HERE, "ton_01_eda_and_stage1.py")],
        "ocsvm":       [sys.executable, os.path.join(STAGE1, "train_ocsvm_if_lof.py")],
        "composite":   [sys.executable, os.path.join(STAGE1, "train_composite_stage1.py")],
        "autoencoder": [sys.executable, os.path.join(STAGE1, "train_autoencoder.py")],
        "ensemble":    [sys.executable, os.path.join(STAGE1, "train_ensemble_stage1.py")],
        "xgb":         [sys.executable, os.path.join(STAGE2, "train_xgboost.py"), "--mode", "hybrid"],
        "stage2":      [sys.executable, os.path.join(HERE,   "ton_02_stage2_and_stats.py")],
        "drift":       [sys.executable, os.path.join(DRIFT,  "detect_concept_drift.py"), "--calibrate"],
        "killchain":   [sys.executable, os.path.join(KCHAIN, "detect_kill_chain.py"),    "--demo"],
        "adaptive":    [sys.executable, os.path.join(STAGE1, "adaptive_threshold.py")],
        "report":      [sys.executable, os.path.join(HERE,   "generate_report_pdf.py")],
    },
    "cic_ids2018": {
        "build":       [sys.executable, os.path.join(HERE, "cic_00_build_dataset.py")],
        # Les étapes suivantes sont identiques — elles lisent X_train.npy etc.
        # depuis METRICS_DIR défini dans le config actif.
        # Pour CIC on ré-utilise les mêmes scripts avec --results-dir overridé via env var.
        "ocsvm":       [sys.executable, os.path.join(STAGE1, "train_ocsvm_if_lof.py")],
        "composite":   [sys.executable, os.path.join(STAGE1, "train_composite_stage1.py")],
        "autoencoder": [sys.executable, os.path.join(STAGE1, "train_autoencoder.py")],
        "ensemble":    [sys.executable, os.path.join(STAGE1, "train_ensemble_stage1.py")],
        "xgb":         [sys.executable, os.path.join(STAGE2, "train_xgboost.py"), "--mode", "hybrid"],
        "drift":       [sys.executable, os.path.join(DRIFT,  "detect_concept_drift.py"), "--calibrate"],
        "killchain":   [sys.executable, os.path.join(KCHAIN, "detect_kill_chain.py"),    "--demo"],
        "adaptive":    [sys.executable, os.path.join(STAGE1, "adaptive_threshold.py")],
    },
}

DEFAULT_ORDER = [
    "build", "eda", "ocsvm", "composite", "autoencoder",
    "ensemble", "xgb", "stage2", "drift", "killchain", "adaptive", "report"
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _env_for(dataset: str) -> dict:
    """Injecte CYNERGIA_DATASET pour que les scripts sachent quel config charger."""
    env = os.environ.copy()
    env["CYNERGIA_DATASET"] = dataset
    return env


def run_step(name: str, cmd: list, extra_args: list, env: dict, dry_run: bool):
    full = cmd + extra_args
    print(f"\n{'='*60}")
    print(f"  ETAPE : {name.upper()}")
    print(f"  CMD   : {' '.join(full)}")
    print(f"{'='*60}")
    if dry_run:
        print("  [DRY-RUN] Skipped.")
        return True
    result = subprocess.run(full, env=env)
    if result.returncode != 0:
        print(f"\n[ERROR] Etape '{name}' échouée (code {result.returncode})")
        return False
    return True


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Pipeline Cynergia — multi-dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument(
        "--dataset", choices=list(STEPS.keys()), default="ton_iot",
        help="Dataset à utiliser (défaut : ton_iot)"
    )
    ap.add_argument(
        "--stages", default=None,
        help="Étapes à exécuter (séparées par virgule). Défaut : toutes dans l'ordre."
    )
    ap.add_argument(
        "--vote", choices=["or", "majority", "and"], default="majority",
        help="Règle de vote pour l'ensemble (défaut : majority)"
    )
    ap.add_argument(
        "--with-ae", action="store_true",
        help="Inclure l'Autoencoder dans l'ensemble"
    )
    ap.add_argument(
        "--no-retrain", action="store_true",
        help="Passer --no-retrain aux scripts de Stage 1"
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Affiche les commandes sans les exécuter"
    )
    ap.add_argument(
        "--stop-on-error", action="store_true", default=True,
        help="Arrête le pipeline si une étape échoue (défaut : True)"
    )
    args = ap.parse_args()

    dataset_steps = STEPS[args.dataset]

    # Résoudre les étapes à exécuter
    if args.stages:
        requested = [s.strip() for s in args.stages.split(",")]
        unknown   = [s for s in requested if s not in dataset_steps]
        if unknown:
            print(f"[ERROR] Étapes inconnues pour {args.dataset}: {unknown}")
            print(f"  Disponibles : {list(dataset_steps.keys())}")
            sys.exit(1)
        order = requested
    else:
        order = [s for s in DEFAULT_ORDER if s in dataset_steps]

    env = _env_for(args.dataset)
    print(f"\n[Cynergia] Dataset  : {args.dataset}")
    print(f"[Cynergia] Étapes   : {order}")
    print(f"[Cynergia] Vote     : {args.vote}")
    print(f"[Cynergia] Avec AE  : {args.with_ae}")

    results = {}
    for step in order:
        cmd = dataset_steps[step]

        # Ajouter des arguments supplémentaires selon l'étape
        extra = []
        if step == "ensemble":
            extra += ["--vote", args.vote]
            if args.with_ae:
                extra += ["--with-ae"]
            if args.no_retrain:
                extra += ["--no-retrain"]
        elif step in ("composite", "ocsvm", "autoencoder"):
            if args.no_retrain:
                extra += ["--no-retrain"]
        elif step == "stage2":
            extra += ["--use-ensemble", args.vote]

        ok = run_step(step, cmd, extra, env, args.dry_run)
        results[step] = "OK" if ok else "FAIL"
        if not ok and args.stop_on_error:
            break

    # Résumé final
    print(f"\n{'='*60}")
    print("  RÉSUMÉ DU PIPELINE")
    print(f"{'='*60}")
    for step, status in results.items():
        icon = "[OK]  " if status == "OK" else "[FAIL]"
        print(f"  {icon} {step}")
    skipped = [s for s in order if s not in results]
    for step in skipped:
        print(f"  [SKIP] {step}")

    all_ok = all(v == "OK" for v in results.values())
    print(f"\n{'Pipeline COMPLET' if all_ok else 'Pipeline INCOMPLET'} — {args.dataset}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
