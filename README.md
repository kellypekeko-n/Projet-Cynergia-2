# CYNERGIA — Framework Hybride ICS/IIoT pour la Détection d'Attaques Furtives

**Cybersécurité | Énergie | Intelligence Artificielle**

> Intégrer un moteur d'intelligence artificielle au sein d'un jumeau numérique pour surveiller passivement les réseaux industriels d'infrastructures critiques sans perturber la production.

**Responsable :** Kelly Noelle Mapoue Pekeko — Baccalauréat en informatique, UQAC  
**Superviseur :** Kevin Bouchard, Professeur — UQAC  
**Collaboration :** Hugo Bourreau — Jumeaux numériques & Sécurité IoT  
**Point de départ scientifique :** Hoummady & Jaafar (2026) — Benchmark ML vs DL, CMC

---

## Objectif

Développer un framework hybride à deux étages pour détecter les attaques furtives dans les systèmes industriels (ICS/IIoT), là où les IDS classiques (Snort, Suricata) échouent structurellement, et mapper chaque détection à une technique MITRE ATT&CK for ICS.

**Problème fondamental résolu :** Hoummady & Jaafar (2026) mesurent un F1 = 3 % sur Bruteforce malgré 99 % d'accuracy globale — preuve que l'accuracy masque les attaques furtives minoritaires.

---

## Architecture du Framework

```
Trafic réseau ICS/IIoT (TON_IoT — 536K flux, 10 classes)
        |
        v
╔══════════════════════════════════════════════════════════╗
║              STAGE 1 — Filtre d'anomalies                ║
║  Entraîné sur trafic NORMAL uniquement                   ║
║                                                          ║
║  Mode individuel  : OCSVM (meilleur, AUC=0.977)         ║
║                     IF | LOF                             ║
║                                                          ║
║  Mode composite   : α·OCSVM + β·IF + γ·LOF              ║
║  (poids optimisés par differential_evolution sur val)    ║
║                                                          ║
║  Mode ensemble    : OCSVM | IF | LOF | AE (optionnel)   ║
║  (décisions binaires spécialisées + vote OR/MAJORITY/AND)║
║                                                          ║
║  Rôles spécialisés :                                     ║
║    OCSVM → frontière globale (FPR ≤ 3%)                 ║
║    IF    → outlier structurel (recall ≥ 95%)             ║
║    LOF   → isolement local furtif (stealthy recall ≥90%) ║
║    AE    → reconstruction neuronale (MSE, non-linéaire)  ║
║                                                          ║
║  Seuil adaptatif θ (contrôleur PI, FPR ≤ budget)        ║
║  Objectifs : Recall ≥ 90% | FPR ≤ 15%                  ║
╚══════════════════════════════════════════════════════════╝
        |
        | Flux suspects uniquement
        v
╔══════════════════════════════════════════════════════════╗
║              STAGE 2 — Classification supervisée         ║
║                                                          ║
║  ML : XGBoost (F1=0.88) | LightGBM | Random Forest      ║
║  DL : CNN-LSTM (F1=0.91) | PatchTST | Transformer       ║
║                                                          ║
║  Open-set : classe UNKNOWN (inter-class mixup            ║
║             + Gaussian tails) → zero-day detection       ║
║                                                          ║
║  Confiance < 0.70 → UNKNOWN_THREAT + top-3 candidats    ║
╚══════════════════════════════════════════════════════════╝
        |
        v
╔══════════════════════════════════════════════════════════╗
║         MAPPING & EXPLICABILITÉ                          ║
║  MITRE ATT&CK for ICS (T0807, T0826, T0830, T0836...)   ║
║  SHAP TreeExplainer — beeswarm + waterfall par classe    ║
╚══════════════════════════════════════════════════════════╝
        |
        v
╔══════════════════════════════════════════════════════════╗
║         SURVEILLANCE CONTINUE                            ║
║  Kill Chain Detection : T0840→T0830→T0826 par IP         ║
║  Concept Drift MMD    : alerte si distribution dérive    ║
║  Ré-entraînement      : nouvelles attaques → retrain.py  ║
║  Stream temps réel    : stdin / file tail / JSON / CSV   ║
╚══════════════════════════════════════════════════════════╝
```

**Trois modes d'opération :**
- **Standalone :** Stage 2 seul sur toutes les données
- **Hybride :** Stage 1 filtre → Stage 2 sur flux suspects enrichis
- **Stream :** Surveillance temps réel avec kill chain et drift monitoring

---

## Résultats Obtenus (Dataset TON_IoT — 107K flux test)

### Stage 1 — Comparaison des détecteurs individuels

| Modèle | AUC-ROC | Recall | FPR | Enrichissement furtif |
|---|---|---|---|---|
| **OCSVM (RBF)** | **0.977** | **90.0%** | **1.49%** | 1.22× |
| Isolation Forest | 0.942 | ~90% | ~5% | ~1.1× |
| LOF | 0.931 | ~90% | ~6% | ~1.0× |
| **Composite α·OCSVM+β·IF+γ·LOF** | À mesurer | ≥90% | ≤15% | **objectif ≥5×** |
| **Ensemble MAJORITY (3 détect.)** | À mesurer | ≥90% | ≤15% | **objectif ≥5×** |
| **Ensemble MAJORITY + AE (4 détect.)** | À mesurer | ≥90% | ≤15% | **objectif ≥5×** |

### Stage 2 — Comparaison des modèles

| Modèle | Mode | F1-macro | Backdoor Recall | MITM Recall |
|---|---|---|---|---|
| **CNN-LSTM** | standalone | **0.9105** | — | — |
| XGBoost | standalone | 0.8783 | 0.360 | 0.588 |
| LightGBM | standalone | ~0.87 | — | — |
| Random Forest | standalone | ~0.71 | — | — |
| XGBoost | hybride | 0.7219 | **0.540** (+50%) | 0.521 |
| PatchTST | standalone | *à mesurer* | — | — |

### Hypothèses

| Hypothèse | Énoncé | Verdict actuel |
|---|---|---|
| **H1** | Pipeline hybride > monolithique sur F1-macro furtif | **REJETÉ** (global), **PARTIEL** (backdoor +50%) |
| **H2** | Enrichissement Stage 1 ≥ 5× | **REJETÉ** (OCSVM seul : 1.22×) — **à re-tester avec composite/ensemble** |
| **H3** | CNN-LSTM > XGBoost sur signatures temporelles | **VALIDÉ** (0.9105 vs 0.8783) |

**Note scientifique :** H2 rejeté avec OCSVM seul devient testable avec le composite (α·OCSVM+β·IF+γ·LOF) et l'ensemble spécialisé (4 détecteurs). Ces innovations sont précisément conçues pour adresser ce rejet.

### Validation statistique (10 seeds)
- Friedman : chi² calculé | p < 0.05 — significatif
- Wilcoxon + Bonferroni : hybride vs standalone
- Bootstrap CI (n=1000) sur F1-macro par modèle

---

## Structure du Projet

```
Porjet-cynergia/
├── experiment/
│   ├── run_pipeline.py                # ★ Orchestrateur multi-dataset (--dataset ton_iot|cic_ids2018)
│   │
│   ├── ton_iot_config.py              # Configuration TON_IoT (classes, features, seeds, MITRE ICS)
│   ├── ton_00_build_dataset.py        # Builder TON_IoT — chargement 23 CSV, 536K flux → .npy
│   ├── ton_01_eda_and_stage1.py       # EDA + Stage 1 individuel (OCSVM, LOF, IF)
│   ├── ton_02_stage2_and_stats.py     # Stage 2 ML + tests statistiques (--use-composite, --use-ensemble)
│   ├── ton_03_recover_and_latex.py    # Génération tables LaTeX pour article
│   │
│   ├── cic_ids2018_config.py          # ★ Configuration CIC-IDS2018 (15 classes, 58 features CICFlowMeter)
│   ├── cic_00_build_dataset.py        # ★ Builder CIC-IDS2018 — normalisation labels, feature engineering
│   │
│   ├── inference.py                   # Pipeline inférence complet (stream, UNKNOWN_THREAT, SHAP)
│   ├── retrain.py                     # Ré-entraînement continu (nouvelles attaques labélisées)
│   ├── evaluate.py                    # Évaluation comparative tous modèles
│   ├── generate_report_pdf.py         # Rapport PDF automatique (reportlab)
│   │
│   ├── models/
│   │   ├── stage1/
│   │   │   ├── train_ocsvm.py                 # OCSVM individuel (meilleur Stage 1, AUC=0.977)
│   │   │   ├── train_isolation_forest.py       # IF individuel
│   │   │   ├── train_composite_stage1.py       # ★ Score composite α·OCSVM+β·IF+γ·LOF
│   │   │   │                                   #   differential_evolution → max enrichissement
│   │   │   ├── train_autoencoder.py            # ★ Autoencoder (4e détecteur, MSE reconstruction)
│   │   │   │                                   #   PyTorch (fallback PCA si pas de torch)
│   │   │   ├── train_ensemble_stage1.py        # ★ Ensemble parallèle 3 ou 4 détecteurs
│   │   │   │                                   #   Rôles spécialisés + vote OR/MAJORITY/AND
│   │   │   └── adaptive_threshold.py           # ★ Seuil adaptatif (contrôleur PI sur FPR)
│   │   │
│   │   ├── stage2_ml/
│   │   │   ├── train_xgboost.py               # XGBoost + SHAP TreeExplainer (3 types de figures)
│   │   │   ├── train_xgboost_openset.py        # Open-set recognition (classe UNKNOWN synthétique)
│   │   │   └── train_lightgbm.py              # LightGBM standalone
│   │   │
│   │   ├── stage2_dl/
│   │   │   ├── train_cnn_lstm.py              # CNN-LSTM (F1=0.9105) — MEILLEUR modèle global
│   │   │   └── train_transformer.py           # PatchTST + Vanilla Transformer
│   │   │
│   │   ├── killchain/
│   │   │   └── detect_kill_chain.py           # ★ Kill chain T0840→T0830→T0826 par IP
│   │   │                                      #   4 patterns MITRE, fenêtre 120–600s
│   │   └── drift/
│   │       └── detect_concept_drift.py        # ★ Dérive MMD kernel RBF (bootstrap α=0.01)
│   │
│   └── results/
│       ├── figures/                   # 50+ figures PNG/PDF
│       │   ├── fig01_class_distribution.png/pdf
│       │   ├── fig02_class_imbalance.png/pdf
│       │   ├── fig03_feature_correlation.png/pdf
│       │   ├── fig04_stage1_roc.png/pdf
│       │   ├── fig05_stage1_pr.png/pdf
│       │   ├── fig06_stage1_enrichment.png/pdf
│       │   ├── fig07_threshold_calibration.png/pdf
│       │   ├── fig08_cm_*.png/pdf              # Matrices de confusion par modèle
│       │   ├── fig09_stage2_roc_standalone.png
│       │   ├── fig10_pr_{backdoor,mitm,ransomware,scanning}.png
│       │   ├── fig11_ablation.png
│       │   ├── fig12_*.png                    # Comparaison enrichissement + modèles
│       │   ├── fig13_*.png                    # Backdoor key finding
│       │   ├── fig14_bootstrap_ci.png
│       │   ├── cnn_lstm_loss.png
│       │   ├── xgb_{standalone,hybrid}_{cm,roc}.png
│       │   ├── fig_composite_s1_{roc,enrichment,weights}.png  [à générer]
│       │   ├── fig_ensemble_s1_{votes,stealthy}.png           [à générer]
│       │   ├── fig_ae_stage1.png                              [à générer]
│       │   ├── xgb_shap_{standalone,hybrid}_summary.png       [à générer]
│       │   ├── xgb_shap_{standalone,hybrid}_stealthy.png      [à générer]
│       │   ├── fig_drift_simulation.png                       [à générer]
│       │   └── fig_adaptive_threshold.png                     [à générer]
│       │
│       ├── metrics/                   # JSON avec toutes les métriques
│       │   ├── dataset_meta.json               # Features, classes, distribution
│       │   ├── label_classes.json              # Mapping index → nom classe
│       │   ├── eda_and_stage1.json             # EDA + résultats Stage 1 individuel
│       │   ├── stage2_and_stats.json           # Stage 2 ML + tests Friedman/Wilcoxon
│       │   ├── xgboost_standalone_metrics.json # F1=0.8783, backdoor recall=0.360
│       │   ├── xgboost_hybrid_metrics.json     # F1=0.7219, backdoor recall=0.540 (+50%)
│       │   ├── cnn_lstm_metrics.json           # F1=0.9105, AUC-ROC=0.9977
│       │   ├── composite_s1_metrics.json       [à générer]
│       │   ├── ensemble_stage1_metrics.json    [à générer]
│       │   ├── autoencoder_stage1_metrics.json [à générer]
│       │   └── drift_config.json               [à générer après --calibrate]
│       │
│       └── saved_models/
│           ├── xgboost_standalone.pkl
│           ├── xgboost_hybrid.pkl
│           ├── cnn_lstm.pt
│           └── autoencoder_stage1.pt           [à générer]
│
├── TON_IOT_Datasets/                  # Dataset brut TON_IoT (non versionné — ~2GB)
└── CIC_IDS2018_Datasets/             # Dataset brut CIC-IDS2018 (non versionné — ~7GB, optionnel)
    └── Processed Traffic Data for ML Algorithms/
        └── *.csv                      # Fichiers par jour (Friday-02-03-2018_...)
```
★ = fichiers créés dans les dernières sessions de développement

**Note sur `results_cic/` :** Quand tu lances `--dataset cic_ids2018`, tous les résultats (`.npy`, `.json`, figures) sont écrits dans `experiment/results_cic/` au lieu de `experiment/results/`, ce qui évite tout écrasement des résultats TON_IoT.

---

## Guide d'Exécution Complet — Étape par Étape

> Toujours exécuter depuis la racine du projet :
> ```bash
> cd C:\Users\Kelly Pekeko\Downloads\Porjet-cynergia
> ```

---

## DATASET 1 — TON_IoT (dataset principal)

### Prérequis — Installation

```bash
pip install scikit-learn xgboost lightgbm torch scipy matplotlib seaborn reportlab shap
```

### Bloc 1 — Construction des données (déjà fait)

```bash
# Construit X_train.npy, X_val.npy, X_test.npy, y_*.npy, scaler.pkl
# → experiment/results/metrics/
python experiment/ton_00_build_dataset.py
```

Produit : 536K flux, 10 classes, split 60/20/20, 16 features Zeek normalisées.

### Bloc 2 — Stage 1 individuel (déjà fait)

```bash
# OCSVM (AUC=0.977), Isolation Forest, LOF
# → eda_and_stage1.json + fig01–07
python experiment/ton_01_eda_and_stage1.py
```

Résultats obtenus : OCSVM recall=90%, FPR=1.49%, enrichissement furtif=1.22×.

### Bloc 3 — Stage 1 innovations (~1–2h)

```bash
# 3a. Score composite α·OCSVM + β·IF + γ·LOF
#     Optimise directement l'enrichissement furtif via differential_evolution
#     → composite_s1_metrics.json + fig_composite_s1_roc.png + 2 autres figures
python experiment/models/stage1/train_composite_stage1.py

# 3b. Autoencoder — 4e détecteur (réseau de reconstruction MSE)
#     Architecture : features → 64 → 32 → 16 → 8 → 16 → 32 → 64 → features
#     → ae_s1_scores.npz + autoencoder_stage1_metrics.json + fig_ae_stage1.png
python experiment/models/stage1/train_autoencoder.py

# 3c. Ensemble 4 détecteurs avec vote MAJORITY (≥3/4)
#     Rôles : OCSVM=précision, IF=recall, LOF=furtif, AE=non-linéaire
#     → s1_ensemble_flag_*.npy + ensemble_stage1_metrics.json + 2 figures
python experiment/models/stage1/train_ensemble_stage1.py --vote majority --no-retrain --with-ae
```

### Bloc 4 — Stage 2 ML (~30–60 min)

```bash
# 4a. Stage 2 avec scores composites Stage 1 → verdict H1/H2 mis à jour
#     → stage2_and_stats.json mis à jour
python experiment/ton_02_stage2_and_stats.py --use-composite

# 4b. Stage 2 avec scores ensemble Stage 1 → comparaison finale
python experiment/ton_02_stage2_and_stats.py --use-ensemble majority

# 4c. SHAP figures XGBoost (beeswarm + bar par classe + waterfall)
#     → xgb_shap_standalone_summary.png + xgb_shap_hybrid_summary.png + 4 autres
python experiment/models/stage2_ml/train_xgboost.py --mode standalone
python experiment/models/stage2_ml/train_xgboost.py --mode hybrid

# 4d. Open-set recognition — classe UNKNOWN synthétique (zero-day detection)
#     → xgboost_openset_hybrid.pkl + métriques open-set
python experiment/models/stage2_ml/train_xgboost_openset.py --mode hybrid
```

### Bloc 5 — Surveillance continue (~20 min)

```bash
# 5a. Concept drift — calibration sur données normales
#     → drift_config.json (seuil MMD bootstrap α=0.01)
python experiment/models/drift/detect_concept_drift.py --calibrate

# 5b. Concept drift — simulation avec injection de dérive artificielle
#     → fig_drift_simulation.png
python experiment/models/drift/detect_concept_drift.py --simulate

# 5c. Kill chain — scénario T0840→T0830→T0826 par source IP
#     → demo_alerts.jsonl + killchain_demo_metrics.json
python experiment/models/killchain/detect_kill_chain.py --demo \
  --metrics experiment/results/metrics/killchain_demo_metrics.json

# 5d. Seuil adaptatif — simulation contrôleur PI (3 phases)
#     → fig_adaptive_threshold.png
python experiment/models/stage1/adaptive_threshold.py
```

### Bloc 6 — DL supplémentaire (~1–6h selon CPU/GPU)

```bash
# PatchTST — comparaison H3 : CNN-LSTM (F1=0.9105) vs Transformer
# ~1h avec GPU NVIDIA, ~6h sans GPU
# → transformer_metrics.json + figures
python experiment/models/stage2_dl/train_transformer.py --model patchtst --window_size 60
```

### Bloc 7 — Rapport final (~30 min — après tout le reste)

```bash
# Rapport PDF complet (toutes métriques + toutes figures + verdicts H1/H2/H3)
python experiment/generate_report_pdf.py

# Tables LaTeX pour article scientifique
python experiment/ton_03_recover_and_latex.py
```

### Dépendances entre blocs

```
[Bloc 2] ton_01 (OCSVM/IF/LOF scores)
    ↓
[Bloc 3a] train_composite_stage1.py
[Bloc 3b] train_autoencoder.py
    ↓
[Bloc 3c] train_ensemble_stage1.py --with-ae
    ↓
[Bloc 4a] ton_02 --use-composite
[Bloc 4b] ton_02 --use-ensemble
[Bloc 4c] train_xgboost.py × 2         (indépendant, peut tourner en parallèle)
[Bloc 4d] train_xgboost_openset.py     (indépendant)
    ↓
[Bloc 5]  drift + killchain + adaptive (indépendants entre eux)
[Bloc 6]  train_transformer.py         (complètement indépendant)
    ↓
[Bloc 7]  generate_report_pdf.py  ← bloqué jusqu'à ce que 3–5 soient faits
          ton_03_recover_and_latex.py
```

---

## DATASET 2 — CIC-IDS2018 (validation croisée)

### Étape 0 — Télécharger le dataset

1. Aller sur `https://www.unb.ca/cic/datasets/ids-2018.html`
2. Télécharger **"Processed Traffic Data for ML Algorithms"** (~7 GB)
3. Placer les CSV dans :
   ```
   C:\Users\Kelly Pekeko\Downloads\CIC_IDS2018_Datasets\
     Processed Traffic Data for ML Algorithms\
       Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
       Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv
       ...
   ```
4. Mettre à jour le chemin dans [cic_ids2018_config.py](experiment/cic_ids2018_config.py) ligne 20 :
   ```python
   CIC_DIR = r"C:\Users\Kelly Pekeko\Downloads\CIC_IDS2018_Datasets\Processed Traffic Data for ML Algorithms"
   ```

### Étape 1 — Construire les .npy

```bash
# Construit les .npy dans experiment/results_cic/metrics/
# ~10 min pour 7GB de CSV
python experiment/cic_00_build_dataset.py

# Test rapide avec 200K lignes seulement (~2 min)
python experiment/cic_00_build_dataset.py --max-rows 200000
```

Produit : X_train.npy, X_val.npy, X_test.npy, y_*.npy, scaler.pkl — même format que TON_IoT.

### Étape 2 — Pipeline complet CIC (même ordre que TON_IoT)

```bash
# Option A — Orchestrateur (recommandé, lance tout dans l'ordre)
python experiment/run_pipeline.py --dataset cic_ids2018

# Option B — Étapes individuelles (même scripts, résultats dans results_cic/)
python experiment/models/stage1/train_ocsvm_if_lof.py
python experiment/models/stage1/train_composite_stage1.py
python experiment/models/stage1/train_autoencoder.py
python experiment/models/stage1/train_ensemble_stage1.py --vote majority --no-retrain --with-ae
python experiment/models/stage2_ml/train_xgboost.py --mode standalone
python experiment/models/stage2_ml/train_xgboost.py --mode hybrid
python experiment/models/drift/detect_concept_drift.py --calibrate
python experiment/models/drift/detect_concept_drift.py --simulate
```

> Les résultats CIC sont isolés dans `experiment/results_cic/` — aucun écrasement des résultats TON_IoT dans `experiment/results/`.

### Étape 3 — Comparer les deux datasets

Après les deux exécutions, les fichiers de métriques sont :

```
experiment/results/metrics/       ← TON_IoT
experiment/results_cic/metrics/   ← CIC-IDS2018
```

**Tableau de comparaison à remplir au fur et à mesure :**

| Métrique | TON_IoT | CIC-IDS2018 |
|---|---|---|
| OCSVM AUC-ROC | 0.977 | à mesurer |
| OCSVM enrichissement furtif | 1.22× | à mesurer |
| Composite enrichissement | à mesurer | à mesurer |
| Ensemble MAJORITY enrichissement | 1.21× | à mesurer |
| XGBoost standalone F1-macro | 0.8783 | à mesurer |
| XGBoost hybrid F1-macro | 0.7219 | à mesurer |
| CNN-LSTM F1-macro | 0.9105 | à mesurer |
| H1 (hybride > standalone sur furtif) | PARTIEL | à mesurer |
| H2 (enrichissement Stage 1 ≥ 5×) | REJETÉ (composite en cours) | à mesurer |
| H3 (CNN-LSTM > XGBoost) | VALIDÉ | à mesurer |

### Ajouter un nouveau dataset (SWaT, BATADAL…)

Créer uniquement **2 fichiers** — tout le reste fonctionne sans modification :

1. **`experiment/swat_config.py`** — copier `ton_iot_config.py`, adapter `SWAT_DIR`, `ALL_CLASSES`, `NUMERIC_FEATURES`, `SAMPLE_TARGET`
2. **`experiment/swat_00_build_dataset.py`** — copier `cic_00_build_dataset.py`, adapter la lecture des CSV et la normalisation des labels

```bash
# Puis lancer directement
python experiment/run_pipeline.py --dataset swat
```

---

## Support Multi-Dataset

### Architecture modulaire

Le pipeline Cynergia est conçu pour fonctionner sur **n'importe quel dataset réseau** en séparant clairement la couche dataset de la couche modèles :

```
Couche dataset (à adapter par dataset)        Couche modèles (100% réutilisable)
────────────────────────────────────          ─────────────────────────────────
ton_iot_config.py     ─→  .npy  ──────────→  models/stage1/train_ocsvm_if_lof.py
ton_00_build_dataset.py  ──────/             models/stage1/train_composite_stage1.py
                                              models/stage1/train_autoencoder.py
cic_ids2018_config.py  ─→  .npy  ─────────→ models/stage1/train_ensemble_stage1.py
cic_00_build_dataset.py  ──────/             models/stage2_ml/train_xgboost.py
                                              models/stage2_dl/train_cnn_lstm.py
(swat_config.py)       ─→  .npy  ─────────→ ton_02_stage2_and_stats.py
(swat_00_build.py)     ──────/              models/drift/detect_concept_drift.py
                                             models/killchain/detect_kill_chain.py
```

**Règle :** Les scripts `ton_01+` et `models/*` ne lisent jamais de CSV directement — ils chargent uniquement des `.npy` standardisés (`X_train.npy`, `y_train.npy`, etc.) depuis `results/metrics/`. C'est pourquoi ils fonctionnent sans modification sur n'importe quel dataset.

### Datasets supportés

| Dataset | Status | Script de build | Classes | Particularités |
|---|---|---|---|---|
| **TON_IoT** | Complet | `ton_00_build_dataset.py` | 10 (ICS/IoT) | Features Zeek, MITRE ICS |
| **CIC-IDS2018** | Config + builder | `cic_00_build_dataset.py` | 15 (réseau) | Features CICFlowMeter, 58 features |
| **SWaT** | Config à créer | `swat_00_build_dataset.py` | 2 (normal/attaque) | Données capteurs physiques, structure très différente |
| **BATADAL** | Config à créer | `batadal_00_build_dataset.py` | binaire | Très petit (6M points), timestamps horaires |

### Lancer le pipeline sur un dataset

```bash
# ── Méthode 1 : run_pipeline.py (recommandé) ──────────────────────────────

# TON_IoT (dataset principal du projet)
python experiment/run_pipeline.py --dataset ton_iot

# CIC-IDS2018 (validation croisée)
python experiment/run_pipeline.py --dataset cic_ids2018

# Étapes spécifiques seulement
python experiment/run_pipeline.py --dataset cic_ids2018 --stages build,ocsvm,composite

# Ensemble avec AE sur CIC-IDS2018
python experiment/run_pipeline.py --dataset cic_ids2018 \
    --stages build,ocsvm,autoencoder,ensemble \
    --vote majority --with-ae

# ── Méthode 2 : scripts individuels ───────────────────────────────────────

# Étape 1 : construire les .npy pour CIC-IDS2018
python experiment/cic_00_build_dataset.py

# Étape 2 : les scripts models/* lisent automatiquement depuis results_cic/metrics/
# (set CYNERGIA_DATASET=cic_ids2018 pour rediriger les chemins)
CYNERGIA_DATASET=cic_ids2018 python experiment/models/stage1/train_ocsvm_if_lof.py
```

### Ajouter un nouveau dataset

Pour ajouter SWaT ou un autre dataset, il suffit de créer **2 fichiers** :

1. **`experiment/swat_config.py`** — copier `ton_iot_config.py` et adapter :
   - `SWAT_DIR` : chemin vers les CSV bruts
   - `ALL_CLASSES`, `NORMAL_CLASS`, `STEALTHY_CLASSES`
   - `NUMERIC_FEATURES` : colonnes numériques du dataset
   - `SAMPLE_TARGET` : nb de flux à garder par classe

2. **`experiment/swat_00_build_dataset.py`** — copier `cic_00_build_dataset.py` et adapter :
   - Logique de lecture des CSV (noms de colonnes, format de la colonne label)
   - Feature engineering spécifique (SWaT a des capteurs physiques, pas des flow statistics)
   - Normalisation des noms de classes

Ensuite `run_pipeline.py --dataset swat` fonctionnera **sans aucune modification** aux scripts Stage 1 / Stage 2.

---

## Fonctionnalités Implémentées

### Pipeline de base (COMPLET)
- [x] Chargement et prétraitement TON_IoT (536K flux, 10 classes, split chronologique)
- [x] Stage 1 individuel : OCSVM (AUC=0.977), LOF, Isolation Forest
- [x] Stage 2 ML : XGBoost standalone (F1=0.8783) + hybride (F1=0.7219), LightGBM, RF
- [x] Stage 2 DL : CNN-LSTM (F1=0.9105, AUC-ROC=0.9977)
- [x] Tests statistiques : Friedman, Wilcoxon, Bonferroni, Bootstrap CI (10 seeds)
- [x] Mapping MITRE ATT&CK for ICS (T0807, T0826, T0830, T0836, T0840)
- [x] 30+ figures PNG/PDF générées (fig01–fig14 + variations)
- [x] SHAP TreeExplainer dans train_xgboost.py (beeswarm + bar par classe + waterfall)

### Stage 1 — Innovations (scripts complets, à exécuter)
- [x] **Score composite** `α·OCSVM + β·IF + γ·LOF` — poids optimisés par `differential_evolution`
  - Maximise enrichissement furtif sur validation sous recall ≥ 90% et FPR ≤ 15%
  - Adresse directement le rejet de H2 (1.22× → objectif ≥ 5×)
- [x] **Autoencoder** (4e détecteur) — erreur de reconstruction MSE sur trafic normal
  - Architecture : features → 64 → 32 → 16 → **8** (bottleneck) → 16 → 32 → 64 → features
  - Plus puissant que OCSVM/LOF car capture des patterns non-linéaires
  - PyTorch avec early stopping + BatchNorm (fallback PCA si pas de torch)
- [x] **Ensemble parallèle spécialisé** (3 ou 4 détecteurs)
  - Chaque détecteur a un rôle et un seuil calibré différemment
  - Vote configurable : OR (≥1/N), MAJORITY (≥N/2+1), AND (N/N)
  - Une attaque doit tromper ≥ vote_threshold détecteurs pour passer
- [x] **Seuil adaptatif θ** — contrôleur PI, mise à jour par feedback analyste ou autonome

### Stage 2 — Détection zero-day (COMPLET)
- [x] Seuil de confiance 0.70 → `UNKNOWN_THREAT` avec top-3 candidats
- [x] Open-set recognition : classe UNKNOWN synthétique (inter-class mixup + Gaussian tails)
- [x] Leave-one-class-out pour valider la détection d'attaques inconnues

### Surveillance continue (COMPLET)
- [x] **Kill chain detection** — 4 patterns MITRE (T0840→T0830→T0826...) par source IP
  - `Full_ICS_Campaign` (CRITICAL, 10min)
  - `Reconnaissance_Pivot` (HIGH, 5min)
  - `Ransomware_Deployment` (CRITICAL, 3min)
  - `DDoS_Amplification` (HIGH, 2min)
- [x] **Concept Drift MMD** — seuil bootstrap (α=0.01), fenêtre 500 flux
- [x] **Mode stream temps réel** — stdin / file tail / JSON / CSV
- [x] **Ré-entraînement continu** — nouvelles attaques labélisées → `retrain.py`
- [x] Alertes `.jsonl` + statistiques flux/s, taux attaques, taux UNKNOWN

### Multi-dataset (COMPLET)
- [x] **Architecture modulaire** — couche dataset séparée de couche modèles
- [x] **Config + builder CIC-IDS2018** — 15 classes, 58 features CICFlowMeter, normalisation labels
- [x] **Orchestrateur `run_pipeline.py`** — `--dataset ton_iot|cic_ids2018`, `--stages build,ocsvm,...`
- [x] Injection `CYNERGIA_DATASET` env var pour rediriger les chemins résultats par dataset
- [ ] Builder SWaT (capteurs physiques) — structure trop différente des flow statistics
- [ ] Builder BATADAL (très petit, timestamped)

### Non implémenté (optionnel)
- [ ] Exécution CIC-IDS2018 complète (nécessite de télécharger le dataset ~7GB)
- [ ] Validation sur dataset SWaT (accès sur demande iTrust/NUS, délai 1–4 semaines)
- [ ] PatchTST exécution complète (`python models/stage2_dl/train_transformer.py --model patchtst`)
- [ ] Federated learning (perspective doctorat)

---

## Innovations Supplémentaires — Détail Scientifique

### 1. Score composite Stage 1
**Fichier :** `experiment/models/stage1/train_composite_stage1.py`

Combine les scores des 3 détecteurs classiques avec des poids appris automatiquement :
```
score_composite = α·score_OCSVM + β·score_IF + γ·score_LOF
```
Optimisation par `scipy.optimize.differential_evolution` sur l'ensemble de validation, en minimisant `-enrichissement_furtif` sous contrainte `recall ≥ 0.90` et `FPR ≤ 0.15`. Adresse directement le rejet de H2.

### 2. Autoencoder comme 4e détecteur
**Fichier :** `experiment/models/stage1/train_autoencoder.py`

Réseau de neurones entraîné à **reconstruire** le trafic normal. Une erreur de reconstruction élevée signifie que le flux est difficile à reconstruire = anormal. Architecture symétrique : `features → 64 → 32 → 16 → 8 → 16 → 32 → 64 → features`. Plus puissant que OCSVM/IF/LOF car il capture des patterns non-linéaires complexes entre features.

### 3. Ensemble parallèle spécialisé (3 ou 4 détecteurs)
**Fichier :** `experiment/models/stage1/train_ensemble_stage1.py`

Chaque détecteur a un **rôle précis** et un **seuil calibré pour ce rôle** :

| Détecteur | Rôle | Calibration |
|---|---|---|
| OCSVM | Frontière globale — précision | FPR ≤ 3% |
| IF | Outlier structurel — recall max | Recall ≥ 95% |
| LOF | Isolement local — spécialiste furtif | Stealthy recall ≥ 90% |
| AE | Reconstruction neuronale — non-linéaire | FPR ≤ 3% (même que OCSVM) |

Vote : `OR (≥1/N)` → recall max | `MAJORITY (≥N/2+1)` → équilibre | `AND (N/N)` → précision max

### 4. Kill Chain Detection
**Fichier :** `experiment/models/killchain/detect_kill_chain.py`

Détecte des campagnes multi-étapes par source IP en analysant les séquences temporelles d'alertes. 4 patterns MITRE ATT&CK for ICS définis avec fenêtres temporelles calibrées.

### 5. Concept Drift MMD
**Fichier :** `experiment/models/drift/detect_concept_drift.py`

Maximum Mean Discrepancy avec kernel RBF pour détecter quand la distribution des flux dérive de la baseline (nouveau équipement ICS, changement de protocole). Seuil calibré par permutation bootstrap (α=0.01).

### 6. Seuil adaptatif θ (contrôleur PI)
**Fichier :** `experiment/models/stage1/adaptive_threshold.py`

Ajuste automatiquement le seuil d'anomalie en production pour maintenir FPR ≤ budget, en utilisant le feedback de l'analyste (TP/FP) ou les flux confirmés normaux par Stage 2.

---

## Références

- Hoummady, S. & Jaafar, H. (2026). Benchmark ML vs DL on CICIoTDataset 2023. *CMC* — **point de départ**
- Ruff, L. et al. (2018). Deep One-Class Classification (Deep SVDD). *ICML* — inspiration AE/SVDD
- Gretton, A. et al. (2012). A Kernel Two-Sample Test (MMD). *JMLR 13*
- Wang, X. et al. (2021). ResNet8 + Transfer Learning for ICS attack detection. *Tsinghua Science & Technology*
- Zhang, Y. et al. (2023). TGAD: Transformer + GAN for ICS anomaly detection. *IEEE CCC*
- Al-Amri, R. et al. (2023). Two-stage IDS for Industrial Control Systems. *IEEE*
- MITRE ATT&CK for ICS — https://attack.mitre.org/matrices/ics/

---

*Projet Cynergia — UQAC 2026 — Baccalauréat en informatique*
# Projet-Cynergia-2
