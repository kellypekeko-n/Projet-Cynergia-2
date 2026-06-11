# Guide d'exécution complet
## Framework Hybride ICS/IIoT — TON_IoT Dataset

---

## 1. Installation

```bash
# Cloner ou accéder au projet
cd "Porjet-cynergia/experiment"

# Installer les dépendances (Python >= 3.10 requis)
pip install -r requirements.txt

# Vérifier l'installation
python -c "import xgboost, lightgbm, torch, sklearn; print('OK')"
```

---

## 2. Ordre d'exécution complet

```bash
# Étape 0 — Construire le dataset TON_IoT (~10 min)
python ton_00_build_dataset.py

# Étape 1 — EDA + Stage-1 détection d'anomalies (~5 min)
python ton_01_eda_and_stage1.py

# Étape 2 — Stage-2 + Pipeline hybride + Tests statistiques (~60-90 min)
python ton_02_stage2_and_stats.py

# Évaluation comparative de tous les modèles
python evaluate.py --model compare

# Inférence sur de nouvelles données
python inference.py --input interactive
```

---

## 3. Scripts par modèle individuel

### Stage 1 — Détection d'anomalies

```bash
# Isolation Forest
python models/stage1/train_isolation_forest.py
# Durée : ~5-10s | RAM : ~500 MB | GPU : non requis
# Sortie : results/models/isolation_forest.pkl
#          results/metrics/if_metrics.json
#          results/figures/if_roc_pr.png

# One-Class SVM (RBF) — MEILLEUR Stage-1 sur TON_IoT
python models/stage1/train_ocsvm.py
# Durée : ~20-40s | RAM : ~1 GB | GPU : non requis
# Sortie : results/models/ocsvm.pkl

# Local Outlier Factor
python models/stage1/train_lof.py
# Durée : ~5-15s | RAM : ~1 GB | GPU : non requis
```

### Stage 2 — Classification supervisée

```bash
# XGBoost standalone (MEILLEUR modèle sur TON_IoT)
python models/stage2_ml/train_xgboost.py --mode standalone
# Durée : ~2-4 min | RAM : ~2 GB | GPU : non requis

# XGBoost hybrid (MEILLEUR recall backdoor)
python models/stage2_ml/train_xgboost.py --mode hybrid

# Random Forest
python models/stage2_ml/train_random_forest.py --mode standalone

# LightGBM (utiliser le script dédié pour éviter le bug numpy/class_weight)
python models/stage2_ml/train_lightgbm.py --mode standalone
```

### Deep Learning (GPU recommandé)

```bash
# CNN-LSTM
python models/stage2_dl/train_cnn_lstm.py --window_size 30 --epochs 50
# Durée : ~30 min CPU / ~5 min GPU | RAM : ~4 GB | GPU : recommandé

# Transformer standard
python models/stage2_dl/train_transformer.py --model vanilla --window_size 60
# Durée : ~45 min CPU / ~8 min GPU

# PatchTST (recommandé pour séries temporelles ICS)
python models/stage2_dl/train_transformer.py --model patchtst --window_size 60
# Durée : ~30 min CPU / ~5 min GPU
```

---

## 4. Résultats expérimentaux réels (TON_IoT)

### Stage 1 — Détection d'anomalies

| Modèle | Recall | FPR | AUC-ROC | AUC-PR | Enrichissement |
|---|---|---|---|---|---|
| Isolation Forest | 0.900 | 0.2049 | 0.8865 | 0.9643 | ×0.88 |
| **One-Class SVM (RBF)** | **0.900** | **0.0149** | **0.9767** | **0.9930** | **×1.22** |
| LOF | 0.900 | 0.1632 | 0.9279 | 0.9810 | ×1.13 |

**OCSVM sélectionné** : FPR le plus bas (1.49%), AUC-ROC le plus élevé (0.9767)

### Stage 2 — Comparaison complète

| Modèle | F1-macro | AUC-PR | MCC | backdoor | mitm |
|---|---|---|---|---|---|
| **XGBoost standalone** | **0.8673** | **0.9435** | **0.9236** | 0.3615 | 0.5877 |
| Random Forest standalone | 0.7108 | 0.9407 | 0.8079 | 0.0062 | 0.5355 |
| LightGBM standalone* | 0.4141 | 0.3277 | 0.4604 | 0.0283 | 0.2322 |
| OCSVM + XGBoost | 0.7337 | 0.8648 | 0.8093 | **0.9250** | 0.5319 |
| OCSVM + RF | 0.6300 | 0.8928 | 0.7498 | 0.0003 | 0.5106 |
| OCSVM + LightGBM* | 0.5068 | 0.3910 | 0.6311 | 0.3760 | 0.4043 |

*LightGBM sous-performe à cause du bug numpy/class_weight.
 Utiliser `train_lightgbm.py` (avec DataFrame) pour les vrais résultats.

### Ablation Study (XGBoost backbone)

| Config | F1-macro | Δ vs A0 | AUC-PR | Observation |
|---|---|---|---|---|
| A0: XGB baseline | 0.8145 | — | 0.9403 | Référence |
| A1: XGB + score anomalie | 0.8176 | +0.003 | 0.9397 | Gain marginal |
| A2: XGB + ADASYN seul | 0.7334 | -0.081 | 0.9189 | ADASYN dégrade ! |
| A3: S1 filtre seul | 0.6330 | -0.182 | 0.8750 | Filtre seul insuffisant |
| A4: S1 + ADASYN | 0.7373 | -0.077 | 0.8431 | Meilleur recall backdoor |
| A5: S1 + score + ADASYN | 0.7373 | -0.077 | 0.8431 | Identique à A4 |

**Découverte clé :** La configuration OCSVM+XGB (hybrid) améliore le recall
backdoor de 0.3615 → 0.9250 (+56.4%) mais réduit le F1-macro global.

---

## 5. Explication des métriques

### F1-macro — Métrique principale
```
F1-macro = moyenne(F1_classe1, F1_classe2, ..., F1_classeN)

F1 d'une classe = 2 × (Precision × Recall) / (Precision + Recall)

Precision = TP / (TP + FP)  : quand on prédit l'attaque, on a raison
Recall    = TP / (TP + FN)  : on détecte la plupart des attaques

F1-macro pénalise les mauvaises performances sur les classes rares
(mitm, ransomware) — c'est la métrique la plus importante pour ce projet.
```

### MCC — Matthews Correlation Coefficient
```
MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))

Intervalle : [-1, +1]
  +1 = classification parfaite
   0 = prédiction aléatoire
  -1 = classification inverse

MCC est robuste même si les classes sont très déséquilibrées.
C'est la seule métrique qui prend en compte tous les 4 quadrants
de la matrice de confusion simultanément.
```

### AUC-PR — Area Under PR Curve
```
La courbe Precision-Recall est plus informative que ROC pour les
données très déséquilibrées (mitm : 0.2%, ransomware : 0.9%).

Si 99% des flux sont normaux :
  Un classificateur qui prédit "Normal" toujours obtient AUC-ROC ≈ 0.9
  mais AUC-PR ≈ 0.01 (car il n'identifie jamais les attaques)

→ Toujours rapporter AUC-PR pour des données ICS/IoT déséquilibrées
```

---

## 6. Comprendre les résultats

### Pourquoi ADASYN dégrade les performances (A2 < A0) ?

ADASYN génère des échantillons synthétiques pour équilibrer les classes.
Sur TON_IoT avec XGBoost :
- ADASYN : 321K → 599K échantillons (+86%)
- Les nouvelles données synthétiques de mitm/ransomware sont "trop faciles"
  (générées par interpolation kNN dans des zones déjà bien couvertes)
- XGBoost surfit ces exemples artificiels
- Solution : utiliser class_weight='balanced' à la place

### Pourquoi le pipeline hybride réduit le F1-macro global ?

1. OCSVM filtre 76% des données → le modèle Stage-2 ne voit que 76% du train
2. Le sous-ensemble filtré a une distribution différente (moins de normal)
3. XGBoost perd de l'information sur les classes dominantes
4. Mais il gagne en sensibilité sur les classes rares (backdoor : 0.36→0.93)

### Pourquoi LightGBM sous-performe avec des arrays numpy ?

LightGBM avec `class_weight='balanced'` et des arrays numpy peut silencieusement
ignorer les poids de classes dans certaines versions. Symptôme : F1-macro ≈ 0.4
(proche de 1/n_classes si le modèle prédit tout dans la classe majoritaire).
**Solution** : utiliser `pd.DataFrame` avec des noms de colonnes.

---

## 7. Débogage courant

```bash
# Vérifier que les données sont générées
ls results/metrics/

# Vérifier les résultats Stage 1
python -c "
import json
with open('results/metrics/eda_and_stage1.json') as f:
    d = json.load(f)
print('Best S1:', d['best_s1'])
print('S1 metrics:', d['stage1'][d['best_s1']])
"

# Vérifier les résultats Stage 2
python evaluate.py --model compare

# Tester l'inférence
python inference.py --input interactive
```

---

## 8. Requirements matérielles

| Étape | CPU | RAM | GPU | Durée (estimée) |
|---|---|---|---|---|
| Build dataset | 4 cores | 4 GB | non | 10 min |
| Stage 1 (IF/OCSVM/LOF) | 4 cores | 2 GB | non | 5 min |
| Stage 2 XGBoost (200 arbres) | 8 cores | 4 GB | non | 3-5 min |
| Stage 2 RF/LightGBM | 8 cores | 3 GB | non | 2-4 min |
| CNN-LSTM | 4 cores | 4 GB | CUDA (recommandé) | 30 min / 5 min GPU |
| PatchTST | 4 cores | 4 GB | CUDA (recommandé) | 30 min / 5 min GPU |
| Tests stats (10 seeds) | 8 cores | 4 GB | non | 20-30 min |

---

## 9. Structure des fichiers de sortie

```
results/
├── metrics/
│   ├── ton_iot_sample.csv         ← Dataset TON_IoT samplé
│   ├── dataset_meta.json          ← Statistiques EDA
│   ├── eda_and_stage1.json        ← Résultats Stage 1
│   ├── stage2_and_stats.json      ← Résultats Stage 2 + tests stats
│   ├── if_metrics.json            ← Métriques Isolation Forest
│   ├── ocsvm_metrics.json         ← Métriques OCSVM
│   ├── xgboost_standalone_metrics.json
│   ├── cnn_lstm_metrics.json
│   └── ...
├── figures/
│   ├── fig01_class_distribution.png/pdf
│   ├── fig02_class_imbalance.png/pdf
│   ├── fig03_feature_correlation.png/pdf
│   ├── fig04_stage1_roc.png/pdf
│   ├── fig05_stage1_pr.png/pdf
│   ├── fig06_stage1_enrichment.png/pdf
│   ├── fig07_threshold_calibration.png/pdf
│   ├── fig08_cm_*.png/pdf         ← Matrices de confusion
│   ├── fig09_stage2_roc.png/pdf
│   ├── fig10_pr_*.png/pdf         ← PR curves par classe
│   ├── fig11_ablation.png/pdf
│   └── fig12_main_comparison.png/pdf
├── saved_models/
│   ├── isolation_forest.pkl
│   ├── ocsvm.pkl
│   ├── xgboost_standalone.pkl
│   ├── xgboost_hybrid.pkl
│   ├── cnn_lstm.pt
│   └── patchtst.pt
└── latex/
    ├── main.tex
    ├── tables/*.tex
    └── sections/*.tex
```
