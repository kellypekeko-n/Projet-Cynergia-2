# Documentation - Framework Hybride de Détection d'Anomalies en Réseau

## Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture du système](#architecture-du-système)
3. [Méthodologie](#méthodologie)
4. [Composants](#composants)
5. [Pipeline d'exécution](#pipeline-dexécution)
6. [Résultats et métriques](#résultats-et-métriques)
7. [Guide d'utilisation](#guide-dutilisation)
8. [Dépendances](#dépendances)

---

## Vue d'ensemble

Ce notebook implémente un **framework hybride de détection d'anomalies en réseau** utilisant une approche à deux étages :

- **Étage 1 (Filtre)** : Détection d'anomalies générales via Isolation Forest ou Autoencoder
- **Étage 2 (Spécialiste)** : Classification des types d'attaques via XGBoost

Le système est conçu pour identifier et classifier les flux malveillants dans des environnements ICS (Industrial Control Systems) en utilisant le dataset CICIDS 2017.

### Objectifs principaux

- Maximiser le **Recall** (intercepter le plus d'attaques possible)
- Minimiser les **Faux Positifs** (réduire les blocages injustifiés de trafic normal)
- Classifier avec précision les types d'attaques détectées

---

## Architecture du système

### Approche en deux étages

```
Trafic réseau
     ↓
┌─────────────────────┐
│  ÉTAGE 1 : FILTRE   │ (Isolation Forest OU Autoencoder)
├─────────────────────┤
│ - Détecte anomalies │
│ - Sépare normal     │
└────────┬────────────┘
         │
    Anomalies détectées
         ↓
┌─────────────────────┐
│ ÉTAGE 2 : EXPERT    │ (XGBoost Classifier)
├─────────────────────┤
│ - Classifie attaques│
│ - Types spécifiques │
└─────────────────────┘
         ↓
   Résultats finaux
   (Type d'attaque)
```

### Deux implémentations alternatives pour l'Étage 1

#### Option 1 : Isolation Forest
- Algorithme de détection d'anomalies basé sur l'isolation
- **Avantages** : Rapide, efficace sur données non-équilibrées
- **Paramètre clé** : `contamination=0.01` (tolère 1% d'anomalies dans données normales)

#### Option 2 : Autoencoder (Deep Learning)
- Réseau de neurones encoder-décodeur
- **Architecture** : 
  - Encodeur : Input → 32 → 16 → 8 (bottleneck)
  - Décodeur : 8 → 16 → 32 → Output
- **Avantages** : Plus sophistiqué, capture anomalies complexes
- **Seuil** : 99e percentile de l'erreur de reconstruction (MSE)

---

## Méthodologie

### 1. Préparation des données

#### Chargement et séparation
```python
# Source : Dataset.csv
df = pd.read_csv("Dataset.csv")
df_normal = df[df['NST_M_Label'] == 'Normal']
df_attacks = df[df['NST_M_Label'] != 'Normal']
```

#### Extraction des caractéristiques
- **Plage** : De la colonne `duration` à `rAckDelayAvg`
- **Nettoyage** : 
  - Remplacement des valeurs infinies par NaN
  - Remplissage des NaN par 0
  - Gestion de l'équilibre des classes

#### Séparation train/test
- Trafic normal : **80% entraînement** (pour l'Étage 1) / **20% test**
- Trafic d'attaques : **100% réservé au test** (le modèle ne les voit jamais en entraînement)

### 2. Normalisation (Standardisation)

```python
scaler = StandardScaler()
X_normal_train_scaled = scaler.fit_transform(X_normal_train)
X_normal_test_scaled = scaler.transform(X_normal_test)
X_attacks_scaled = scaler.transform(X_attacks)
```

**Important** : Le scaler est **entraîné UNIQUEMENT sur le trafic normal**
- Cela assure que les anomalies restent détectables

### 3. Entraînement Étage 1

#### Isolation Forest
```python
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_normal_train_scaled)
predictions = iso_forest.predict(X_test_global)  # -1 = anomalie, 1 = normal
```

#### Autoencoder
```python
# Entraînement uniquement sur le trafic normal
autoencoder.fit(X_normal_train_scaled, X_normal_train_scaled,
    epochs=40, batch_size=64, validation_split=0.1, callbacks=[early_stop])

# Détection par seuil de reconstruction
mse_values = np.mean((X_test - predictions)^2, axis=1)
threshold = np.percentile(mse_normal, 99)  # 99e percentile
anomalies = mse_values > threshold
```

### 4. Filtrage et passage à l'Étage 2

```python
# Seules les anomalies détectées par l'Étage 1 passent à l'Étage 2
masque_anomalies = (predictions_etage1 == -1)  # ou (mse_global > seuil)
X_pour_etage2 = X_test_global[masque_anomalies]
y_pour_etage2 = y_test_global[masque_anomalies]
```

### 5. Encodage des étiquettes

```python
encoder_etage2 = LabelEncoder()
y_encoded = encoder_etage2.fit_transform(y_pour_etage2_raw)
# Exemple : 'Normal' → 0, 'DDoS' → 1, 'PortScan' → 2, etc.
```

### 6. Entraînement et évaluation Étage 2

```python
# Division des anomalies détectées
X_train_e2, X_test_e2, y_train_e2, y_test_e2 = train_test_split(
    X_pour_etage2, y_pour_etage2, test_size=0.3, random_state=42
)

# Entraînement du classifieur
xgb_spec = XGBClassifier(n_estimators=100, random_state=42)
xgb_spec.fit(X_train_e2, y_train_e2)

# Évaluation
y_pred = xgb_spec.predict(X_test_e2)
print(classification_report(y_test_e2, y_pred, 
      target_names=encoder_etage2.classes_))
```

---

## Composants

### Cellule 1 : Chargement et Préparation
- Charge `Dataset.csv`
- Sépare trafic normal et attaques
- Normalise les données
- **Sortie** : X_normal_train_scaled, X_normal_test_scaled, X_attacks_scaled

### Cellule 2 : Étage 1 - Isolation Forest
- Entraîne un modèle IsolationForest sur le trafic normal
- Évalue le Recall et les Faux Positifs
- **Métriques clés** :
  - Attaques interceptées (%)
  - Faux Positifs générés

### Cellule 3 & 4 : Étage 1 - Isolation Forest + Étage 2 - XGBoost
- Version initiale et version corrigée avec `.loc`
- Filtre les anomalies et entraîne XGBoost
- **Rapport** : Classification Report complet (Précision, Recall, F1-Score)

### Cellule 5 : Installation TensorFlow
- Installe TensorFlow pour les modèles Autoencoder

### Cellule 6 : Autoencoder
- Définit architecture encoder-décodeur
- Entraîne sur le trafic normal uniquement
- **Callback** : EarlyStopping pour éviter le surapprentissage

### Cellule 7 : Autoencoder - Évaluation Étage 1
- Calcule MSE (Mean Squared Error) pour tous les flux
- Définit le seuil d'anomalie (99e percentile)
- Évalue Recall et Faux Positifs
- Prépare les anomalies pour l'Étage 2

### Cellule 8 : Autoencoder + XGBoost - Résultats finaux
- Entraîne XGBoost sur anomalies détectées par l'Autoencoder
- Affiche rapport de classification final

---

## Pipeline d'exécution

### Flux complet d'exécution

```
1. Exécuter Cellule 1
   → Charger et normaliser les données

2. Choisir Étage 1 :
   
   Option A : Isolation Forest
   → Exécuter Cellule 2
   → Exécuter Cellule 3 ou 4
   
   Option B : Autoencoder
   → Exécuter Cellule 5 (TensorFlow)
   → Exécuter Cellule 6 (Entraînement)
   → Exécuter Cellule 7 (Évaluation)
   → Exécuter Cellule 8 (Résultats finaux)

3. Analyser les résultats
   → Classification Report
   → Métriques Recall/Faux Positifs
```

### Ordre recommandé

1. **Toujours** : Cellule 1 (préparation obligatoire)
2. **Pour un test rapide** : Cellule 2 (Isolation Forest simple)
3. **Pour analyse avancée** : Cellules 5-8 (Autoencoder complet)

---

## Résultats et métriques

### Métriques évaluées

#### Étage 1 (Filtre d'anomalies)

| Métrique | Description | Importance |
|----------|-------------|-----------|
| **Recall** | % d'attaques interceptées | ⭐⭐⭐ Critique |
| **Faux Positifs** | % trafic normal bloqué à tort | ⭐⭐⭐ Critique |
| **Seuil d'anomalie** | Paramètre de décision | ⭐⭐ Important |

#### Étage 2 (Classification d'attaques)

| Métrique | Description |
|----------|-------------|
| **Précision** | Corrélation des prédictions |
| **Recall** | Couverture par classe d'attaque |
| **F1-Score** | Moyenne harmonique Précision/Recall |
| **Support** | Nombre d'échantillons par classe |

### Exemple de sortie

```
=== ÉVALUATION DE L'ÉTAGE 1 (Isolation Forest) ===
Attaques interceptées (Recall) : 8542 / 9000 (94.91%)
Fausses alertes générées (Faux Positifs) : 145 / 2250

=== RÉSULTATS FINAUX : PIPELINE HYBRIDE ===
              precision    recall  f1-score   support

       Normal       0.92      0.88      0.90       500
         DDoS       0.95      0.97      0.96       450
      Botnet       0.89      0.91      0.90       320
    PortScan       0.87      0.85      0.86       280

    accuracy                           0.91      1550
   macro avg       0.91      0.90      0.91      1550
weighted avg       0.91      0.91      0.91      1550
```

---

## Guide d'utilisation

### Préalables

- Python 3.7+
- Dataset `Dataset.csv` dans le même répertoire que le notebook
- Colonnes requises : `duration`, `rAckDelayAvg`, `NST_M_Label`, ...[tous les features]

### Installation des dépendances

```bash
pip install pandas numpy scikit-learn xgboost tensorflow
```

### Exécution étape par étape

#### 1. Préparation
```python
# Cellule 1 : Charge le dataset et normalise
# Vérifie : X_normal_train_scaled.shape, X_normal_test_scaled.shape, X_attacks_scaled.shape
```

#### 2. Test rapide avec Isolation Forest
```python
# Cellule 2 : Entraîne et évalue le filtre
# Affiche : Recall (%) et Faux Positifs
```

#### 3. Évaluation complète (Option A)
```python
# Cellule 3 ou 4 : Combine Isolation Forest + XGBoost
# Affiche : Classification Report détaillé
```

#### 4. Analyse avancée avec Autoencoder (Option B)
```python
# Cellule 5 : Installe TensorFlow
# Cellule 6 : Entraîne l'Autoencoder (40 epochs)
# Cellule 7 : Évalue le filtrage par Autoencoder
# Cellule 8 : XGBoost sur anomalies détectées
# Affiche : Classification Report final
```

### Paramètres à ajuster

#### Isolation Forest
```python
IsolationForest(
    contamination=0.01,      # 0.005 pour plus strict, 0.02 pour plus permissif
    random_state=42
)
```

#### Autoencoder
```python
# Architecture
encoded = Dense(32, activation="relu")(input_layer)    # Augmenter si dimensions élevées
encoded = Dense(16, activation="relu")(encoded)
bottleneck = Dense(8, activation="relu")(encoded)       # Réduire pour forcer compression

# Entraînement
autoencoder.fit(...,
    epochs=40,               # Augmenter pour convergence
    batch_size=64,          # Réduire si RAM limitée
    validation_split=0.1,
    callbacks=[EarlyStopping(...)]
)
```

#### XGBoost
```python
XGBClassifier(
    n_estimators=100,       # 50-200 selon données
    random_state=42,
    max_depth=6,           # Augmenter pour modèles complexes
    learning_rate=0.1      # Réduire pour convergence plus stable
)
```

### Interprétation des résultats

#### Cas 1 : Isolation Forest fonctionne bien
- Recall > 90% : Le filtre détecte bien les attaques
- Faux Positifs < 10% : Peu de trafic normal bloqué
- → **Action** : Utiliser Isolation Forest en production (plus rapide)

#### Cas 2 : Isolation Forest insuffisant
- Recall < 80% ou Faux Positifs > 20%
- → **Action** : Essayer l'Autoencoder (Cellules 5-8)

#### Cas 3 : XGBoost faible performance
- F1-Score < 0.80 pour certaines classes
- → **Action** : Ajuster hyperparamètres, augmenter données

---

## Dépendances

### Bibliothèques Python

| Bibliothèque | Version | Usage |
|-------------|---------|-------|
| **pandas** | ≥1.0 | Manipulation de données |
| **numpy** | ≥1.18 | Opérations numériques |
| **scikit-learn** | ≥0.22 | Isolation Forest, StandardScaler, Train/Test Split, Métriques |
| **xgboost** | ≥1.0 | Classifieur XGBoost |
| **tensorflow** | ≥2.0 | Autoencoder (Keras) |

### Installation complète

```bash
pip install pandas numpy scikit-learn xgboost tensorflow
```

### Compatibilité OS

- ✅ Windows (avec `sys.executable` pour pip)
- ✅ macOS
- ✅ Linux

---

## Conclusions et recommandations

### Avantages du framework hybride

1. **Détection robuste** : Deux approches complémentaires
2. **Seuil bas de détection** : Rappel élevé
3. **Classification précise** : XGBoost identifie types d'attaques
4. **Scalabilité** : Deux étages permettent optimisation indépendante

### Cas d'usage

- ✅ Monitoring de trafic ICS en temps réel
- ✅ Analyse post-incident d'attaques réseau
- ✅ Validation de listes noires (IDS/IPS)
- ✅ Recherche en cybersécurité

### Limitations

- ⚠️ Necessité d'exempt du trafic normal pour entraînement
- ⚠️ Peut nécessiter tuning fin des hyperparamètres
- ⚠️ Performance dépendante de la qualité du dataset
- ⚠️ Pas d'explainability native (black box)

---

**Document généré automatiquement à partir du notebook `isolationdesTrafic.ipynb`**
Date: 2024
