# Détection d'anomalies : Ensembles de vote pondérés avec PyOD

## Vue d'ensemble

Système complet de détection d'anomalies combinant plusieurs algorithmes PyOD via trois stratégies d'agrégation :
- **Hard Voting** : vote majoritaire (simple ou pondéré)
- **Soft Voting** : moyenne normalisée des scores continus
- **S&F** : apprentissage adaptatif des poids via prédictions out-of-fold

L'ensemble offre une robustesse accrue par rapport à chaque modèle seul, particulièrement 
sur données déséquilibrées ou bruitées.

## Démarrage rapide

### Installation des dépendances

```bash
pip install numpy pandas scikit-learn pyod streamlit joblib
```

### Lancer le benchmark interactif

```bash
# À la racine du projet
streamlit run streamlit_app/Hub.py
```

Puis dans l'interface :
1. Sélectionnez **"Benchmark ADBench"**
2. Choisissez un dataset, la métrique d'optimisation, et les modèles/stratégies
3. Cliquez "Lancer le benchmark"

### Exemple minimal en Python

```python
from classes.MyVotingPyOD import MyVotingPyOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF

# Créer des estimateurs
estimateurs = [
    ("iforest", IForest(contamination=0.1, random_state=42)),
    ("lof", LOF(contamination=0.1))
]

# Ensemble avec S&F
ensemble = MyVotingPyOD(
    estimators=estimateurs,
    voting='S&F',
    verbose=True
)

# Entraîner et prédire
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
scores = ensemble.decision_function(X_test)
```

## Architecture

### Core modules

| Fichier | Rôle |
|---------|------|
| `classes/MyVotingPyOD.py` | Implémentation ensemble (hard/soft/S&F) |
| `classes/utils/Trouve_params_pyod.py` | GridSearch + CV pour PyOD |
| `streamlit_app/app_benchmark_adbench.py` | Interface benchmark principale |
| `docs/ALL.md` | Documentation complète (1500+ lignes) |

### Données

- Téléchargées via `classes/utils/download_adbench.py`
- Format : `.npz` (NumPy) avec `X` (features) et `y` (anomaly labels 0/1)
- Stockées dans `data/adbench/`

## Concepts clés

### Hard Voting

Vote majoritaire sur prédictions binaires (0=inlier, 1=outlier), possiblement pondéré 
par fiabilité estimée.

### Soft Voting

1. Normalisation robuste des scores continus par percentile-rank
2. Moyenne pondérée → score agrégé [0,1]
3. Seuil fixe = percentile(1 - contamination) sur train
4. Prédiction binaire par comparaison au seuil

### S&F (Source & Faits)

Optimisation itérative des poids relatifs des estimateurs :
- Genère prédictions out-of-fold (StratifiedKFold)
- Lance coordinate ascent multiplicatif avec clipping
- Maximise F1-Score ou Accuracy sur OOF
- Converge en 20-50 itérations typiquement

## Évaluation

### Métriques

- **Accuracy** : simple mais masque déséquilibres
- **F1-Score** : équilibre precision/recall → recommandé
- **ROC-AUC** : qualité du classement à tous les seuils
- **PR-AUC** (average_precision) : meilleur pour anomalies rares

Train et Test affichés séparément : écart important = surapprentissage.

### Résultats typiques

Sur ADBench (3000+ examples, ~10-20% anomalies) :
- Hard : F1 ≈ 0.60-0.65, train/test gap petit
- Soft : F1 ≈ 0.64-0.70, robuste au déséquilibre
- S&F : F1 ≈ 0.66-0.72, meilleur en anomalies rares

## Troubleshooting

### Poids S&F divergent ou non convergents
→ Vérifier que `y_train` est fourni et contient au moins 2 classes  
→ Augmenter `max_iter` dans `_fit_SF()`

### Rappel anomalies trop faible (soft)
→ Seuil trop conservateur ; réduire manuellement ou basculer en S&F

### OOM sur gros datasets (>100k)
→ Réduire `test_size` ou `cv` folds  
→ Utiliser sous-ensemble pour optimisation hyperparamètres

## Extensions futures

- Support deep learning (autoencoders via PyOD)
- Explainabilité (SHAP pour votes)
- API REST FastAPI
- Persistence + batch prediction
- Metrics dashboard interactif

## Structure projet

```
PROJET-STAGE/
├── classes/
│   ├── MyVotingPyOD.py
│   ├── MyVotingClassifier.py
│   ├── MyVotingOutlier.py
│   └── utils/
│       ├── Trouve_params_pyod.py
│       ├── ChargeurDonnees.py
│       └── download_adbench.py
├── streamlit_app/
│   ├── Hub.py
│   ├── app_benchmark_adbench.py
│   ├── app_explore_adbench.py
│   └── app_anomalie.py
├── data/adbench/              ← Datasets NPZ
├── docs/ALL.md                ← Documentation 1500+ lignes
├── tests/                     ← Tests unitaires
└── README.md                  ← Ce fichier
```

## Références

- **PyOD** : https://pyod.readthedocs.io/
- **scikit-learn** : https://scikit-learn.org/
- **Streamlit** : https://streamlit.io/
- **ADBench** : Diverse Benchmark for Anomaly Detection (Yuan et al., 2021)

## License

[À compléter selon votre choix : MIT, Apache 2.0, etc.]

## Contact

[Votre nom]  
[Email]  
Île-de-France, France
