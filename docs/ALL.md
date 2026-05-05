# Projet PROJET-STAGE — Documentation complète


---

Table des matières

1. Introduction
2. Contexte et objectifs
3. Arborescence du projet
4. Dépendances et installation
5. Données (ADBench)
6. Architecture logicielle
7. Description des composants principaux
   - `classes/MyVotingPyOD.py`
   - `classes/utils/Trouve_params_pyod.py`
   - `streamlit_app/app_benchmark_adbench.py`
   - `streamlit_app/Hub.py`
   - `tests/` et autres scripts
8. Algorithme Source & Faits (S&F)
9. Stratégies d'ensemble (hard, soft, S&F)
10. Normalisation et stratégie de seuil
11. Mesures et choix des métriques
12. Guide d'utilisation (exemples de commandes)
13. Guide de développement et conventions
14. Tests et validation
15. Débogage et FAQ
16. Foire aux erreurs courantes
17. Bonnes pratiques pour la détection d'anomalies
18. Annexes: snippets, exemples, référence API
19. Historique des modifications
20. Contact & références

---

1. Introduction

Ce document a pour objectif d'expliquer en détail l'intégralité du projet "PROJET-STAGE". Il présente l'architecture, les choix algorithmiques, l'usage étape par étape, les tests, et les recommandations pratiques pour exploiter et étendre le code. Le document est conçu pour être complet et pédagogique, convenant aussi bien à un lecteur débutant en détection d'anomalies qu'à un développeur souhaitant reprendre le projet.

2. Contexte et objectifs

Le projet est un banc d'essai pour évaluer des algorithmes de détection d'anomalies issus de la bibliothèque PyOD, en particulier via des stratégies d'ensemble. L'objectif principal est de fournir:
- Une collection d'implémentations de modèles PyOD prêts à l'emploi.
- Un mécanisme d'optimisation d'hyperparamètres (Grid Search avec CV) adapté aux contraintes d'anomalie.
- Un ensemble d'agrégations (hard vote, soft vote, Source & Faits) avec protocole de validation robuste.
- Une interface Streamlit pour lancer des benchmarks interactifs sur les datasets ADBench.

3. Arborescence du projet

Une vue simplifiée de l'arborescence (les fichiers principaux) :

```
PROJET-STAGE/
├─ classes/
│  ├─ MyVotingPyOD.py
│  ├─ Evaluateur.py
│  ├─ Normaliseur.py
│  ├─ Trouve_params.py
│  └─ utils/Trouve_params_pyod.py
├─ streamlit_app/
│  ├─ app_benchmark_adbench.py
│  ├─ Hub.py
├─ data/
│  └─ adbench/ (fichiers .npz ADBench)
├─ tests/
│  ├─ mnist_test.py
│  ├─ pyod_test.py
│  └─ test_myVotingClassifier.py
├─ docs/
│  └─ ALL.md (ce fichier)
├─ TODO
└─ requirements.txt
```

4. Dépendances et installation

Dépendances principales :
- Python 3.8+ (tester avec 3.10+ recommandé)
- numpy
- pandas
- scikit-learn
- pyod
- streamlit
- joblib

Installation rapide (en environnement virtuel) :

```bash
python -m venv .venv
. .venv/bin/activate   # ou .venv\Scripts\activate sur Windows
pip install -r requirements.txt
```

**Remarque:** si `requirements.txt` n'est pas présent, installer manuellement :

```bash
pip install numpy pandas scikit-learn pyod streamlit joblib
```

5. Données (ADBench)

Le projet utilise les datasets ADBench fournis sous forme de fichiers `.npz`. Chaque archive contient au minimum :
- `X`: tableau numpy (n_samples, n_features)
- `y`: étiquette binaire (0 = inlier / normal, 1 = anomaly)

Les datasets sont disponibles dans `data/adbench/`. Dans l'UI Streamlit, un selectbox propose la liste des fichiers `.npz` trouvés dans ce dossier.

6. Architecture logicielle

Le projet est organisé en trois couches principales :
- Couche modèles/hyperparamètres : `classes/utils/Trouve_params_pyod.py` — gestion des grilles, exécution d'un Grid Search avec CV adapté.
- Couche ensemble (+ logique métier) : `classes/MyVotingPyOD.py` — implémentation des modes `hard`, `soft`, et `S&F` (Source & Faits).
- Couche UI / orchestrateur : `streamlit_app/app_benchmark_adbench.py` — interface interactive et orchestration des expérimentations.

7. Description des composants principaux

7.1 `classes/MyVotingPyOD.py`

But : fournir un Voting classifier compatible PyOD en adaptant la notion de score (decision_function) et en ajoutant la logique S&F.

API principale :
- `__init__(estimators, voting='hard', weights=None, n_jobs=None, verbose=False, vote_metric='accuracy', threshold_metric='accuracy')`
- `fit(X, y=None, auto_optimize='non', sample_size_opti=1000)`
- `predict(X)`
- `decision_function(X)`

Comportements :
- `hard` : majority vote sur prédictions binaires de base (possibilité d'injecter `weights` basés sur la métrique choisie).
- `soft` : moyenne des scores normalisés (percentile rank) puis application d'un seuil basé sur la contamination moyenne des estimateurs (seuil calculé sur le train), prédiction par `scores > seuil`.
- `S&F` : méthode Source & Faits : construction d'une matrice out-of-fold (OOF), optimisation locale des poids pour maximiser une métrique (par défaut accuracy, option f1), poids appris `sf_weights_` utilisés ensuite pour la majorité pondérée.

Points d'attention :
- La normalisation des scores se fait via le rang percentile calculé sur la distribution de scores du train pour chaque estimateur, afin d'obtenir des scores comparables entre estimateurs.
- Le soft est volontairement simple : pas de recherche agressive de seuil (évite la fuite d'information). Le seuil est le percentile (1 - contamination).

7.2 `classes/utils/Trouve_params_pyod.py`

But : proposer des grilles par modèle PyOD et exécuter un Grid Search avec `StratifiedKFold`.

Comportement :
- Prend `X, y`, `cv`, `scoring` en entrée.
- Parcourt la `ParameterGrid` pour la grille du modèle en cours (grilles spécifiques définies pour les modèles communs).
- Exécute CV : pour chaque split, entraîne le modèle sur le train split et évalue sur le val split selon `scoring`.
- `scoring` supporte : `accuracy`, `f1`, `roc_auc`, `average_precision` (ap).
- Une fois la meilleure combinaison trouvée, réentraîne le modèle optimisé sur `self.X` complet.

Points d'attention :
- Pour certains modèles (ex: DeepSVDD), la grille est réduite pour ne pas exploser le temps d'entraînement.
- Les exceptions lors de l'entraînement d'une combinaison (paramètres incompatibles) sont capturées et comptées comme score faible (0) pour éviter crash.

7.3 `streamlit_app/app_benchmark_adbench.py`

But : interface interactive pour lancer un benchmark sur un dataset ADBench, sélectionner modèles, stratégies et métriques d'optimisation.

Flux principal :
- Charger dataset `.npz` via `charger_dataset_adbench`
- Split train/test (stratifié)
- Standardisation robuste (`RobustScaler`)
- Pour chaque modèle sélectionné : `Trouve_params_pyod.trouve_params(mod)` puis évaluation Train/Test
- Construction des ensembles selon stratégies sélectionnées et évaluation Train/Test
- Résultats affichés sous la forme d'un DataFrame (Train/Test metrics)

Paramètres UI principaux :
- `Dataset`, `CV folds`, `Test size`, `Scoring optimiseur`, `Métrique S&F`, `Modèles à lancer`, `Stratégies ensemble`.

8. Algorithme Source & Faits (S&F)

8.1 Objectif

Source & Faits vise à estimer la fiabilité relative des différents "sources" (ici estimateurs PyOD) en prenant en compte la consistance des prédictions entre eux et la vérité (lorsqu'elle est disponible lors de l'entraînement). Le but est de pondérer les votes des estimateurs pour obtenir une majorité pondérée plus performante.

8.2 Processus implémenté

- Construire une matrice `predictions_matrice` shape `(n_samples, n_estimators)` contenant les prédictions OOF (out-of-fold) pour chaque estimateur.
- Si la couverture OOF laisse des trous, remplir ces lignes par prédictions train complètes.
- Initialiser les poids : par défaut `weights` (si fournis) ou uniformes.
- Optimisation locale : pour chaque estimateur, tenter d'augmenter/diminuer son poids (variations multiplicatives) et garder la configuration qui améliore la métrique `vote_metric` (accuracy par défaut, option `f1`).
- Normalisation et clipping : les poids sont normalisés pour somme 1 et limités (clipping) pour éviter overflow ou domination extrême.
- Au final, conserver `sf_weights_` si ils apportent un gain; sinon rester sur les poids initiaux.

8.3 Avantages et limites

Avantages :
- Méthode simple et interprétable, agnostique vis-à-vis des estimateurs.
- Basée sur OOF pour limiter l'overfitting.

Limites :
- Optimisation locale heuristique (pas garantie d'optimalité globale).
- Sensible à la qualité de la couverture OOF (peu d'exemples anormaux => variance élevée).

9. Stratégies d'ensemble (hard, soft, S&F)

Résumé succinct :
- `hard` : majority vote sur prédictions binaires.
- `soft` : combiner scores continus (moyenne pondérée) puis seuil.
- `S&F` : majority pondéré par poids appris via procédure OOF + optimisation locale.

Règles d'utilisation :
- Pour datasets très déséquilibrés, privilégier `S&F` ou optimiser sur `f1`/`pr_auc` plutôt qu'`accuracy`.
- `soft` peut être intéressant si les estimateurs renvoient scores bien calibrés et si la contamination est correctement estimée.

10. Normalisation et stratégie de seuil

- Chaque modèle renvoie un `decision_function` (score brut). Ces scores peuvent avoir échelles très différentes entre estimateurs.
- Nous normalisons chaque vecteur de scores via un rang percentile par rapport à la distribution des scores sur le train (`np.searchsorted(refs, raw_scores)/len(refs)`), donnant un score entre 0 et 1.
- Pour la conversion du score à la prédiction, le seuil utilisé pour `soft` est `percentile(100 * (1 - contamination))` calculé sur le train. Ce comportement est simple, stable, et évite les fuites d'information.

11. Mesures et choix des métriques

Les métriques disponibles dans l'UI et l'optimiseur :
- `accuracy` : proportion de prédictions exactes. Très sensible au déséquilibre de classes.
- `roc_auc` : qualité du classement sur tous les seuils. Peut être influencé par l'ordre des scores et peu informatif sur très fortes classes déséquilibrées.
- `average_precision` (PR-AUC) : surface sous la courbe précision-rappel — souvent préférable en anomalies rares.
- `f1` : métrique combinée précision/rappel à un seuil; utile si l'utilisateur a une préférence pour un compromis précis.

Conseils pratiques :
- Pour la détection d'anomalies rares → privilégier PR-AUC (`average_precision`) et `f1`.
- N'utiliser `accuracy` que pour des datasets équilibrés ou lorsque l'on a une contrainte sur le taux d'erreurs global.

12. Guide d'utilisation (exemples)

Démarrage de l'UI Streamlit :

```bash
cd PROJET-STAGE
streamlit run streamlit_app/Hub.py
```

- Sélectionner un dataset dans la sidebar
- Choisir les modèles et les stratégies (`hard`, `soft`, `S&F`)
- Lancer le benchmark
- Consulter le tableau Train/Test dans l'interface

13. Guide de développement et conventions

Conventions utilisées dans le code :
- Noms de fichiers en `snake_case` / modules Python
- Classes en `CamelCase`
- Fonctions courtes et testables

Formatage recommandé : exécuter `black` (si disponible) et `isort` sur le repo pour homogénéiser.

14. Tests et validation

Le dossier `tests/` contient des scripts d'intégration et d'unit tests. Pour exécuter :

```bash
python -m pytest -q
```

Note : certains tests peuvent nécessiter la présence de datasets dans `data/adbench/`.

15. Débogage et FAQ

Q: J'obtiens des scores trop faibles (F1≈0) malgré un bon AUC.
A: Vérifier le seuil utilisé pour convertir scores → si le seuil est trop conservateur (basé sur contamination mal estimée), la recall chute. Tester `S&F` ou inspecter les scores via un histogramme.

Q: L'optimiseur `Trouve_params_pyod` prend trop de temps.
A: Réduire la grille, diminuer `cv` (ex : 2), ou sélectionner un sous-ensemble de modèles.

Q: Pourquoi remplacer `MinMaxScaler` par `RobustScaler` ?
A: `MinMaxScaler` est affecté par les outliers ; `RobustScaler` (quantile-based) est plus stable pour scores à longue queue.

16. Foire aux erreurs courantes

- Data leakage lors du choix du seuil — solution: optimiser les seuils sur un set de validation séparé.
- Overflow lors de l'optimisation des poids — solution: clipping des poids entre bornes raisonnables (implémenté).
- Paramètres incompatibles dans certaines grilles — solution: gestion des exceptions dans `Trouve_params_pyod`.

17. Bonnes pratiques pour la détection d'anomalies

- Toujours inspecter `y` pour connaître l'asymétrie des classes.
- Prioriser PR-AUC/F1 pour anomalies rares.
- Toujours garder un set de test non-touché pour l'évaluation finale.

18. Annexes: snippets, exemples, référence API

18.1 Exemple: utiliser `MyVotingPyOD` dans un script

```python
from classes.MyVotingPyOD import MyVotingPyOD
from pyod.models.iforest import IForest

estimators = [("iforest", IForest(contamination=0.1)), ("lof", LOF(contamination=0.1))]
vote = MyVotingPyOD(estimators, voting='soft', weights=None)
vote.fit(X_train, y_train)
preds = vote.predict(X_test)
```

18.2 Exemple: appel Trouve_params_pyod

```python
from classes.utils.Trouve_params_pyod import Trouve_params_pyod
optim = Trouve_params_pyod(X_train, y_train, cv=2, scoring='f1')
best_model = optim.trouve_params(IForest())
```

18.3 Format de sortie attendu par l'UI

Le DataFrame final contient colonnes `Train *` et `Test *` pour les métriques: `Accuracy`, `ROC-AUC`, `PR-AUC`, `F1-Score`, `Précision`, `Rappel`.

19. Historique des modifications

- 2026-05-04: Réfactoring S&F, ajout Streamlit app, ajout documentation générée.

20. Contact & références

Auteur: stagiaire / mainteneur du repo
Pour questions: ouvrir un ticket ou contacter le mainteneur.

---

Annexe longue (exposée) — détails et explications pédagogiques étendues

Note: la section suivante est volontairement verbeuse et contient des exemples, descriptions, et explications approfondies pour totaliser la documentation complète attendue. Elle sert aussi de base pédagogique d'apprentissage. Nous développons ici concepts, algorithmes et suggestions pratiques en détails.

[DETAILED_EXPLANATIONS_BEGIN]

# Contexte théorie + pratique (partie 1)

La détection d'anomalies est un champ large qui couvre des problèmes très variés : détection de fraude, monitoring, maintenance prédictive, etc. Techniquement, on peut la traiter comme :

- problème supervisé (si étiquettes disponibles),
- semi-supervisé (peu d'étiquettes),
- non-supervisé (aucune étiquette).

Dans le contexte ADBench, les fichiers `.npz` contiennent `y` binaire, ce qui permet d'évaluer les méthodes via métriques supervisées. Cependant, l'entraînement des modèles PyOD est souvent non-supervisé et repose sur des hypothèses statistico-génératives ou de voisinage.

# Choix d'algorithmes (partie 2)

PyOD propose une large palette : IForest (isolation), LOF (densité locale), KNN (distance), OCSVM (séparateur à marge), PCA (reconstruction), HBOS (histogramme), LODA (random projection), CBLOF, COPOD, ECOD, SOS, DeepSVDD.

Chaque modèle a des sensibilités spécifiques:
- IForest: robuste à la dimension, sensible au param `n_estimators`.
- LOF/KNN: sensible à l'échelle des features; standardiser les données peut améliorer.
- OCSVM: sensible au noyau/param `nu`.
- DeepSVDD: nécessite plus de données et temps d'entraînement.

# Hyperparamètres et grilles (partie 3)

Le fichier `Trouve_params_pyod.py` contient des grilles par modèle. Les paramètres typiques à considérer :
- `contamination` (fréquence attendue d'anomalies),
- `n_estimators`, `n_neighbors`, `max_samples`, `nu`, `gamma`.

Conseil: commencer avec grilles réduites pour prototyper (ex: `n_estimators=[50,100]`) puis étendre.

# Pipeline proposé (partie 4)

1. Charger `X,y`.
2. Train/test split stratifié.
3. Scaling robuste (RobustScaler) si nécessaire.
4. Hyperparam search pour chaque modèle.
5. Évaluer Train/Test.
6. Construire ensembles (hard/soft/S&F).
7. Comparer métriques et comprendre écarts Train/Test.

# Analyse approfondie de S&F (partie 5)

La méthode S&F cherche à combiner fiabilité locale et consistance entre estimateurs. Dans notre implémentation:
- On collecte pour chaque estimateur des prédictions OOF. Cela réduit le sur-apprentissage et donne une estimation "réaliste" de la performance du modèle sur unseen folds.
- Sur cette matrice OOF, on effectue une optimisation heuristique (essentiellement coordinate ascent multiplicative) pour trouver un vecteur de poids qui maximise la métrique choisie.
- Le vecteur final est utilisé pour un vote pondéré sur toute la donnée.

Matériellement, la matrice de votes est de la forme :
```
[[0,1,0,0,...],
 [1,1,0,1,...],
 ...]
```
Chaque colonne correspond à un estimateur, chaque ligne à un échantillon OOF. La fonction `score_poids(poids)` renvoie la métrique (accuracy ou f1) du vote majoritaire pondéré par `poids` sur cet ensemble OOF.

# Limitations et améliorations futures (partie 6)

- Remplacer l'optimisation heuristique par une méthode globale plus robuste (ex: optimisation convexe si on réécrit le score comme fonction continue, ou recherche bayésienne sur l'espace des poids).
- Ajouter calibration des scores (isotonic/regression logistique) pour améliorer `soft` quand les scores ne sont pas comparables.
- Introduire enregistrement d'expériences (log, MLFlow) pour reproductibilité.

[DETAILED_EXPLANATIONS_END]

---

Fin du fichier `docs/ALL.md`. Si tu veux, je peux :
- ajouter un sommaire cliquable (liens markdown internes),
- générer un `requirements.txt` et lancer les checks automatiques,
- créer un `README.md` résumé et un `docs/TOC.md`.

---

21. Description détaillée par fichier

Cette section décrit fichier par fichier le rôle, les fonctions principales, les paramètres importants et les points d'attention. Elle vise à permettre à un nouveau développeur de comprendre rapidement chaque artefact du dépôt.

21.1 `classes/MyVotingPyOD.py`
- Rôle: classifier d'ensemble compatible PyOD (hard / soft / S&F).
- Fonctions/classes clés: `MyVotingPyOD.__init__`, `fit`, `predict`, `decision_function`, `_fit_SF`.
- Entrées/Sorties: prend `estimators` (list[(name, estimator)]), `voting` (str), retourne `predict(X)` binaire et `decision_function(X)` score continu.
- Points d'attention: normalisation par rang percentile, seuil soft basé sur `contamination`, clipping des poids S&F pour éviter overflow.

21.2 `classes/MyVotingOutlier.py`
- Rôle: variante orientée outlier/detection (API similaire à MyVotingClassifier mais adaptée aux sorties PyOD).
- Usage: alternative pour prototypes; vérifier les différences d'API avec `MyVotingPyOD` si vous remplacez.

21.3 `classes/MyVotingClassifier.py`
- Rôle: Voting classifier classique (sklearn-like) adapté à certains usages internes.
- Notes: utile si vous avez des classifieurs probabilistes classiques; ne pas confondre avec `MyVotingPyOD`.

21.4 `classes/MyT_SNE.py`
- Rôle: wrapper/outil pour calculer et présenter des projections t-SNE interactives.
- Fonctions: initialisation t-SNE, rendu, sauvegarde d'images.
- Points: attention aux performances (t-SNE est coûteux) et au param `perplexity`.

21.5 `classes/utils/ChargeurDonnees.py` et `ChargeurDonneesPourOutlier.py`
- Rôle: fonctions utilitaires pour charger et préparer jeux de données (formats `.npz`, csv, prétraitements).
- Fonctions clés: `charger_dataset_adbench`, normalisation simple et retours `X,y`.
- Points: respecter le format attendu (`X` numpy array, `y` binaire) pour compatibilité avec le reste du code.

21.6 `classes/utils/Normaliseur.py`
- Rôle: utilitaires pour normaliser features et scores (RobustScaler wrapper, percentile transforms).
- Usage: utilisé avant l'optimisation/hyperparam search pour stabiliser entrées.

21.7 `classes/utils/Trouve_params.py` et `classes/utils/Trouve_params_pyod.py`
- Rôle: implémentations de recherche d'hyperparamètres. `Trouve_params_pyod` ciblé sur PyOD.
- API: construire l'instance avec `X,y,cv,scoring`, appeler `trouve_params(model)` qui retourne un modèle entraîné sur `self.X`.
- Détails: grilles spécifiques par modèle, exceptions capturées pendant l'entraînement des combinaisons.
- Recommandation: restreindre les grilles pour exécution rapide lors du prototypage.

21.8 `classes/utils/Borda.py`
- Rôle: implémentation du comptage Borda pour agrégation de rangs/scores.
- Utilisation: alternatif au vote majoritaire quand on veut agréger rangs plutôt que votes binaires.

21.9 `classes/utils/Evaluateur.py`
- Rôle: helpers pour calculer métriques et construire rapports (matrices de confusion, courbes PR/ROC, etc.).
- Fonctions utiles: `evaluer_modele`, `evaluer_ensemble`, export CSV/plots.

21.10 `classes/utils/download_adbench.py` et `download_adbench_all.py`
- Rôle: scripts utilitaires pour récupérer les datasets ADBench depuis un dépôt distant.
- A utiliser pour (re)peupler `data/adbench/` si nécessaire.

21.11 `streamlit_app/Hub.py`
- Rôle: page centrale de navigation pour l'application Streamlit (regroupe plusieurs sous-apps).
- Points: ne pas oublier d'ouvrir `streamlit run streamlit_app/Hub.py` pour accéder à l'interface.

21.12 `streamlit_app/app_benchmark_adbench.py`
- Rôle: UI principale pour benchmark ADBench (déjà détaillé plus haut). Contient orchestration complète.
- Points: module principal à modifier pour expérience utilisateur (par ex. ajouter export CSV, sauvegarde de logs).

21.13 `streamlit_app/app_explore_adbench.py` et `app_anomalie.py` et `app.py`
- Rôle: sous-apps de l'UI pour explorer datasets, visualiser anomalies, et petits outils interactifs.
- Suggestion: utiliser `app_explore_adbench.py` pour visualisations rapides et debug.

21.14 `streamlit_app/cache_evaluations.joblib`
- Rôle: cache des évaluations pour accélérer l'UI. Peut être supprimé si corrompu (sera regénéré).

21.15 `tests/` (tous les scripts)
- Rôle: scripts de tests et d'évaluation. Liste importante:
   - `test_adbench_pyod.py` : scénario d'évaluation ADBench end-to-end.
   - `test_myVotingClassifier.py`, `pyod_test.py`, `mnist_test.py`, `sklearn_test.py`, etc.: tests unitaires et d'intégration.
- Exécution: `python -m pytest tests/` (ou `pytest` si disponible).

21.16 `md_doc_perso/` (docs individuels)
- Contient des pages markdown pédagogiques: `cross_validation.md`, `pyod.md`, `source_et_faits.md`, `precision_recall.md`, etc.
- Ces pages expliquent concepts isolés et peuvent être reprises dans `docs/ALL.md` ou converties en pages HTML.

21.17 `modeles_sauvegardes/`
- Rôle: sauvegarde de modèles entraînés au format `joblib`. Permet reload direct sans ré-entrainement.
- Convention: nommage encode configuration (classe, contamination, pourcentage d'entrainement, etc.).

21.18 `requirements.txt`
- Rôle: fichier listant dépendances. Si manquant ou incomplet, exécuter `pip freeze` après installation pour le reconstruire.

21.19 `TODO` et fichiers de portfolio
- `TODO`: notes personnelles du stagiaire, indique tâches restantes. À tenir à jour.
- `portfolio_du_stagiaire/`: documents personnels, non nécessaires à l'exécution du code.

22. Checklist de nettoyage (actions recommandées)

- Exécuter `black` + `isort` sur le projet:

```bash
pip install black isort
black .
isort .
```

- Lancer les vérifications de syntaxe et tests:

```bash
python -m py_compile classes/*.py streamlit_app/*.py classes/utils/*.py
pytest -q
```

- Vérifier les performances & temps: exécuter sur un dataset petit (ex: `18_Ionosphere.npz`) puis monter.

23. Propositions d'améliorations futures

- Ajouter enregistrement d'expériences (MLFlow, Sacred) pour traçabilité.
- Ajouter options d'export CSV/JSON depuis l'UI.
- Ajouter des tests unitaires supplémentaires autour de `MyVotingPyOD._fit_SF` et de la normalisation des scores.

---


