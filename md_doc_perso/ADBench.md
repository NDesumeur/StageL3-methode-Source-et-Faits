# ADBench (Anomaly Detection Benchmark) & PyOD

## 1. Qu'est-ce que ADBench ?
**ADBench** (Anomaly Detection Benchmark) est l'un des benchmarks les plus récents et complets (publié à NeurIPS 2022) dédié à l'évaluation des algorithmes de détection d'anomalies.
Il a été conçu pour standardiser la comparaison des algorithmes sur des dizaines de jeux de données, majoritairement **tabulaires**.

Pour les **données tabulaires** (votre cas), ADBench rassemble des dizaines de jeux de données classiques (souvent tirés de ODDS - Outlier Detection DataSets) avec des caractéristiques très variées :
*   Petit à grand nombre d'échantillons (de quelques centaines à des centaines de milliers).
*   Dimension faible à très haute (feature space).
*   Taux de contamination (pourcentage d'anomalies) variant de 0.01% à près de 40%.

### Pourquoi utiliser ADBench au lieu de jeux faits main ?
*   **Standardisation :** C'est le standard de l'industrie pour la littérature scientifique. 
*   **Diversité des topologies :** Anomalies locales, globales, clusterisées, etc.
*   **Intégration idéale avec PyOD :** ADBench a été littéralement pensé pour tester, entre autres, la librairie PyOD.

---

## 2. Les Algorithmes ciblés (Écosystème PyOD)
Voici les modèles exigés par votre cahier des charges que nous allons implémenter. L'avantage de PyOD est que tous ces modèles partagent la même API (`.fit()`, `.decision_function()`, `.predict()`).

| Algorithme | Famille / Concept |
| :--- | :--- |
| **iForest** | Isolation Forest (Arbres, isolation aléatoire) |
| **LOF** | Local Outlier Factor (Densité locale / Voisins) |
| **CBLOF** | Cluster-Based Local Outlier Factor (Basé sur le clustering K-Means) |
| **COF** | Connectivity-Based Outlier Factor (Variante de LOF pour les structures en ligne) |
| **HBOS** | Histogram-based Outlier Score (Rapide, hypothèse d'indépendance des features) |
| **KNN** | K-Nearest Neighbors (Distance au k-ème voisin) |
| **LODA** | Lightweight On-line Detector of Anomalies (Ensemble de projections 1D) |
| **OCSVM** | One-Class SVM (Frontière spatiale / Noyaux RBF) |
| **PCA** | Principal Component Analysis (Erreur de reconstruction) |
| **COPOD** | Copula-Based Outlier Detection (Statistique, basé sur les copules empiriques) |
| **ECOD** | Empirical Cumulative Distribution Functions (Totalement non paramétrique et très rapide) |
| **Deep SVDD** | Deep Support Vector Data Description (Réseau de neurones, apprend le centre d'une sphère) |
| *(SOF/SOS)* | *Stochastic Outlier Selection (Basé sur l'affinité probabiliste)* |

---

## 3. Plan d'architecture pour le nouveau projet

Puisque nous n'avons plus de T-SNE ou de gestion d'images pures, nous allons créer une architecture taillée pour le **benchmark pur**.

### A. Récupération des données (ADBench)
*   Il faudra télécharger les datasets tabulaires compatibles ADBench (souvent via `fetch_datasets()` ou les CSV de ODDS).
*   Prétraitement universel : Beaucoup d'algos PyOD (comme le KNN, OCSVM, DeepSVDD) exigent d'avoir des données strictement standardisées (`MinMaxScaler` ou `StandardScaler`).

### B. L'optimisateur (`Trouve_params_pyod`)
*   Refaire une classe d'optimisation (GridSearch) mais adaptée aux spécificités de PyOD. PyOD ne se comporte pas exactement comme scikit-learn avec *GridSearchCV* natif sans faire appel à un *wrapper*. On écrira un optimiseur qui évalue sur le F1-Score / ROC-AUC.

### C. L'Ensemble (Vote PyOD)
Contrairement à la prédiction pure, PyOD marche par "Scores d'anomalies" (`decision_function()`).
Notre vote devra agréger les **scores normalisés** :
*   **Vote Hard :** Majorité simple des `.predict()`.
*   **Vote Soft / AVG :** Moyenne stricte des scores d'anomalies.
*   **MAX :** Score maximum (modèle le plus pessimiste).
*   **MIN :** Score minimum (modèle le plus optimiste).
*   **S&F (Soft & Filter) :** Votre méthode maison.

### D. L'Interface (Streamlit)
*   Menu de sélection du dataset ADBench.
*   Lancement du gros calcul de grilles et de fitting.
*   Tableau **Borda** de classement des 12+ modèles ainsi que des stratégies de Vote.
*   Graphiques Radar / Courbes ROC PR pour visualiser l'avantage des ensembles sur les modèles individuels.