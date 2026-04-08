

## MyVotingClassifier.py
- C'est quoi : Notre propre version du classifieur par vote à majorité (Voting Classifier), recodé avec Numpy.
- À quoi ça sert : Au lieu de ne faire confiance qu'à un seul algorithme (comme un arbre de décision ou un KNN), il fait voter plusieurs modèles de base simultanément et renvoie la réponse qui a eu le plus de voix. "L'union fait la force".
- **Paramètres :**
  - `estimators` (list) : Liste des modèles à combiner sous forme de tuples, par exemple `[('knn', knn), ('rf', forest)]`.
  - `voting` (str) : Mode de vote. `'hard'` (vote à la majorité), `'soft'` (moyenne des probabilités), ou `'S&F'` (Source & Faits) qui évalue dynamiquement la fiabilité des modèles.
  - `weights` (list) : Poids manuels des modèles.
  - `verbose` (bool) : Si True, affiche les étapes et les temps d'entraînement.
- **Méthodes :**
  - `fit(X, y, auto_optimize='non')` : Encode les labels et entraîne les modèles. En mode `'S&F'`, calcule les scores de fiabilité des modèles. Permet aussi d'optimiser automatiquement les modèles avec `auto_optimize`.
  - `predict(X)` : Fait prédire chaque modèle sur les données `X` et renvoie la classe majoritaire (selon les poids).
  - `predict_proba(X)` : Valide seulement en vote "soft", renvoie la probabilité moyenne accordée à chaque classe.
  - `transform(X)` : Renvoie les prédictions brutes (ou probabilités) de tous les modèles sans procéder au vote final.
  - `score(X, y)` : Calcule et renvoie la précision finale (accuracy) en comparant les prédictions au vrai résultat `y`.
  - `score_confiance(X)` : fournit pour chaque donnée de `X` un vrai "dépouillement des votes". Précise si la victoire était unanime ou très disputée et offre une analyse d'explicabilité introuvable de manière native dans Scikit-Learn.
- **Exemple d'utilisation :**
```python
from classes.MyVotingClassifier import MyVotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Créer les modèles individuels
m1 = DecisionTreeClassifier(max_depth=3)
m2 = KNeighborsClassifier(n_neighbors=3)

# Les regrouper dans notre Voting Classifier (vote classique 'hard')
mon_vote = MyVotingClassifier(
    estimators=[('arbre', m1), ('knn', m2)],
    voting='hard',
    weights=[1.0, 2.0] #  KNN x2 
)

# Entraîner et prédire
mon_vote.fit(X_train, y_train)
predictions = mon_vote.predict(X_test)
```

## MyT_SNE.py
- C'est quoi : L'implémentation pédagogique de zéro du t-SNE (t-distributed Stochastic Neighbor Embedding).
- À quoi ça sert : C'est une technique de réduction de dimension non-supervisée très pointue. Elle permet d'aplatir des données ultra complexes (haute dimension) vers du 2D pour pouvoir les afficher et repérer visuellement des grappes (clusters) de données similaires.
- **Paramètres :**
  - `n_components` (int) : Dimension de sortie (habituellement 2 pour des graphes 2D).
  - `perplexity` (float) : La "perplexité", c'est la jauge sur le nombre de voisins considérés pour chaque point (par défaut 30.0). Plus elle est haute, plus l'algorithme prend en compte une vue d'ensemble.
  - `learning_rate` (float) : Vitesse d'apprentissage de la descente de gradient (par défaut 200.0).
  - `max_iter` (int) : Nombre maximum d'itérations pour placer les points au mieux (par défaut 1000).
- **Méthodes :**
  - `fit_transform(X)` : Fonction centrale. Prends les données X, lance l'algorithme complet en 5 étapes, et renvoie leurs nouvelles coordonnées 2D optimisées.
  - `fit(X)` : Version alternative de `fit_transform(X)` classique en Machine Learning.
  - `afficher(X=None, y=None, afficher_score=False, save_path=None)` : Génère un graphique Matplotlib de l'espace 2D avec des couleurs par classe. Permet d'exporter l'image au format PNG via `save_path`. Si `afficher_score` est actif, calcule et affiche la Trustworthiness et le score de Silhouette dans le titre.
  - `score_voisinage(X, n_neighbors=5)` : Calcule la Trustworthiness (Confiance). Note si deux points qui étaient amis en haute dimension sont bien restés amis sur le graphique 2D final pour vérifier la qualité.
  - `calculer_silhouette_score(y)` : Évalue la densité et la séparation géographique des classes (clusters) créées en 2D. Contrairement à la Trustworthiness qui regarde l'origine en N-Dimension, cette méthode évalue uniquement la clarté visuelle et spatiale des amas obtenus.
- **Exemple d'utilisation :**
```python
from classes.MyT_SNE import MyTSNE

# Initialiser le t-SNE avec initialisation PCA pour plus de stabilité
tsne = MyTSNE(n_components=2, perplexity=30.0, max_iter=500, init='pca')

# Réduire les données (passer de par ex. 64 dimensions à 2)
donnees_2d = tsne.fit_transform(X_scaled)

# Afficher le rendu avec la classe cible pour constater les regroupements
tsne.afficher(X=X_scaled, y=y_cibles, afficher_score=True)
```

## utils/Normaliseur.py
- C'est quoi : Une classe pour simplifier la modification de l'échelle des données.
- À quoi ça sert : Les algorithmes de Machine Learning fonctionnent souvent mieux quand les valeurs sont dans des petites plages. Cette classe encapsule plusieurs méthodes de scikit-learn pour mettre facilement à l'échelle.
- **Paramètres :**
  - `methode` (str) : Choisit la méthode à appliquer. Peut être `'minmax'`, `'standard'`, `'robust'`, `'maxabs'`, ou `'auto'` (détecte automatiquement la méthode la plus adaptée à la distribution).
- **Méthodes :**
  - `deviner_meilleure_methode(X)` : Méthode statique employée par le mode `'auto'`. Analyse mathématiquement les percentiles de tes données (P99, P1) et les valeurs extrêmes pour choisir intelligemment entre minmax, robust, ou standard.
  - `fit(X)` : Analyse les données X et calcule les échelles nécessaires.
  - `transform(X)` : Applique la transformation mathématique sur X.
  - `fit_transform(X)` : Fait l'analyse `fit` puis applique la transformation `transform` en une seule étape.
- **Exemple d'utilisation :**
```python
from classes.Normaliseur import Normaliseur

# Création du normaliseur en mode StandardScaler
norm = Normaliseur(methode='standard')

# Ajuster et transformer les données d'entraînement
X_train_clean = norm.fit_transform(X_train)

# Transformer uniquement les données de test (basé sur l'entraînement)
X_test_clean = norm.transform(X_test)
```

## utils/Trouve_params.py
- C'est quoi : Une classe pour trouver automatiquement les meilleurs réglages (hyperparamètres) pour un modèle.
- À quoi ça sert : Elle connaît déjà en interne un dictionnaire des réglages standards pertinents à essayer pour chaque grand type d'algorithme (SVC, KNN, Random Forest...). On lui donne un modèle, et elle cherche le combo parfait.
- **Paramètres :**
  - `X, y` : Les données d'entraînement.
  - `cv` (int) : Le nombre de sous-découpages (plis) pour la validation croisée lors des tests (par défaut 3).
  - `scoring` (str) : La méthode pour évaluer quelle configuration est gagnante (ex: `'f1_macro'`).
  - `n_jobs`, `verbose` : Paramètres pour utiliser tous les coeurs du processeur (`n_jobs=-1`) et afficher les messages dans la console (`verbose=2`).
- **Méthodes :**
  - `trouver_grille(model)` : Méthode interne automatique qui détecte le type du modèle et charge la grille de tests correspondante (optimisée récemment pour supporter les problèmes multi-classes via par ex. le solver `saga` sur LogisticRegression).
  - `trouve_params(model)` : Effectue une recherche exhaustive (GridSearchCV) de TOUTES les combinaisons possibles de la grille. Très précis mais lent. Renvoie le modèle avec les meilleurs réglages.
  - `trouve_params_rapide(model, n_iter)` : Effectue une recherche aléatoire (RandomizedSearchCV) qui pioche uniquement `n_iter` combinaisons au hasard. Plus rapide pour les gros algorithmes.
- **Exemple d'utilisation :**
```python
from classes.Trouve_params import Trouve_params
from sklearn.svm import SVC

# On a un modèle brut, on ne sait pas quoi mettre comme paramètres
modele_brut = SVC()

# On instancie notre chercheur de paramètres
chercheur = Trouve_params(X_train, y_train, cv=5, scoring='accuracy')

# Trouve la version parfaitement réglée du modèle
meilleur_modele = chercheur.trouve_params(modele_brut)

# On peut l'utiliser directement !
meilleur_modele.predict(X_test)
```

## utils/Evaluateur.py
- C'est quoi : Une classe qui s'occupe de calculer et d'afficher les scores finaux.
- À quoi ça sert : Elle évite d'importer et de retaper la dizaine de formules de scikit-learn à chaque test. Capable de gèrer toute seule l'évaluation simple ou l'évaluation solide par validation croisée.
- **Paramètres :**
  - `model` : Le modèle entraîné à évaluer.
  - `X, y` : Les données (de test si évaluation classique, globales si validation croisée).
  - `cross_val` (bool) : `True` pour enclencher une validation croisée, `False` pour un test classique.
  - `cv_folds` (int) : Nombre de plis si `cross_val` est activée (par défaut 5).
- **Méthodes :**
  - `evaluate()` : Lance la boucle d'évaluation. Elle appelle toute seule `_evaluate_cv` ou `_evaluate_standard` selon vos paramètres initiaux, prépare Accuracy, Précision, Recall, F1-Score et la matrice de confusion.
  - `print_metrics(nom_modele=None)` : Affiche joliment les scores globaux et un rapport de classification dans la console.
  - `plot_confusion_matrix(save_path=None)` : Génère et affiche la figure (Matplotlib) de la Matrice de confusion pour voir visuellement là où l'algorithme s'est trompé. Option pour enregistrer sur le disque.
- **Exemple d'utilisation :**
```python
from classes.Evaluateur import Evaluateur

# Modèle déjà entraîné
mon_modele.fit(X_train, y_train)

# On initie l'évaluateur en mode Classique (Train/Test)
eval_standard = Evaluateur(mon_modele, X_test, y_test, cross_val=False)

# On compile les résultats
eval_standard.evaluate()

# On affiche les métriques dans le terminal
eval_standard.print_metrics("Mon super modèle")

# On dessine la Matrice de confusion (heatmap)
eval_standard.plot_confusion_matrix()
```

## utils/ChargeurDonnees.py
- C'est quoi : Un assistant pour formater et simplifier l'importation de jeux de données, notamment pour l'interface web.
- À quoi ça sert : Scikit-Learn offre plusieurs dizaines de manières d'importer des données (`fetch_openml`, `load_digits`, `make_moons`...). Cette classe statique centralise tout. Elle garantit qu'on récupérera toujours les données harmonisées (`X`, `y`) et les noms des colonnes via un seul appel unifié.
- **Méthodes :**
  - `lister_datasets_scikit()` : Récupère une liste textuelle des datasets disponibles (ex: 'Iris', 'MNIST', 'Lunes'). Idéal pour peupler un `selectbox` Streamlit.
  - `charger_scikit(nom_choisi)` : Le coeur de la classe. Elle appelle la bonne fonction en interne, gère les conversions de DataFrame Pandas vers Numpy Arrays si besoin, et renvoie un tuple directement exploitable : `X`, `y`, `noms_features`, `noms_classes`.
- **Exemple d'utilisation :**
```python
from classes.utils.ChargeurDonnees import ChargeurDonnees

# Demander la liste des options pour un menu déroulant
options_menu = ChargeurDonnees.lister_datasets_scikit()

# L'utilisateur choisit 'Wine'
X, y, colonnes, classes = ChargeurDonnees.charger_scikit("Wine (Classification de vins)")
```
