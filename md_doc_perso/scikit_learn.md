# Scikit-Learn : Comment ça marche ? 

Le gros avantage de Scikit-Learn (sklearn), c'est que TOUT fonctionne de la même manière. 
Le process en 3 étapes :
1. **Choisir** : On choisi un modele (ex: `modele = DecisionTree()`).
2. **Apprendre (`fit`)** : On donne nos données pour qu'il comprenne la logique (`modele.fit(X, y)`).
3. **Utiliser (`predict` ou `transform`)** : On lui donne des nouvelles données pour deviner (`modele.predict(X_test)`).

## Les gros modules et modèles à connaitre
Voici un tour d'horizon des modules et des algorithmes qu'on utilise souvent :

- **`sklearn.linear_model`** : Les algorithmes qui tracent des droites. Cool pour séparer des données assez simples.
  - `LogisticRegression` : Le grand classique, parfait pour de la classification binaire ou simple.
  - `RidgeClassifier` / `SGDClassifier` : Bien pour gérer des gros datasets avec des pénalités pour éviter le surapprentissage.
  - `Perceptron` :  simple mais parfois efficace.
- **`sklearn.tree`** : La logique pure.
  - `DecisionTreeClassifier` : L'Arbre de Décision. Ça pose plein de questions logiques en chaîne (style "Est-ce que X > 5 ?"). Facile à expliquer (on peut le dessiner).
- **`sklearn.ensemble`** : On met plein d'algorithmes ensemble pour qu'ils votent et évitent de faire n'importe quoi.
  - `RandomForestClassifier` : Plein d'arbres de décision qui votent .
  - `ExtraTreesClassifier` : Comme la forêt aléatoire, mais avec des décisions encore plus au hasard pour être ultra-robuste.
  - `AdaBoostClassifier` : L'algorithme apprend de ses erreurs, le modèle suivant se concentre sur les erreurs du précédent.
  - `BaggingClassifier` : On lance pleins de modèles sur des bouts de données différents et on moyenne le tout.
- **`sklearn.neighbors`** : Le système de voisinage.
  - `KNeighborsClassifier (KNN)` : Ne "calcule" presque rien, il regarde juste la catégorie des "X voisins les plus proches" de ta donnée.
- **`sklearn.svm`** : Les frontières complexes.
  - `SVC` (Support Vector Classifier) : Cherche à tracer LA meilleure frontière (la plus large possible ou même tordue de fou en 3D) pour séparer les catégories.
- **`sklearn.naive_bayes`** : Les probabilités un peu naïves.
  - `GaussianNB` / `MultinomialNB` / `BernoulliNB` : Très forts pour le texte ou les calculs de probabilités rapides.
- **`sklearn.preprocessing`** : Le pressing. Pour nettoyer, mettre les chiffres à la même échelle (Normaliser) avec `StandardScaler`, `MinMaxScaler`, etc. Important si les colonnes ont des échelles très différentes 
- **`sklearn.metrics`** : C'est ici qu'on trouve `accuracy_score`, `f1_score`, etc. pour voir si le modèle est nul ou pas.

## Quand utiliser lequel ? 
il y a une carte magique sur leur site qui dit exactement quoi utiliser selon les données :
https://scikit-learn.org/stable/machine_learning_map.html