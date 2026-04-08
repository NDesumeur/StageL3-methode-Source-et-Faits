# Aide-mémoire NumPy 

NumPy = le module pour faire des maths en Python avec des gros tableaux sans faire de boucles lentes.

## 1. Créer des tableaux
- `np.array(liste)` : Convertit une liste Python normale en tableau NumPy.
- `np.zeros(shape)` / `np.ones(shape)` : Tableaux remplis de 0 ou de 1 (super pour initialiser).
- `np.zeros_like(arr)` : Copie la forme d'un tableau existant mais remplit tout de 0.
- `np.eye(N)` : Matrice diagonale de 1. ex dans le code : `np.eye(nb_classes)[predictions]`.
- `np.arange(start, stop, step)` : Comme `range()` mais en tableau.

## 2. Infos utiles
- `arr.shape` : Les dimensions (ex: 100 lignes, 2 colonnes -> `(100, 2)`).
- `arr.dtype` : Le type.

## 3. Manipuler les formes
- `arr.reshape(...)` : change le tableau dans une autre dimension.
- `arr.astype(...)` : Force le changement de type.
- `np.vstack((a, b))` / `np.hstack((a, b))` : Empile des tableaux verticalement (lignes) ou horizontalement (colonnes).
- `np.fill_diagonal(arr, 0)` : Met des 0 sur la diagonale.

## 4. Maths et Statistiques (sans boucle)
- `np.mean(arr)` : La moyenne. Marche bien avec des conditions : `np.mean(preds == vrais_labels)` te donne ta précision direct
- `np.sum(arr, axis=0)` : Ajoute tout (l'axe dit si on additionne les lignes ou colonnes).
- `np.argmax(arr)` : Te donne l'indice de la valeur la plus grande.
- `np.maximum(arr, 1e-12)` : Force un minimum pour éviter de diviser par zéro 
- `np.clip(arr, min, max)` : Bloque les valeurs pour qu'elles débordent pas.

## 5. Recherche & Tri
- `np.where(condition, si_vrai, si_faux)` : Ex: `np.where(proba > 0.5, 1, 0)`.
- `np.unique(arr)` : Liste les éléments sans doublon.
- `np.argsort(arr)` : renvoie les *indices* pour trier, pas les valeurs.

## 6. L'aléatoire (Random)
- `np.random.seed(42)` : Bloque le hasard pour avoir toujours le même résultat (indispensable pour débug).
- `np.random.choice(arr, size)` : Pioche au hasard (ex: prendre un échantillon de 1000 images).
