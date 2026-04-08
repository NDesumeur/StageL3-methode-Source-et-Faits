# Parallélisation : n_jobs & joblib (Notes rapides)

## C'est quoi `n_jobs` ?
Au lieu de faire des calculs hyper lourds 1 par 1, on demande à l'ordinateur de se diviser en plusieurs petits cerveaux pour tout faire en même temps 
- `n_jobs=1` ou `None` : L'ordi fait tout tout seul (séquentiel).
- `n_jobs=2` : On divise le boulot par 2.
- `n_jobs=-1` : L'ordinateur utilise TOUS ses processeurs à 100%.

## Comment l'implémenter
```python
from joblib import Parallel, delayed

# 1. On prépare des paquets de travail (avec 'delayed')
# = " lancer predict(X) sur CHAQUE modèle mais en attente de Parallel"
taches = [delayed(modele.predict)(X) for modele in liste_modeles]

# 2. On lance tout en même temps sur tous les coeurs (-1)
resultats = Parallel(n_jobs=-1)(taches)
```

## LE gros piège de l'entraînement (`fit`)

**Ce qu'il faut faire** : Utiliser `clone()` de Scikit-Learn pour donner un modèle 100% vierge à chaque processeur.
```python
from sklearn.base import clone

for nom, modele in liste:
    copie_neuve = clone(modele) # <- INDISPENSABLE 
    tache = delayed(copie_neuve.fit)(X, y)
    taches.append(tache)
    
# tout s'entraîne proprement en parallèle 
Parallel(n_jobs=-1)(taches)
```
