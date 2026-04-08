# Cross-Validation (Notes rapides)

- **C'est quoi ?** 
  Découper les données pour tester un modèle sans tricher. 
- **Le K-Fold (ex: K=5)** :
  1. On met un bout de côté (le Test final).
  2. On coupe le reste en 5 morceaux.
  3. Le modèle s'entraîne sur 4 morceaux, passe l'exam sur le 5ème.
  4. Il recommence 5 fois en changeant le morceau de l'exam.
  5. A la fin -> on fait la moyenne des 5 notes.

## Stratified K-Fold (K-Fold intelligent)
Mieux que le découpage au hasard Ça garde les proportions (ex toujours 20% d'anomalies par morceau) pour pas que le modèle s'entraîne sur un bloc sans aucune anomalie.

```python
from sklearn.model_selection import StratifiedKFold
# ... (le code coupe proprement en gardant les proportions)
# puis boucle sur les 5 blocs et calcule la moyenne des scores
```

## Résultats 
- Ex de scores: 0.75, 1.00, 0.86, 0.60, 0.73
- Moyenne = 0.79 -> C'est le score fiable du modèle. 
