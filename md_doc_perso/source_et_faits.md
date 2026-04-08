# Algorithme Source & Faits


## Comment ça marche

1. **Le point de départ (Égalité)** : 
   - Au début, on ne sait pas qui est fort ou nul. 
   - Tous les modèles (sources) ont un **poids de 1.0**.

2. **Le Vote Ponderé** :
   - Les modèles votent pour une prédiction.
   - On additionne les poids de ceux qui sont d'accord. Le choix avec le plus gros score gagne (c'est notre "Fait" provisoire).
   - *(Au 1er tour, comme tout le monde vaut 1.0, c'est juste un vote de majorité absolue).*

3. **Mise à jour des poids** :
   - Maintenant qu'on a notre "Vérité", on regarde qui avait deviné juste.
   - Modèle qui a souvent raison = Son poids **augmente**.
   - Modèle qui a souvent tort = Son poids **baisse** .

4. **On recommence** :
   - On refait l'Étape 2, mais cette fois avec les **nouveaux poids**. Les bons modèles ont maintenant le pouvoir de contredire la masse.
   - **Quand est-ce qu'on s'arrête ?** Quand les poids ne bougent presque plus d'un tour à l'autre (`delta < epsilon`).

---

## Ce qu'il faut en code Python

- **Une matrice $(N_{donnees}, N_{modeles})$** avec TOUTES les prédictions calculées une bonne fois pour toutes (pour gagner du temps).
- **Un tableau de poids** (ex: `current_weights = [1.0, 1.0, 1.0...]`).
- **Une boucle `while`** mathématique très simple :

```python
# La logique 
while ecart_des_poids > 0.001 et nombre_tours < 100:
    # 1. Obtenir la majorité selon le poids de chacun
    predictions_groupe = vote_pondere(matrice_predictions, current_weights)
    
    # 2. Noter les modèles selon la majorité et changer leur poids
    current_weights = recalculer_poids(matrice_predictions, predictions_groupe)
```