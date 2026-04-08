# Précision, Recall, F1-Score

Quand on fait de la détection d'anomalies (où 99% des données sont "normales"), la métrique "Accuracy" (Précision globale) sert à rien : si je dis toujours "Tout est normal" sans réfléchir, j'ai 99% de réussite, mais mon modèle peut être mauvais.

Il faut se baser là-dessus :

## 1. La Matrice de Confusion (Les 4 Cas)
- **Vrai Positif (TP)** : Alerte ! C'est vraiment une anomalie. (BRAVO)
- **Faux Positif (FP)** : Alerte ! Mais en fait c'est normal. (FAUSSE ALARME)
- **Faux Négatif (FN)** : "Tout va bien", alors que c'était une anomalie. (LE PIRE, on rate un tricheur)
- **Vrai Négatif (TN)** : "Tout va bien", c'est normal. (Classique)

## 2. La Précision 
Formule : `TP / (TP + FP)`
- Si le modèle dit "Anomalie", à quel point il a raison ? 
- Plus c'est haut, moins y'a de fausses alarmes. Mais peut-être qu'on loupe plein d'anomalies.

## 3. Le Recall
Formule : `TP / (TP + FN)`
- Sur TOUTES les vraies anomalies qui existent, on en a attrapé combien ?
- Plus c'est haut, moins il y a d'anomalies non détectées. Mais souvent, le détecteur fait beaucoup de fausses alarmes.

## 4. Le F1-Score 
- C'est la moyenne ("harmonique") entre la Précision et le Recall.
- Idéal si on veut un équilibre, ni trop (FP), ni trop (FN).
- Super indicateur pour comparer la vraie fiabilité des modeles sur un jeu de données.

```python
# Code Python rapide
from sklearn.metrics import classification_report

# 0=Normal, 1=Anomalie
vrais_res = [0, 0, 0, 1, 1, 1, 0, 0]
predictions = [0, 0, 1, 1, 1, 0, 0, 0] 

# Donne direct Précision, Recall et F1-Score pour chaque classe et la moyenne 
print(classification_report(vrais_res, predictions))
```