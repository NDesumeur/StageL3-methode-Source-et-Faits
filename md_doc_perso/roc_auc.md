# La Métrique ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

La métrique ROC-AUC est l'une des métriques les plus importantes en **détection d'anomalies** (bien plus robuste que la simple "Précision" ou le "F1-Score" quand les anomalies sont très rares).

---

## 1. Pourquoi utiliser ROC-AUC au lieu de Précision/Rappel/F1 ?

Dans la vraie vie (comme dans le dataset `10_cover`), les anomalies sont **très déséquilibrées** (parfois moins de 1% des données).
- **Le F1-Score** dépend du "seuil temporel" défini par l'algorithme. Si l'algorithme déclare qu'il y a 5% d'anomalies alors qu'il n'y en a que 1%, toutes les fausses alertes s'effondrent mathématiquement son score et le mettent presque à 0.
- **Le ROC-AUC**, lui, ne dépend d'**AUCUN seuil de coupure**. Il évalue uniquement la **capacité du modèle à bien classer les données**. Il se demande : *"Si je pioche une donnée normale au hasard, et une anomalie au hasard, est-ce que mon modèle a bien donné un score de dangerosité plus élevé à l'anomalie ?"*

C'est pour ça qu'un modèle peut avoir un mauvais F1-Score (à cause d'un mauvais seuil) mais un excellent ROC-AUC (ses "croyances" internes sont parfaitement ordonnées).

---

## 2. Comment la courbe ROC est-elle construite ?

Elle compare en permanence deux notions clés :
- **TPR (True Positive Rate)** : Le taux de "Vraies Alertes". Sur 100 anomalies réelles, combien as-tu réussi à en trouver ? Aussi appelé le **Rappel/Sensibilité**.
- **FPR (False Positive Rate)** : Le pourcentage de "Fausses Alertes". Sur 100 données parfaitement normales, combien de fois t'es-tu trompé en hurlant à l'anomalie ?

La Courbe ROC est construite en faisant varier virtuellement ton **"seuil de tolérance"** de 100% sévère jusqu'à 0% très laxiste, et on relie tous les points (FPR sur axe X, TPR sur axe Y).

---

## 3. L'Aire sous la Courbe : le fameux AUC (Area Under Curve)

L'AUC correspond simplement au pourcentage d'aire remplie en dessous de cette courbe :

* **AUC = 1.0 (100%) : Modèle Parfait.** Il trouve toutes les anomalies sans jamais faire de fausse alerte sur une valeur saine.
* **AUC = 0.5 (50%) : Pile ou Face.** Ton modèle (comme ton LOF tout à l'heure à 0.51) ne comprend rien. Il lance une pièce en l'air au hasard.
* **AUC < 0.5 : Modèle inversé.** Le modèle confond systématiquement les normes et les anomalies.

---

## 4. Pourquoi tes F1-Scores étaient "si bas" malgré un bon AUC ?

Un modèle PyOD qui renvoie `y_pred` fait lui-même "la coupure" de seuil s'il trouve que la distance est trop grande.

Imaginons que ton modèle `IForest` note chaque ligne de 0 à 100.
Ton IForest avait un excellent ROC-AUC de `0.8870`. Cela signifie que dans la vaste majorité des cas, les anomalies avaient un score de 95 et les valeurs sûres un score de 40.

**Mais où couper ?**
S'il décide à l'aveugle *"Allez, tout ce qui a plus de 50 de score c'est un outlier"*, il inclura au milieu d'anomalies réelles des centaines de valeurs normales proches de 60. Le modèle s'est "mouillé" bêtement. A la seconde où tu as plein de Faux Positif, le dénominateur de ta `précision` explose et il entraîne le `F1-score` au fond du gouffre (ici `0.08`).

C'est pourquoi, en recherche, on utilise **l'AUC-ROC** : on juge le modèle sur ses "notes" brutes (`decision_function`) et non sur ses "choix binaires" ratés (`predict`).