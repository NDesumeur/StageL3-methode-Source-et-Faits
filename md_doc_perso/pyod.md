# PyOD (Python Outlier Detection) - Notes rapides

C'est comme Scikit-Learn, ça s'utilise pareil (`fit`, `predict`), mais c'est **100% fait pour trouver des anomalies**.

## 1. Le Score d'Anomalie
Au lieu de juste dire "Normal" ou "Anomalie", PyOD donne une note à chaque ligne de données. 
Plus la note (le score) est haute, plus la donnée est bizzare. Ça permet de trier les pires cas en premier

## 2. Le Seuil 
Souvent on dit à PyOD : "Je pense qu'il y a 5% de tricheurs dans mes données" (`contamination=0.05`). 
Il calcule tous les scores, et il coupe à 5%. Les 5% qui ont le pire score sont déclarés comme anomalies.

## 3. Comment il repère une anomalie ? 
- **Par la distance (KNN, etc.)** 
- **Par la densité (LOF, etc.)** 
- **Par l'isolation (Isolation Forest)** 
