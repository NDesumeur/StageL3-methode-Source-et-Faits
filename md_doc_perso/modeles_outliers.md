# Les Modèles de Détection d'Anomalies (Outliers)

La détection d'anomalies consiste à identifier des données rares ou suspectes qui s'écartent considérablement de la majorité des autres données (qu'on appelle données "normales"). 

Dans `scikit-learn`, la convention est toujours la même pour les sorties de ces modèles de détection d'outliers :
* **`1`** : Donnée normale (Inlier)
* **`-1`** : Anomalie (Outlier)

Voici l'explication des 3 modèles couramment utilisés que nous avons implémentés.

---

## 1. Isolation Forest (IF)
**Concept :** Les anomalies sont "peu nombreuses et différentes". Elles sont donc plus faciles à isoler que les données normales.

**Comment ça marche ?** 
Basé sur le même principe que les Random Forests (Forêts Aléatoires), cet algorithme construit plusieurs arbres de décision de façon purement aléatoire. 
Pour chaque arbre, il choisit une caractéristique au hasard et coupe les données au hasard jusqu'à isoler chaque point individuellement. 
* Comme une anomalie est très différente des autres, elle sera isolée en **très peu de coupes** (elle se trouvera près de la racine de l'arbre).
* Une donnée normale, très ressemblante aux autres, nécessitera **beaucoup de coupes** pour être séparée de ses voisines.
L'algorithme fait la moyenne du nombre de coupes nécessaires sur tous les arbres. Si ce nombre est petit, c'est une anomalie.

**Avantages :** 
Très rapide, consomme peu de mémoire, et fonctionne extrêmement bien sur les jeux de données avec beaucoup de dimensions (comme les images).

---

## 2. Local Outlier Factor (LOF)
**Concept :** Une anomalie est un point qui se trouve dans une zone beaucoup moins "dense" que ses propres voisins.

**Comment ça marche ?**
Il s'agit d'un algorithme basé sur la densité et sur les plus proches voisins (comme le KNN). 
Le LOF calcule la "densité locale" d'un point (à quel point il est entouré de proches voisins) et la compare à la densité locale de ses $k$ voisins. 
* Si la densité du point est similaire à celle de ses voisins, c'est un point normal (même score).
* Si sa densité est beaucoup **plus faible** que celle de ses voisins (il est isolé par rapport au groupe à côté de lui), son score s'éloigne de 1 et il est classé comme anomalie.

**Avantages :**
Excellent pour détecter des anomalies **locales**. Contrairement à d'autres méthodes qui cherchent des anomalies globales, le LOF peut repérer une donnée anormale même si elle est proche d'un groupe dense, tant que sa propre densité est relativement plus faible.
*(Note : Dans sklearn, pour utiliser `predict` sur de nouvelles données avec LOF, il faut activer le paramètre `novelty=True`).*

---

## 3. Elliptic Envelope (EE)
**Concept :** Les données normales suivent une distribution de probabilité classique (courbe en cloche / distribution Gaussienne).

**Comment ça marche ?**
C'est une approche purement mathématique et statistique. L'Elliptic Envelope va supposer que les données normales forment un gros "nuage" central. L'algorithme va dessiner une **ellipse** (une sorte d'ovale en plusieurs dimensions) qui englobe la majorité de cette distribution normale.
Toutes les données qui se retrouvent à l'intérieur de l'ellipse sont considérées comme normales. Celles qui tombent trop à l'extérieur sont des anomalies. L'algorithme est conçu pour ignorer les valeurs extrêmes pendant qu'il dessine son ellipse (covariance robuste).

**Limites avec les images (les matrices singulières) :**
Il fonctionne mal si les données ne respectent pas une forme de nuage classique ou s'il y a trop de valeurs fixes (comme les coins noirs d'une image MNIST où le pixel vaut tout le temps `0`). Dans ce cas, la variance est nulle, l'ellipse "s'aplatit" complètement et les mathématiques de l'algorithme "cassent" (matrice singulière). C'est pour ça qu'on ajoute souvent un micro-bruit (`np.random.normal`) pour forcer une infime variance.