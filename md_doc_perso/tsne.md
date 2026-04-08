# t-SNE 

Le **t-SNE** (t-distributed Stochastic Neighbor Embedding) n'est pas juste un outil magique qui "crée des clusters". C'est un algorithme mathématique complexe de réduction de dimensionnalité. Son but est de conserver au maximum les structures locales (les voisinages) des données de haute dimension (ex: 784 pixels) lorsqu'elles sont projetées en basse dimension (ex: 2D ou 3D).

## L'Architecture de l'algorithme en 5 ÉTAPES

l'algorithme suit une procédure stricte :

### Étape 1 : Calcul des distances (Haute dimension)
On calcule la distance (généralement euclidienne au carré) entre chaque paire de points dans l'espace d'origine.

### Étape 2 : Transformer les distances en probabilités ($P$)
Les distances brutes sont converties en "probabilités conditionnelles" de voisinage. 
- Mathématiquement : on évalue la probabilité qu'un point $A$ choisisse $B$ comme son voisin.
- **Perplexité (Recherche dichotomique)** : Pour s'adapter à la densité des données, l'algorithme fait une recherche précise pour ajuster la variance autour de chaque point. Un point dans une zone dense aura un rayon de voisinage plus petit qu'un point isolé.

### Étape 3 : Initialisation en basse dimension ($Y$)
On place les points dans le nouvel espace 2D/3D.
- Initialisation aléatoire : distribution normale standard (`1e-4 * N(0,1)`).
- Initialisation **PCA** : Souvent préférée car elle donne une première direction logique en utilisant l'Analyse en Composantes Principales, ce qui accélère la convergence.

### Étape 4 : Probabilités en basse dimension ($Q$)
On recalcule la probabilité que les points soient voisins, mais cette fois dans l'espace réduit (2D).
- **Le secret du "t" de t-SNE** : Au lieu d'utiliser une loi normale comme à l'étape 2, on utilise la **loi de Student** ($q \approx \frac{1}{1+d^2}$). 
- **Pourquoi ? Le Crowding Problem** : En haute dimension, on a beaucoup d'espace. En 2D, l'espace manque. La loi de Student possède des "queues lourdes", ce qui permet de repousser légèrement les points éloignés pour éviter que tout soit au centre.

### Étape 5 : Optimisation (Divergence KL)
Le but est que la carte 2D ($Q$) ressemble exactement à l'espace original ($P$).
On utilise la descente de gradient sur la **Divergence de Kullback-Leibler** (KL Divergence).
- Si $P > Q$ : Les points étaient proches en vrai mais sont loin en 2D => **Force d'attraction**.
- Si $P < Q$ : Les points étaient loin en vrai mais sont proches en 2D => **Force de répulsion**.
On met à jour la position des points avec des **gains adaptatifs** et du **momentum** pour stabiliser la descente.

---

## Paramètres fondamentaux

- **`perplexity`** (souvent 30) : Définit la taille du voisinage local.
  - Trop bas $\to$ Les points ne voient que leurs très proches voisins, la carte devient du bruit sans structure.
  - Trop haut $\to$ Le voisinage englobe des groupes différents, tout fusionne en une grosse boule.
- **Early Exaggeration** : Pendant les 250 premières itérations, on multiplie les probabilités $P$ par 12. Cela force les clusters à s'éloigner artificiellement les uns des autres pour créer de l'espace sur la carte avant de les laisser se stabiliser.
- **Algorithme (`exact` vs `barnes_hut`)** : Le calcul standard (exact) prend un temps fou ($O(N^2)$). C'est pour ça que la version de `scikit-learn` utilise `barnes_hut` ($O(N \log N)$) en interne pour accélérer.

## Les Limites 
- **Les axes (X, Y) n'ont pas d'unité** : L'algorithme se concentre sur les voisinages locaux. Si le cluster bleu est en haut à gauche et le rouge en bas à droite, la grande distance qui les sépare n'a pas vraiment de sens géographique robuste.
- **La taille des clusters** : Un cluster visuellement très étendu en t-SNE ne veut pas dire qu'il l'est dans la réalité. La perplexité égalise la densité.