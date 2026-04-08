import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.manifold import trustworthiness
from sklearn.manifold._utils import _binary_search_perplexity

class MyTSNE:
    """
    Implémentation algorithmique de la méthode t-SNE (t-distributed Stochastic Neighbor Embedding)

    Les différentes parties du code (Architecture en 5 étapes) :
    1. Calcul des distances en haute dimension (Espace d'origine)
    2. Transformation des distances en probabilités de voisinage P (Perplexity et recherche dichotomique)
    3. Initialisation de l'embedding Y (Espace réduit) : Aléatoire ou PCA
    4. Calcul des probabilités Q en basse dimension (Loi de Student pour régler le problème d'agglomération)
    5. Calcul du gradient de la divergence de Kullback-Leibler et optimisation (Momentum, Gains adaptatifs)
    
    Comme pour la version officielle de Scikit-Learn, nous avons choisi des valeurs par défaut pour les hyperparamètres qui sont généralement efficaces pour une grande variété de jeux de données.
    et aussi on utilise 1e-12 comme valeur minimale pour éviter les divisions par zéro ou les logarithmes de zéro, ce qui est une pratique courante dans les implémentations de t-SNE.
    Ce qui diffère par rapport à la version officielle de Scikit-Learn :
    - Algorithme de Calcul : On utilise pas Barnes-Hut mais on calcule les distances et probabilités de manière brute (O(N^2)). C'est plus lent mais plus simple à comprendre et à implémenter.
      """
    
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, max_iter=1000, random_state=None, init='pca', n_jobs=None):
        self.n_components = n_components # Nombre de dimensions de l'espace réduit 
        self.perplexity = perplexity # Contrôle de la densité locale (nombre de voisins pour le calcul des probabilités)
        self.learning_rate = learning_rate # Taux d'apprentissage pour la descente de gradient
        self.max_iter = max_iter # Nombre d'itérations pour l'optimisation
        self.random_state = random_state # Pour la reproductibilité des résultats
        self.init = init # 'random' ou 'pca' pour l'initialisation des points
        self.n_jobs = n_jobs # Pour la parallélisation
        self.embedding_ = None # Stocke les coordonnées finales après fit_transform()
        
    def get_params(self, deep=True):
        return {"n_components": self.n_components, "perplexity": self.perplexity, 
                "learning_rate": self.learning_rate, "max_iter": self.max_iter, 
                "random_state": self.random_state, "init": self.init, "n_jobs": self.n_jobs} 
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _calculer_probabilites_haute_dimension(self, distances):
        """
        Étape 2 : Transformer les distances en probabilités de voisinage P.
        Méthode : Recherche de la perplexité exacte (Binary Search) pour chaque point.
        Plutôt que d'utiliser une variance globale, on utilise un algorithme de recherche 
        dichotomique pour ajuster la variance par point selon la densité de son voisinage.
        """
        distances_propres = np.maximum(distances, 0).astype(np.float32)
        n_samples = distances_propres.shape[0]
        
        # Calcule les probabilités conditionnelles P(j|i)
        P_cond = _binary_search_perplexity(distances_propres, self.perplexity, 0)
        
        # Symétrisation globale : p_ij = (p_j|i + p_i|j) / (2N)
        P = (P_cond + P_cond.T) / (2.0 * n_samples)
        
        return np.maximum(P, 1e-12)

    def _initialiser_embedding(self, X, n_samples):
        """
        Étape 3 : Initialiser les points Y dans l'espace réduit.
        - Si init='random' : Tirage aléatoire selon une loi normale standard, mis à l'échelle par 1e-4.
        - Si init='pca' : Utilise l'Analyse en Composantes Principales (PCA) pour pré-positionner les points.
        """
        if self.init == 'pca':
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            
            # Scikit-Learn standardise souvent le PCA d'initialisation en divisant par l'écart type global
            Y = pca.fit_transform(X)
            Y = Y / np.std(Y) * 1e-4 # std np.std renvoie l'écart type global de tous les éléments de Y, on multiplie par 1e-4 pour que les points soient proches au départ, ce qui aide à la convergence
            return Y
            
        else:
            if isinstance(self.random_state, np.random.RandomState):# Si c'est déjà un RandomState, on l'utilise directement
                rng = self.random_state
            elif self.random_state is None:
                rng = np.random
            else:
                rng = np.random.RandomState(self.random_state)

            return 1e-4 * rng.standard_normal(size=(n_samples, self.n_components)) 
    

    def _calculer_probabilites_basse_dimension(self, Y):
        """
        Étape 4 : Calculer les probabilités Q en basse dimension.
        Méthode : Utilisation de la loi de Student (1 / (1 + distance^2)) 
        pour régler le problème d'agglomération (crowding problem).
        """
        # Distances euclidiennes au carré entre les points Y
        distances_Y = pairwise_distances(Y, metric='sqeuclidean', n_jobs=self.n_jobs) 

        #  Application de la loi de Student : q = 1 / (1 + distance^2) 
        inv_distances = 1.0 / (1.0 + distances_Y)
        
        # Un point n'est pas le voisin de lui-même
        np.fill_diagonal(inv_distances, 0.0)

        # Normalisation
        Q = inv_distances / np.sum(inv_distances) 
        return np.maximum(Q, 1e-12), inv_distances # np.maximum renvoie le maximum élément par élément entre Q et 1e-12, ce qui garantit que Q ne contient pas de valeurs trop petites qui pourraient causer des problèmes numériques lors du calcul du gradient ou de la divergence KL.
    

    def _calculer_gradient(self, P, Q, Y, inv_distances):
        """
        Étape 5 : Gradient de la divergence de Kullback-Leibler.
        Méthode : Force d'attraction (si P>Q) et de répulsion (si P<Q) entre les points.

        Explication de l'optimisation mathématique :
        La vraie formule du gradient pour un point i est : 
        Gradient_i = 4 * Somme_j [ (P_ij - Q_ij) * inv_dist_ij * (Y_i - Y_j) ]
        """
        # "intensite" correspond au facteur : (P_ij - Q_ij) * inv_dist_ij
        intensite = (P - Q) * inv_distances
        
        # Partie 1 de l'équation : Somme_j [ intensite_ij ]
        somme_intensite = np.sum(intensite, axis=1)
        
        # Partie 2 de l'équation : Somme_j [ intensite_ij * Y_j ]
        # - somme_intensite[:, np.newaxis] * Y correspond au côté gauche
        # - np.dot(intensite, Y) correspond au côté droit (la somme des multiplications)
        gradient = 4.0 * (somme_intensite[:, np.newaxis] * Y - np.dot(intensite, Y))
        
        return gradient

    def fit_transform(self, X):
        """ 
        Exécute l'optimisation t-SNE complète.
        Intègre une phase de "surestimation" (early exaggeration) pour forcer 
        l'écartement des clusters au début de l'entraînement.
        j'ai choisi des valeurs équivalentes à celles de la version officielle de Scikit-Learn.
        et j'ai utilisé le même fonctionnement de base car pas très évident à comprendre.
        """
        n_samples = X.shape[0]
        
        # Calcul des distances euclidiennes au carré
        distances = pairwise_distances(X, metric='sqeuclidean', n_jobs=self.n_jobs)
        
        P = self._calculer_probabilites_haute_dimension(distances) # Étape 2 : Probabilités en haute dimension
        Y = self._initialiser_embedding(X, n_samples) # Étape 3 : Initialisation de l'embedding

        # Phase de surestimation des probabilités au début on prend 12 comme dans scikit_learn 
        P_exaggerated = P * 12.0

        vitesse = np.zeros_like(Y) # Initialiser la vitesse à zéro
        gains = np.ones_like(Y) # Gains adaptatifs pour chaque dimension de chaque point

        # Pour chaque itération, on calcule les probabilités en basse dimension, le gradient, puis on ajuste les points Y en fonction du gradient et du momentum.
        
        for iteration in range(self.max_iter):
            Q, inv_distances = self._calculer_probabilites_basse_dimension(Y)
            gradient = self._calculer_gradient(P_exaggerated, Q, Y, inv_distances)

            # Ajustement du taux d'apprentissage 
            # Comme dans scikit-learn, on utilise des gains adaptatifs pour chaque dimension de chaque point.
            direction_identique = (np.sign(gradient) == np.sign(vitesse))
            gains[direction_identique] *= 0.8
            gains[direction_identique == False] += 0.2
            gains = np.clip(gains, 0.01, np.inf)

            # Application du Momentum et mise à jour des points Y
            momentum = 0.5 if iteration < 250 else 0.8
            vitesse = (momentum * vitesse) - (self.learning_rate * gains * gradient)
            Y = Y + vitesse

            if iteration == 250:
                P_exaggerated = P # Fin de la phase de surestimation

        self.embedding_ = Y
        return self.embedding_

    def fit(self, X):
        """
        Entraîne le modèle t-SNE sur les données d'entrée X.
        - X: Données d'entrée (numpy array).
        """
        self.fit_transform(X)
        return self


    def afficher(self, X=None, y=None, afficher_score=False, save_path=None, en_ligne=False):
        """
        Affiche l'embedding 2D avec une graphique matplotlib.
        - X : Optionnel, mais nécessaire si on veut afficher le score de voisinage.
        - y : Optionnel, étiquettes pour colorier les classes.
        - afficher_score : Si True, calcule et affiche la métrique Trustworthiness et Silhouette dans le titre.
        - save_path : Si fourni (ex: "rendu.png"), sauvegarde l'image en haute résolution au lieu de juste l'afficher.
        - en_ligne : Si True, n'appelle pas plt.show() pour éviter les erreurs dans Streamlit.
        """
        if self.embedding_ is None:
            raise ValueError("Erreur : Le modèle n'a pas encore calculé de coordonnées. Lancez fit_transform() d'abord.")
        
        if afficher_score and (X is None or y is None):
            raise ValueError("Erreur : Les paramètres X et y doivent être fournis à afficher() pour pouvoir calculer les scores.")

        plt.figure(figsize=(10, 8))
        
        # S'il n'y a pas d'étiquettes, on affiche tous les points en couleur unie
        if y is None:
            plt.scatter(self.embedding_[:, 0], self.embedding_[:, 1], c='#1f77b4', alpha=0.6, s=30)
        else:
            # Palette dynamique : 'tab10' est parfait pour <= 10 classes, sinon on bascule sur 'viridis'
            n_classes = len(np.unique(y))
            cmap_name = 'tab10' if n_classes <= 10 else 'viridis'
            cmap = plt.get_cmap(cmap_name, n_classes)
            
            scatter = plt.scatter(self.embedding_[:, 0], self.embedding_[:, 1], 
                                  c=y, cmap=cmap, alpha=0.7, s=40, edgecolors='white', linewidths=0.5)
           
            # Ajouter une légende de couleurs pour les classes
            cbar = plt.colorbar(scatter, ticks=np.unique(y))
            cbar.set_label("Classes / Étiquettes", rotation=270, labelpad=15)
            # Ajouter des annotations de classe au centre de chaque cluster
            for classe in np.unique(y):
                center = np.median(self.embedding_[y == classe], axis=0)
                txt = plt.text(center[0], center[1], str(classe), fontsize=14, fontweight='bold', c='black')
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

        titre = "Visualisation t-SNE de l'espace de données"
        if afficher_score:
            score_trust = self.score_voisinage(X)
            score_silhouette = self.calculer_silhouette_score(y)
            titre += f"\n(Confiance : {score_trust:.3f} | Silhouette : {score_silhouette:.3f})"
        # Personnalisation du graphique pour une meilleure lisibilité
        # On ajoute un titre avec les scores d'évaluation si demandé, et on personnalise les axes et la grille.
        plt.title(titre, fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("Dimension 1", fontsize=12)
        plt.ylabel("Dimension 2", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        # On masque les axes pour une meilleure esthétique.
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        # On utilise tight_layout pour éviter que les éléments du graphique ne soient coupés, surtout si on ajoute des titres ou des légendes.
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # On retourne la figure pour permettre une utilisation dans Streamlit ou d'autres environnements, et on affiche le graphique si on n'est pas en ligne.
        fig = plt.gcf()
        if not en_ligne:
            plt.show()
            
        return fig

    def score_voisinage(self, X, n_neighbors=5):
        """
        Calcule la 'Trustworthiness' (Confiance) de l'embedding calculé.
        C'est la métrique non-supervisée standard pour évaluer si le t-SNE a bien fonctionné. 
        Elle vérifie si les 'n_neighbors' plus proches voisins d'un point dans l'espace 
        d'origine (X) sont toujours ses voisins sur le dessin 2D (Y).
        
        Une pénalité est appliquée si des points éloignés en réalité se retrouvent 
        écrasés artificiellement ensemble sur la carte 2D (ce qu'on appelle une 'intrusion').

        Paramètres :
        - X : Données brutes de départ (haute dimension).
        - n_neighbors : Le nombre de voisins locaux à vérifier (5 par défaut).

        Retourne :
        - score : Une valeur entre 0.0 et 1.0. Un score > 0.90 indique une excellente préservation.
        """
        if self.embedding_ is None:
            raise ValueError("Erreur : Impossible de calculer le score. Il faut d'abord entraîner le modèle avec fit_transform().")
            
        try:
            score = trustworthiness(X, self.embedding_, n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        except TypeError:
            score = trustworthiness(X, self.embedding_, n_neighbors=n_neighbors)
        return score

    def calculer_silhouette_score(self, y):
        """
        Calcule le score de Silhouette des points dans l'espace 2D final.
        Contrairement à la Trustworthiness qui compare la 2D à la haute dimension,
        la Silhouette évalue concrètement si les points d'une même classe (y) forment 
        des amas d'un seul bloc bien espacés et isolés (idéal pour évaluer l'explicabilité visuelle).
        
        Retourne : Un score (idéalement proche de 1.0).
        """
        if self.embedding_ is None:
            raise ValueError("Erreur : Impossible de calculer le score. Lancez fit_transform() d'abord.")
            
        try:
            return silhouette_score(self.embedding_, y, n_jobs=self.n_jobs)
        except TypeError:
            return silhouette_score(self.embedding_, y)
