import numpy as np
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from sklearn.base import clone
from scipy.stats import mode
import time

from .utils.Trouve_params_pyod import Trouve_params_pyod

class MyVotingPyOD:
    """
    Voting classifier pour modèles PyOD.

    Cette classe fournit trois modes d'agrégation principaux:
    - 'hard' : vote majoritaire sur prédictions binaires (0/1). Peut utiliser des poids fixes fournis via `weights`.
    - 'soft' : moyenne des scores normalisés (percentile rank) entre estimateurs suivie d'un seuil (basé sur la contamination moyenne) pour produire une prédiction binaire.
    - 'S&F'  : Source & Faits — apprentissage de poids relatifs via prédictions out-of-fold puis vote pondéré.

    Principales méthodes:
    - `fit(X, y=None, auto_optimize='non')` : entraîne tous les estimateurs et prépare les références pour le soft; calcule les poids S&F si demandé.
    - `decision_function(X)` : retourne un score continu agrégé (utilisé par soft).
    - `predict(X)` : retourne 0/1 en fonction du mode de vote.

    Remarques d'implémentation importantes:
    - Les scores bruts de chaque estimateur sont convertis en rang percentiles calculés sur leurs distributions d'entraînement afin de rendre les échelles comparables.
    - Le seuil du soft est volontairement simple : percentile lié à la contamination moyenne des estimateurs. Cela évite la fuite d'information et l'overfitting du seuil.
    - La procédure S&F utilise une optimisation heuristique (coordinate ascent multiplicatif) sur un ensemble out-of-fold pour éviter d'optimiser directement sur le train complet.
    """
    def __init__(self, estimators: list[tuple], voting: str = 'hard', weights: list[float] = None, n_jobs: int = None, verbose: bool = False, vote_metric: str = 'accuracy', threshold_metric: str = 'accuracy'):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.vote_metric = vote_metric
        self.threshold_metric = threshold_metric
        self.estimators_ = []
        self.named_estimators_ = {}
        self.score_refs_ = []
        self.contamination = 0.1
        self.entraine = False
        self.sf_weights_ = None
        self.sf_max_iter = 50
        self.sf_epsilon = 1e-4

    def fit(self, X, y=None, auto_optimize='non', sample_size_opti=1000):
        """
        Entraîne les estimateurs fournis et prépare les mécanismes d'agrégation.

        Comportement:
        - si `auto_optimize` est 'rapide' ou 'normal' et `y` fourni => on exécute `Trouve_params_pyod` en sous-échantillon pour optimiser chaque estimateur.
        - entraîne chaque estimateur sur les données (parallélisation possible via `n_jobs`).
        - calcule `score_refs_` : pour chaque estimateur, la distribution triée des scores d'entraînement (utilisée pour la normalisation percentile).
        - si `voting` == 'S&F' : appelle `_fit_SF` pour apprendre les poids pondérés via OOF.
        - si `voting` inclut 'soft' : calcule un seuil simple basé sur la contamination moyenne.

        Paramètres:
        - `X`: ndarray (n_samples, n_features)
        - `y`: labels (optionnel mais recommandé pour S&F)
        - `auto_optimize`: 'non'|'rapide'|'normal'
        - `sample_size_opti`: taille max du sous-échantillon pour l'optimiseur
        """
        # 1. Optimisation éventuelle du modèle
        if auto_optimize in ['rapide', 'normal'] and y is not None:
            indices = np.random.choice(len(X), min(sample_size_opti, len(X)), replace=False)
            X_opti, y_opti = X[indices], y[indices]
            
            chercheur = Trouve_params_pyod(X_opti, y_opti, cv=3, scoring='f1', verbose=0)
            nouveaux_estimateurs = []
            for name, classifier in self.estimators:
                try:
                    modele_opti = chercheur.trouve_params(classifier)
                    nouveaux_estimateurs.append((name, clone(modele_opti)))
                except Exception:
                    nouveaux_estimateurs.append((name, classifier))
            self.estimators = nouveaux_estimateurs

        # 2. Version simple: on entraîne sur tout X.
        X_train_fit = X
        y_train_fit = y

        # 3. Entraînement en parallèle sur le training set
        taches_paralleles = []
        for name, clf in self.estimators:
            modele_copie = clone(clf)
            tache = delayed(modele_copie.fit)(X_train_fit)
            taches_paralleles.append(tache)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(taches_paralleles)
        
        for i in range(len(self.estimators)):
            name = self.estimators[i][0]
            self.named_estimators_[name] = self.estimators_[i]
            
        # 4. Références de scores (rang percentile) pour le mode soft (calculées sur X_train_fit)
        self.score_refs_ = []
        contaminations = []
        for modele in self.estimators_:
            # `decision_function` retourne un score brut (échelle dépend du modèle).
            # On stocke ici la distribution triée des scores sur l'ensemble d'entraînement afin
            # de pouvoir transformer ultérieurement des scores bruts en rang percentiles (0..1).
            scores = modele.decision_function(X_train_fit).ravel()
            self.score_refs_.append(np.sort(scores))
            contaminations.append(getattr(modele, 'contamination', 0.1))
            
        self.contamination = np.mean(contaminations)
        self.entraine = True
        
        # 5. Mode Source et Faits : poids initiaux appris sur train (OOF validation sur X_train_fit)
        if self.voting in ['S&F', 'source et faits']:
            # Apprentissage des poids via procédure out-of-fold pour limiter le sur-apprentissage
            # et estimer la fiabilité relative des estimateurs.
            self._fit_SF(X_train_fit, y_train_fit)

        # 6. Soft simple: moyenne des scores puis seuil contamination sur TRAIN.
        if self.voting in ['soft', 'average']:
            train_soft_scores = self.decision_function(X_train_fit)
            self.threshold_ = np.percentile(train_soft_scores, 100 * (1 - self.contamination))
            if self.verbose:
                print(f"[MyVotingPyOD] Seuil soft simple (train): {self.threshold_:.4f}")
        else:
            self.threshold_ = None

        return self

    def decision_function(self, X):
        """
        Calcule le score combiné pour soft voting et autres variantes continues.
        Pour S&F, on n'utilise que les votes binaires, pas les scores continus.
        """
        if not self.entraine:
            raise ValueError("Modèle non entraîné.")

        probas_list = []
        for i, modele in enumerate(self.estimators_):
            raw_scores = modele.decision_function(X).ravel()
            # Normalisation robuste: percentile rank dans la distribution de scores train.
            refs = self.score_refs_[i]
            norm_score = np.searchsorted(refs, raw_scores, side='right') / len(refs)
            probas_list.append(norm_score)

        probas = np.column_stack(probas_list)
        
        # Soft simplifié: moyenne pondérée (ou simple) des scores normalisés.
        if self.voting in ['soft', 'average']:
            if self.weights is not None:
                weights = np.asarray(self.weights)
                weights = weights / weights.sum()
                scores_combines = np.average(probas, axis=1, weights=weights)
            else:
                scores_combines = np.mean(probas, axis=1)
        else:
            # Fallback continu minimal pour compatibilité.
            scores_combines = np.mean(probas, axis=1)

        return scores_combines

    def predict(self, X):
        if not self.entraine:
            raise ValueError("Modèle non entraîné.")

        # Demande des prédictions brutes (0=inlier, 1=outlier)
        predictions_matrice_list = Parallel(n_jobs=self.n_jobs)(
            delayed(estimator.predict)(X) for estimator in self.estimators_
        )
        predictions = np.column_stack(predictions_matrice_list)
        n_samples = len(X)
        
        # Défaut : si aucun poids fournis, on considère des poids égaux (=1) pour tous les estimateurs.
        n_estimators = predictions.shape[1]
        effective_weights = np.asarray(self.weights) if self.weights is not None else np.ones(n_estimators, dtype=float)

        if self.voting == 'hard':
            maj_vote = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                colonne_votes = predictions[i, :]
                gagnant = np.bincount(colonne_votes, weights=effective_weights).argmax()
                maj_vote[i] = gagnant
            return maj_vote

        elif self.voting in ['S&F', 'source et faits']:
            # S&F : Hard vote avec poids appris itérativement, pas des poids fixes F1**3
            # Les poids convergent via Truth Discovery binaire.
            maj_vote = np.zeros(n_samples, dtype=int)
            poids_sf = self.sf_weights_ if self.sf_weights_ is not None else np.ones(len(self.estimators_))
            for i in range(n_samples):
                colonne_votes = predictions[i, :]
                gagnant = np.bincount(colonne_votes, weights=poids_sf).argmax()
                maj_vote[i] = gagnant
            return maj_vote

        elif self.voting in ['soft', 'average']:
            # Soft simplifié: score moyen + seuil fixe basé contamination.
            scores = self.decision_function(X)
            return (scores > self.threshold_).astype(int)

        else:
            # Par défaut, fallback hard si stratégie inconnue. Utilise aussi `effective_weights`.
            maj_vote = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                colonne_votes = predictions[i, :]
                gagnant = np.bincount(colonne_votes, weights=effective_weights).argmax()
                maj_vote[i] = gagnant
            return maj_vote

    def _fit_SF(self, X_train, y_train=None, epsilon_arret=0.001, max_iter=20):
        """
        Algorithme Source & Faits pour PyOD.
        On construit des prédictions hors-échantillon, puis on cherche les poids qui maximisent
        une métrique de vote configurable (accuracy par défaut, F1 si demandé).
        Le meilleur vecteur trouvé est conservé pour éviter toute dégradation.
        """
        from sklearn.model_selection import StratifiedKFold, KFold
        from sklearn.metrics import f1_score, accuracy_score

        n_samples = X_train.shape[0]
        n_estimators = len(self.estimators_)
        predictions_matrice = -np.ones((n_samples, n_estimators), dtype=int)

        # Générer des prédictions out-of-fold quand on a des labels et deux classes.
        if y_train is not None and len(np.unique(y_train)) > 1:
            splitter = StratifiedKFold(n_splits=min(5, len(np.unique(y_train)) + 3), shuffle=True, random_state=42)
            for train_idx, val_idx in splitter.split(X_train, y_train):
                for m, (_, estimator) in enumerate(self.estimators):
                    modele_fold = clone(estimator)
                    modele_fold.fit(X_train[train_idx])
                    predictions_matrice[val_idx, m] = modele_fold.predict(X_train[val_idx])
        else:
            splitter = KFold(n_splits=min(5, n_samples), shuffle=True, random_state=42)
            for train_idx, val_idx in splitter.split(X_train):
                for m, (_, estimator) in enumerate(self.estimators):
                    modele_fold = clone(estimator)
                    modele_fold.fit(X_train[train_idx])
                    predictions_matrice[val_idx, m] = modele_fold.predict(X_train[val_idx])

        # Si certains points n'ont pas été couverts, on retombe sur les prédictions train.
        trous = np.any(predictions_matrice < 0, axis=1)
        if np.any(trous):
            predictions_matrice[trous] = np.column_stack([estimator.predict(X_train[trous]) for estimator in self.estimators_])

        n_estimators = predictions_matrice.shape[1]
        
        # Initialiser avec les poids F1 qui marchent déjà bien
        if self.weights is not None and len(self.weights) == n_estimators:
            fiabilite_sources = np.asarray(self.weights, dtype=float).copy()
            fiabilite_sources = fiabilite_sources / fiabilite_sources.sum()
        else:
            fiabilite_sources = np.ones(n_estimators, dtype=float) / n_estimators

        def vote_majoritaire(poids):
            pred = np.zeros(predictions_matrice.shape[0], dtype=int)
            for i in range(predictions_matrice.shape[0]):
                pred[i] = np.bincount(predictions_matrice[i, :], weights=poids).argmax()
            return pred

        def normalise(poids):
            poids = np.asarray(poids, dtype=float)
            somme = poids.sum()
            if somme <= 0:
                return np.ones_like(poids) / len(poids)
            return poids / somme

        def score_poids(poids):
            if y_train is None or len(np.unique(y_train)) <= 1:
                return -1.0
            preds = vote_majoritaire(poids)
            if self.vote_metric == 'f1':
                return f1_score(y_train, preds, zero_division=0)
            return accuracy_score(y_train, preds)

        meilleur_poids = normalise(fiabilite_sources.copy())
        meilleur_score = score_poids(meilleur_poids)

        # Optimisation locale du hard vote: on teste des variations multiplicatives sur chaque poids.
        pas = 0.35
        for iteration in range(max_iter):
            amelioration = False
            for idx in range(n_estimators):
                poids_base = meilleur_poids.copy()

                candidats = [
                    poids_base,
                    np.where(np.arange(n_estimators) == idx, np.clip(poids_base[idx] * (1.0 + pas), 1e-6, 1e3), poids_base),
                    np.where(np.arange(n_estimators) == idx, np.clip(poids_base[idx] * (1.0 - pas), 1e-6, 1e3), poids_base),
                ]

                meilleur_local = meilleur_poids
                meilleur_local_score = meilleur_score

                for candidat in candidats:
                    candidat = normalise(candidat)
                    score = score_poids(candidat)
                    if score > meilleur_local_score:
                        meilleur_local_score = score
                        meilleur_local = candidat

                if meilleur_local_score > meilleur_score:
                    meilleur_score = meilleur_local_score
                    meilleur_poids = meilleur_local
                    amelioration = True

            if not amelioration:
                pas *= 0.5
                if pas < 0.05:
                    break

        # Si on a des labels, on s'assure de ne jamais faire pire que le hard vote de base.
        if y_train is not None and len(np.unique(y_train)) > 1:
            score_hard = f1_score(y_train, vote_majoritaire(normalise(fiabilite_sources)), zero_division=0)
            if score_hard > meilleur_score:
                meilleur_score = score_hard
                meilleur_poids = normalise(fiabilite_sources)

        self.sf_weights_ = meilleur_poids if meilleur_score >= 0 else normalise(fiabilite_sources)
        
        if self.verbose:
            print(f"Poids S&F retenus : {list(np.round(self.sf_weights_, 4))}")
