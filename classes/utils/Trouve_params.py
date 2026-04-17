from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import time

class Trouve_params:
    """
    Classe permettant de trouver automatiquement les meilleurs hyperparamÃ¨tres (GridSearch)
    selon le type de modÃ¨le qui lui est fourni.
    Elle contient en interne les grilles de recherches standard pour les modÃ¨les communs.
    """
    def __init__(self, X, y, cv=3, scoring='f1_macro', n_jobs=-1, verbose=2):
        self.X = X
        self.y = y
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
    
        self.grilles_connues = {
            SVC: {
                'C': [0.1, 1, 10], 
                'gamma': ['scale', 'auto', 0.01], 
                'kernel': ['rbf', 'linear']
            },
            OneClassSVM: {
                'nu': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0, 10.0]
            },
            KNeighborsClassifier: {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            },
            RidgeClassifier: {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            RandomForestClassifier: {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]
            },
            DecisionTreeClassifier: {
                'max_depth': [None, 10, 20, 50],
                'criterion': ['gini', 'entropy']
            },
            LogisticRegression: {
                'C': [0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'saga'],
                'max_iter': [1000, 2000] 
            },
            SGDClassifier: {
                'loss': ['log_loss', 'hinge', 'modified_huber'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': [0.0001, 0.001, 0.01],
                'max_iter': [2000, 3000] 
            },
            ExtraTreesClassifier: {
                'n_estimators': [20, 50, 100],
                'max_depth': [None, 10, 20]
            },
            AdaBoostClassifier: {
                'n_estimators': [20, 50, 100],
                'learning_rate': [0.1, 0.5, 1.0]
            },
            BaggingClassifier: {
                'n_estimators': [10, 20, 50],
                'max_samples': [0.5, 0.8, 1.0]
            },
            MultinomialNB: {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            },
            GaussianNB: {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            },
            GradientBoostingClassifier: {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5],
                'max_depth': [3, 5, 10]
            },
            HistGradientBoostingClassifier: {
                'max_iter': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.5]
            },
            LinearSVC: {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000, 2000]
            },
            IsolationForest: {
                'n_estimators': [50, 100, 200, 300],
                'max_samples': ['auto', 0.5, 0.8],
                'max_features': [1.0, 0.8],
                'contamination': ['auto', 0.01, 0.05, 0.1]
            },
            LocalOutlierFactor: {
                'n_neighbors': [5, 10, 20, 30, 50],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'contamination': ['auto', 0.01, 0.05, 0.1, 0.2],
                'novelty': [True]
            },
            EllipticEnvelope: {
                'contamination': [0.01, 0.05, 0.1, 0.2, 0.3],
                'assume_centered': [True, False],
                'support_fraction': [None, 0.7, 0.9]
            }
        }

    def trouver_grille(self, model):
        """envoie la bonne grille en fonction du type de modÃ¨le"""
        type_modele = type(model)
        if type_modele in self.grilles_connues:
            return self.grilles_connues[type_modele]
        else:
            raise ValueError(f"Le modÃ¨le {type_modele.__name__} n'a pas de grille de paramÃ¨tres prÃ©-dÃ©finie.")

    def trouve_params(self, model):
        """
        Trouve les meilleurs paramÃ¨tres, entraÃ®ne le modÃ¨le et le renvoie.
        """
        nom_modele = type(model).__name__
        grille = self.trouver_grille(model)

        if self.verbose > 0:
            print(f"\n[{nom_modele}] Lancement de la recherche automatique des paramÃ¨tres")
            print(f"[{nom_modele}] Grille testÃ©e : {grille}")
        
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=grille, 
            cv=self.cv, 
            scoring=self.scoring, 
            n_jobs=self.n_jobs, 
            verbose=self.verbose
        )
        
        start_time = time.time()
        grid_search.fit(self.X, self.y)
        
        if self.verbose > 0:
            print(f"\n[{nom_modele}] Recherche terminÃ©e en {time.time() - start_time:.2f} secondes !")
            print(f"[{nom_modele}] => Meilleurs paramÃ¨tres : {grid_search.best_params_}")
            print(f"[{nom_modele}] => Meilleur score ({self.scoring}) : {grid_search.best_score_ * 100:.2f}%")
        
        # Renvoie le modÃ¨le avec les meilleurs paramÃ¨tres (dÃ©jÃ  entraÃ®nÃ© par GridSearch)
        return grid_search.best_estimator_

    def trouve_params_rapide(self, model, n_iter=10):
        """
        Trouve les meilleurs paramÃ¨tres en utilisant RandomizedSearchCV.
        PlutÃ´t que de tester TOUTES les combinaisons (GridSearch), il en teste un nombre limitÃ© (n_iter).
        IdÃ©al pour gagner du temps quand la grille de paramÃ¨tres est trÃ¨s grande.
        """
        nom_modele = type(model).__name__
        grille = self.trouver_grille(model)

        if self.verbose > 0:
            print(f"\n[{nom_modele}] Lancement de la recherche RAPIDE (RandomizedSearch, {n_iter} itÃ©rations max)")
            print(f"[{nom_modele}] Grille dans laquelle piocher : {grille}")
        
        # On calcule le nombre total de combinaisons possibles
        total_combinaisons = 1
        for valeurs in grille.values():
            total_combinaisons *= len(valeurs)
            
        # Si la grille a moins de combinaisons que n_iter, on rÃ©duit n_iter pour Ã©viter une erreur
        n_iter_reel = min(n_iter, total_combinaisons)
        
        random_search = RandomizedSearchCV(
            estimator=model, 
            param_distributions=grille, 
            n_iter=n_iter_reel,
            cv=self.cv, 
            scoring=self.scoring, 
            n_jobs=self.n_jobs, 
            verbose=self.verbose,
            random_state=42 
        )
        
        start_time = time.time()
        random_search.fit(self.X, self.y)
        
        if self.verbose > 0:
            print(f"\n[{nom_modele}] Recherche rapide terminÃ©e en {time.time() - start_time:.2f} secondes !")
            print(f"[{nom_modele}] => Meilleurs paramÃ¨tres trouvÃ©s : {random_search.best_params_}")
            print(f"[{nom_modele}] => Meilleur score ({self.scoring}) : {random_search.best_score_ * 100:.2f}%")
        
        return random_search.best_estimator_



