from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import time

class Trouve_params:
    """
    Classe permettant de trouver automatiquement les meilleurs hyperparamètres (GridSearch)
    selon le type de modèle qui lui est fourni.
    Elle contient en interne les grilles de recherches standard pour les modèles communs.
    """
    def __init__(self, X, y, cv=3, scoring='f1_macro', n_jobs=-1, verbose=2):
        self.X = X
        self.y = y
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
    
        self.grilles_connues = {
            # paramètres trouvés sur internet, il peut y avoir mieux que ça. 
            SVC: {
                'C': [0.1, 1, 10], 
                'gamma': ['scale', 'auto', 0.01], 
                'kernel': ['rbf', 'linear']
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
            }
        }

    def trouver_grille(self, model):
        """envoie la bonne grille en fonction du type de modèle"""
        type_modele = type(model)
        if type_modele in self.grilles_connues:
            return self.grilles_connues[type_modele]
        else:
            raise ValueError(f"Le modèle {type_modele.__name__} n'a pas de grille de paramètres pré-définie.")

    def trouve_params(self, model):
        """
        Trouve les meilleurs paramètres, entraîne le modèle et le renvoie.
        """
        nom_modele = type(model).__name__
        grille = self.trouver_grille(model)

        if self.verbose > 0:
            print(f"\n[{nom_modele}] Lancement de la recherche automatique des paramètres")
            print(f"[{nom_modele}] Grille testée : {grille}")
        
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
            print(f"\n[{nom_modele}] Recherche terminée en {time.time() - start_time:.2f} secondes !")
            print(f"[{nom_modele}] => Meilleurs paramètres : {grid_search.best_params_}")
            print(f"[{nom_modele}] => Meilleur score ({self.scoring}) : {grid_search.best_score_ * 100:.2f}%")
        
        # Renvoie le modèle avec les meilleurs paramètres (déjà entraîné par GridSearch)
        return grid_search.best_estimator_

    def trouve_params_rapide(self, model, n_iter=10):
        """
        Trouve les meilleurs paramètres en utilisant RandomizedSearchCV.
        Plutôt que de tester TOUTES les combinaisons (GridSearch), il en teste un nombre limité (n_iter).
        Idéal pour gagner du temps quand la grille de paramètres est très grande.
        """
        nom_modele = type(model).__name__
        grille = self.trouver_grille(model)

        if self.verbose > 0:
            print(f"\n[{nom_modele}] Lancement de la recherche RAPIDE (RandomizedSearch, {n_iter} itérations max)")
            print(f"[{nom_modele}] Grille dans laquelle piocher : {grille}")
        
        # On calcule le nombre total de combinaisons possibles
        total_combinaisons = 1
        for valeurs in grille.values():
            total_combinaisons *= len(valeurs)
            
        # Si la grille a moins de combinaisons que n_iter, on réduit n_iter pour éviter une erreur
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
            print(f"\n[{nom_modele}] Recherche rapide terminée en {time.time() - start_time:.2f} secondes !")
            print(f"[{nom_modele}] => Meilleurs paramètres trouvés : {random_search.best_params_}")
            print(f"[{nom_modele}] => Meilleur score ({self.scoring}) : {random_search.best_score_ * 100:.2f}%")
        
        return random_search.best_estimator_
