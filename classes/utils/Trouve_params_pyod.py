import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.sos import SOS
import time

class Trouve_params_pyod:
    """
    Optimiseur d'hyperparamètres sur mesure pour l'API PyOD.
    Effectue une validation croisée (GridSearch) en évaluant soit le F1-Score, soit le ROC-AUC.
    """
    def __init__(self, X, y, cv=3, scoring='roc_auc', verbose=1):
        self.X = X
        self.y = y
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
    
        # Grilles spécifiques aux modèles PyOD
        self.grilles_connues = {
            IForest: {
                'n_estimators': [50, 100, 200],
                'max_samples': ['auto', 256]
            },
            LOF: {
                'n_neighbors': [10, 20, 30, 50],
                'metric': ['minkowski', 'euclidean']
            },
            KNN: {
                'n_neighbors': [5, 10, 20],
                'method': ['largest', 'mean', 'median']
            },
            OCSVM: {
                'nu': [0.01, 0.05, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.1]
            },
            PCA: {
                'n_components': [0.5, 0.7, 0.9] # Ratio de variance expliquée
            },
            HBOS: {
                'n_bins': [10, 20, 50],
                'tol': [0.1, 0.5]
            },
            LODA: {
                'n_bins': [10, 20, 50],
                'n_random_cuts': [50, 100, 200]
            },
            CBLOF: {
                'n_clusters': [8, 15],
                'alpha': [0.7, 0.9]
            },
            COF: {
                'n_neighbors': [10, 20]
            },
            SOS: {
                'perplexity': [4.5, 10, 50]
            },
            DeepSVDD: {
                'n_epochs': [10, 20],
                'hidden_neurons': [[64, 32], [32, 16]],
                'batch_size': [32, 64]
            },
            # ECOD et COPOD sont dits "Parameter-free" (statistiques paramétriques directes)
            COPOD: {},
            ECOD: {}
        }

    def trouver_grille(self, model):
        """Récupère la grille associée au modèle PyOD."""
        type_modele = type(model)
        if type_modele in self.grilles_connues:
            return self.grilles_connues[type_modele]
        else:
            raise ValueError(f"Le modèle {type_modele.__name__} n'a pas de grille enregistrée.")

    def trouve_params(self, model):
        """
        Effectue la recherche des meilleurs hyperparamètres (K-Fold cross-validation)
        pour un modèle PyOD donné.
        """
        st = time.time()
        grille = self.trouver_grille(model)
        type_modele = type(model)
        
        # S'il n'y a pas de paramètres à optimiser (ex: ECOD)
        if not grille:
            if self.verbose:
                print(f"[{type_modele.__name__}] Modèle sans hyperparamètre. Entraînement direct.")
            model.fit(self.X)
            return model
            
        meilleur_score = -np.inf
        meilleurs_params = None
        
        kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        parametres_a_tester = list(ParameterGrid(grille))
        
        for params in parametres_a_tester:
            scores_cv = []
            
            for train_idx, val_idx in kf.split(self.X, self.y):
                X_tr, X_val = self.X[train_idx], self.X[val_idx]
                y_tr, y_val = self.y[train_idx], self.y[val_idx]
                
                # Instance du modèle avec les paramètres actuels
                mod_instance = type_modele(**params)
                
                try:
                    mod_instance.fit(X_tr)
                    
                    if self.scoring == 'roc_auc':
                        # ROC-AUC préfère les scores continus
                        preds = mod_instance.decision_function(X_val)
                        score = roc_auc_score(y_val, preds)
                    else:
                        # F1 préfère les prédictions binaires (0/1)
                        preds = mod_instance.predict(X_val)
                        score = f1_score(y_val, preds, average='macro')
                        
                    scores_cv.append(score)
                except Exception as e:
                    # Capture les crashs liés à des paramètres incompatibles sur certains datasets
                    scores_cv.append(0)
            
            moyenne_score = np.mean(scores_cv)
            
            if moyenne_score > meilleur_score:
                meilleur_score = moyenne_score
                meilleurs_params = params
                
        if self.verbose:
            print(f"[{type_modele.__name__}] Optimisation terminée ({time.time() - st:.1f}s) | "
                  f"Meilleur {self.scoring.upper()}: {meilleur_score:.4f} | Params: {meilleurs_params}")
            
        # On recrée l'objet final avec les meilleurs attributs, et on l'entraîne sur TOUTES les données fournies
        modele_final = type_modele(**meilleurs_params)
        modele_final.fit(self.X)
        return modele_final