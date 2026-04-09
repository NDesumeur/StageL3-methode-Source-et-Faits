import numpy as np
from typing import Literal 
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from joblib import Parallel, delayed
import time

from .utils.Trouve_params import Trouve_params 

class MyVotingOutlier:
    """
    Implémentation personnalisée d'un Voting Classifier.
    Reproduit la logique de base de sklearn.ensemble.VotingClassifier en mode 'hard' et 'soft' et ajout du mode 'S&F'.


    NON implémenté :
    - get_featurs_names_out 
    - get_metadata_routing
    - set_score_request 
    - set_output (Je me concentre sur le numpy, je sais qu'il peut y avoir du pandas ou  autres types de données.)
    """
    def __init__(self, estimators: list[tuple], voting: Literal['hard', 'soft', 'S&F'] = 'hard', weights: list[float] = None, n_jobs: int = None, flatten_transform: bool = True, verbose: bool = False, sf_metric: Literal['accuracy', 'f1'] = 'accuracy'):
        """
        Paramètres :
        - estimators : Liste de tuples (nom, modele) des modèles de base.
        - voting : 'hard' (vote majoritaire) ou 'soft' (moyenne des probabilité), ou 'S&F' (Source et faits).
        - weights : Liste de poids pour pondérer les votes (optionnel).
        - n_jobs : Nombre de jobs parallèles (None = 1, -1 = tous les processeurs).
        - flatten_transform : Si True, les sorties de transform sont aplaties en une seule matrice (utile pour 'soft').
        - verbose : Affiche le temps d'entraînement si True.
        - sf_metric : 'accuracy' ou 'f1'. Utilisé par le mode S&F pour juger la fiabilité (F1 pénalise les modèles qui ratent la classe minoritaire).
        """
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.verbose = verbose
        self.sf_metric = sf_metric
        self.estimators_ = []
        self.named_estimators_ = {}
        self.le_ = LabelEncoder()
        self.classes_ = None
        self.entraine = False

    def fit(self, X, y, auto_optimize='non', sample_size_opti=1000, **fit_params):
        """
        Entraîne les modèles de base sur les données d'entraînement.
        - X : Données d'entraînement.
        - y : Étiquettes d'entraînement.
        - auto_optimize : 'rapide' pour RandomizedSearch, 'normal' pour GridSearch complet, 'non' pour ignorer.
        - sample_size_opti : Taille de l'échantillon utilisé pour l'optimisation (par défaut 1000).
        - fit_params : Dictionnaire de paramètres spécifiques à chaque modèle (optionnel).
        exemple de fit_params : {'model1': {'param1': value1}, 'model2': {'param2': value2}}
        """
        y_non_label = self.le_.fit_transform(y) # Encode les labels en entiers
        self.classes_ = self.le_.classes_ # Liste des classes originales (labels) comme dans sklearn 

        if auto_optimize in ['rapide', 'normal']:
            
            # Échantillonnage aléatoire pour la recherche de paramètres
            indices = np.random.choice(len(X), min(sample_size_opti, len(X)), replace=False)
            X_opti, y_opti = X[indices], y_non_label[indices]
            
            if self.verbose:
                print(f"[{self.__class__.__name__}] Lancement de l'auto-optimisation ({auto_optimize}) sur {len(X_opti)} données")
                
            chercheur = Trouve_params(X_opti, y_opti, cv=3, verbose=0)
            
            nouveaux_estimateurs = []
            for name, classifier in self.estimators:
                if self.verbose:
                    print(f"[{self.__class__.__name__}] Recherche {auto_optimize} pour '{name}'")
                try:
                    if auto_optimize == 'rapide':
                        modele_opti = chercheur.trouve_params_rapide(classifier, n_iter=4)
                    else:
                        modele_opti = chercheur.trouve_params(classifier)

                    # On clone le modèle optimisé pour s'assurer qu'on repart à zéro pour le vrai fit global
                    nouveaux_estimateurs.append((name, clone(modele_opti)))
                    if self.verbose:
                        print(f"[{self.__class__.__name__}] -> Nouveaux paramètres: {modele_opti.get_params()}")
                except Exception as e:
                    if self.verbose:
                        print(f"[{self.__class__.__name__}] -> Ignoré (pas de grille connue pour {type(classifier).__name__}).")
                    nouveaux_estimateurs.append((name, classifier))
            
            self.estimators = nouveaux_estimateurs


        if self.estimators is None or len(self.estimators) == 0:
            raise ValueError("Aucun estimateur n'a été fourni.")

        if self.verbose:
            print(f"[{self.__class__.__name__}] Démarrage de l'entraînement des modèles (n_jobs={self.n_jobs}).")

        taches_paralleles = []
        for name, clf in self.estimators:
            #  On clone le modèle ('clone') pour garder self.estimators complétement vierge.
            modele_copie = clone(clf)
            
            #  On récupère les paramètres spécifiques de ce modèle (s'il y en a)
            params_specifiques = {}
            if fit_params and name in fit_params:
                params_specifiques = fit_params[name]
                
            #  On demande à Joblib ("delayed") de préparer l'appel à la fonction .fit()
            tache = delayed(modele_copie.fit)(X, y_non_label, **params_specifiques)
            taches_paralleles.append(tache)

        #  Lancement de l'exécution simultanée des entraînements
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(taches_paralleles)
        # On remplit le dictionnaire des estimateurs nommés pour un accès plus facile
        for i in range(len(self.estimators)):
            name = self.estimators[i][0]
            self.named_estimators_[name] = self.estimators_[i]
            
        self.entraine = True
            
        if self.voting == 'S&F':
            if self.verbose:
                print(f"[{self.__class__.__name__}] Calcul des poids Source & Faits sur le set d'entraînement")
            self._fit_SF(X)
            
        return self

    def get_params(self, deep=True):
        """
        Renvoie les paramètres du classifieur.
        on utilise nom__param pour différencier les paramètres du classifier et ceux des modèles de base.
        """
        params = {
            'estimators': self.estimators,
            'voting': self.voting,
            'weights': self.weights,
            'n_jobs': self.n_jobs,
            'flatten_transform': self.flatten_transform,
            'verbose': self.verbose,
            'sf_metric': getattr(self, 'sf_metric', 'accuracy')
        }
        
        if deep:
            # Récupérer récursivement les paramètres de chaque modèle de base
            for name, estimator in self.estimators:
                if hasattr(estimator, 'get_params'):
                    for key, value in estimator.get_params(deep=True).items():
                        params[f'{name}__{key}'] = value
        
        return params

    def set_params(self, **params):
        """
        Définit les paramètres de cet estimateur ou de ses sous-modèles. Donc permet de modifier les paramètres de MyVotingClassfier directement.
        ex: set_params(weights=[0.5, 0.5], svm__C=1.0) pour définir les poids du vote et le paramètre C du modèle SVM.
        """
        if not params:
            return self
            
        # Séparer les paramètres directs (ex: 'voting') et ceux des sous-modèles (ex: 'svm__C')
        for key, value in params.items():
            if '__' in key:
                # Paramètre destiné à un sous-modèle (ex: svm__kernel)
                estimator_name, estimator_param = key.split('__', 1)
                for name, classifier in self.estimators:
                    if name == estimator_name:
                        # Si le modèle a lui-même un set_params, on l'utilise
                        if hasattr(classifier, 'set_params'):
                            classifier.set_params(**{estimator_param: value})
                        else:
                            setattr(classifier, estimator_param, value)
            else:
                # Paramètre direct pour MyVotingClassifier (ex: voting, weights)
                setattr(self, key, value)
                
        return self
    def _predict_encoded_safe(self, estimator, X):
        predictions = estimator.predict(X)
        try:
            return self.le_.transform(predictions)
        except Exception:
            return predictions

    def _predict_proba_safe(self, estimator, X):
        if hasattr(estimator, 'predict_proba'):
            return estimator.predict_proba(X)
        elif hasattr(estimator, 'decision_function'):
            import numpy as np
            scores = estimator.decision_function(X)
            scores = np.clip(scores, -100, 100) # Evite overflow exp()
            prob_1 = 1 / (1 + np.exp(-scores))
            prob_0 = 1 - prob_1
            
            out = np.zeros((len(scores), 2))
            if self.classes_ is not None and len(self.classes_) == 2 and -1 in self.classes_ and 1 in self.classes_:
                idx_1 = self.le_.transform([1])[0]
                idx_m1 = self.le_.transform([-1])[0]
                out[:, idx_1] = prob_1
                out[:, idx_m1] = prob_0
            else:
                out[:, 0] = prob_0
                out[:, 1] = prob_1
            return out
        else:
            raise AttributeError(f"{type(estimator).__name__} ne supporte ni predict_proba ni decision_function.")

    def transform(self, X):
        """
        Transforme les données d'entrée X en utilisant les modèles de base.
        - X : Données d'entrée. 
        Retourne : Tableau des prédictions transformées.
        """
        if not self.entraine:
            return "Les modèles de base doivent être entraînés avant de faire des transformations. Veuillez appeler la méthode fit() d'abord."
        
        if self.voting == 'soft':
            # Demande des probabilités aux modèles en simultané (Multithreading)
            probas_list = Parallel(n_jobs=self.n_jobs)(
                delayed(self._predict_proba_safe)(estimator, X) for estimator in self.estimators_
            )
            if not self.flatten_transform:
                return np.array(probas_list).transpose((1, 0, 2))
            else:
                return np.hstack(probas_list)
                
        # hard et S&F
        if self.voting in ['hard', 'S&F']:
            # Demande des prédictions de classes en simultané
            predictions_list = Parallel(n_jobs=self.n_jobs)(
                delayed(self._predict_encoded_safe)(estimator, X) for estimator in self.estimators_
            )
            predictions = np.array(predictions_list)
            # Transpose pour avoir le format attendue (n_samples, n_estimators)
            return predictions.T # .T vient de np.array et permet de faire la transposition de la matrice (inverser les lignes et les colonnes)


    
    def fit_transform(self, X, y, **fit_params):
        """
        Entraîne les modèles de base et transforme les données d'entrée X.
        - X : Données d'entraînement.
        - y : Étiquettes d'entraînement.
        - fit_params : Dictionnaire de paramètres spécifiques à chaque modèle (optionnel).
        Retourne : Tableau des prédictions transformées.
        """
        self.fit(X, y, **fit_params)
        return self.transform(X)
    
    def predict(self, X):
        """
        Prédit les classes pour les données d'entrée X.
        - X : Données de test.
        Retourne : Tableau des classes prédites.
        """
        if self.entraine != True:
            return "Les modèles de base doivent être entraînés avant de faire des prédictions. Veuillez appeler la méthode fit() d'abord."
            
        # appel pour tous les modeles
        predictions_matrice_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_encoded_safe)(estimator, X) for estimator in self.estimators_
        )

        if self.voting == 'S&F':
            # Vote pondéré utilisant les sf_weights_ appris lors du fit()
            predictions_matrice = predictions_matrice_list
            n_estimators = len(self.estimators_)
            n_samples = len(X)
            
            votes_finaux = [0 for _ in range(n_samples)]
            for i in range(n_samples):
                score_classes = {}
                for m in range(n_estimators):
                    vote = predictions_matrice[m][i]
                    poids = self.sf_weights_[m] # On utilise le poids S&F mémorisé
                    
                    if vote in score_classes:
                        score_classes[vote] += poids
                    else:
                        score_classes[vote] = poids
                
                # Trouver la classe gagnante pour la donnée i
                meilleur_score = -1
                classe_gagnante = -1
                for classe, score in score_classes.items():
                    if score > meilleur_score:
                        meilleur_score = score
                        classe_gagnante = classe
                votes_finaux[i] = classe_gagnante
                
            return self.le_.inverse_transform(votes_finaux)

        if self.voting == 'soft':
            # Vote basé sur la moyenne des probabilités prédites par les modèles de base
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
            return self.le_.inverse_transform(maj_vote)

        elif self.voting == 'hard':
            # Vote majoritaire simple (sans pondération)
            predictions = np.array(predictions_matrice_list)
            maj_vote = np.zeros(predictions.shape[1], dtype=int)
            
            # Compter les votes pour chaque image
            for i in range(predictions.shape[1]): # Pour chaque image
                colonne_votes = predictions[:, i]  #  prend toutes les lignes mais laisse seulement la colonne i (les votes pour l'image i)
                # Si des poids sont fournis, on les utilise pour pondérer les votes
                # bincount compte combien il y a de 0, de 1, etc.
                # argmax prend l'index du plus grand compte (le vainqueur)
                gagnant = np.bincount(colonne_votes, weights=self.weights).argmax()
                maj_vote[i] = gagnant
                
            # On retransforme les entiers en vrais labels
            return self.le_.inverse_transform(maj_vote)

    def predict_proba(self, X):
        """
        Prédit les probabilités pour les classes pour les données d'entrée X.
        - X : Données de test.
        Retourne : Tableau des probabilités prédites pour chaque classe.
        """
        if self.voting != 'soft':
            return "La méthode predict_proba n'est pas importante pour le mode 'hard'."
            
        probas_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_proba_safe)(estimator, X) for estimator in self.estimators_
        )
        probas = np.array(probas_list)
        
        # Une fois qu'on a toutes les réponses, on en fait la moyenne pondérée ou simple
        if self.weights is not None:
            weights = np.asarray(self.weights)
            # On s'assure que la somme des poids vaut 1
            weights = weights / weights.sum()
            avg_probas = np.average(probas, axis=0, weights=weights) # On utilise les poids fournis pour faire la moyenne pondérée des probabilités
        else:
            avg_probas = np.mean(probas, axis=0)
            
        return avg_probas
    
    def score_confiance(self, X):
        """
        Analyse spécifique pour comprendre la "solidité" du vote final (utile pour la décision).
        Indique pour chaque ligne de donnée si le vote a été unanime, majoritaire fort, ou très disputé.
        
        Retourne une liste de dictionnaires contenant pour chaque donnée:
        - 'prediction_finale': La classe choisie
        - 'taux_confiance' : Pourcentage de voix remportées par la classe gagnante,
        - 'details_votes': Le détail exact de ce qu'ont voté les différents modèles.
        """
        if not self.entraine:
            raise ValueError("Erreur : Entraînez d'abord le modèle avec fit()")

        predictions_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_encoded_safe)(estimator, X) for estimator in self.estimators_
        )
        predictions_brutes = np.array(predictions_list)
        
        resultats_confiance = []
        n_estimators = len(self.estimators)
        
        for i in range(predictions_brutes.shape[1]): # Pour chaque donnée de X (ex: chaque image)
            # Les votes pour cette donnée précise
            votes_i = predictions_brutes[:, i]
            
            # Application des poids si disponibles et selon le mode de vote
            if self.voting == 'S&F':
                poids = self.sf_weights_
            else:
                poids = self.weights if self.weights is not None else np.ones(n_estimators)

            # Analyse des votes
            comptage_voix = {}
            details_modeles = {}
            total_voix_exprimees = 0
            # On parcourt les votes de chaque modèle pour cette donnée i et on compte les voix pour chaque classe, en tenant compte des poids
            for j in range(len(self.estimators)):
                nom_modele = self.estimators[j][0]
                vote_entier = votes_i[j]
                # On inverse la transformation pour retrouver le label original de la classe votée
                label_original = self.le_.inverse_transform([vote_entier])[0]
                poids_du_vote = poids[j]
                # On stocke le détail du vote de ce modèle pour cette donnée
                details_modeles[nom_modele] = label_original
                comptage_voix[label_original] = comptage_voix.get(label_original, 0) + poids_du_vote
                total_voix_exprimees += poids_du_vote
                
            # Identifier la classe gagnante
            # comptage_voix : {'classe1': poids_total_voix_classe1, 'classe2': poids_total_voix_classe2, ...}
            # comptage_voix.get('classe1', 0) = nombre de voix (pondérées) pour classe1
            gagnant = max(comptage_voix, key=comptage_voix.get)
            voix_gagnant = comptage_voix[gagnant]
            taux_confiance = (voix_gagnant / total_voix_exprimees) * 100
            
            resultats_confiance.append({
                'prediction_finale': gagnant,
                'taux_confiance': round(taux_confiance, 2),
                'details_votes': details_modeles,
                'score_voix': f"{voix_gagnant}/{total_voix_exprimees}"
            })
            
        return resultats_confiance

    def score(self, X, y):
        """
        Calcule la précision du classifieur sur les données de test.
        - X : Données de test.
        - y : Étiquettes de test.
        Retourne : Précision du classifieur.
        """
        predictions = self.predict(X) 
        return np.mean(predictions == y)

    def _fit_SF(self, X_train, epsilon_arret=0.001, max_iter=100):
        """
        Méthode 'Source et Faits' optimisée sur les boucles, même fonctionnement que la version SF1 laissée dans le code.
        """
        # Obtenir toutes les prédictions
        # REMPLACE : La liste de listes par une matrice 2D NumPy (Lignes=Images, Colonnes=Estimateurs)
        # Le .T (transpose) permet d'avoir (n_samples, n_estimators)
        predictions_matrice = np.array([self._predict_encoded_safe(estimator, X_train) for estimator in self.estimators_]).T
        
        n_samples = predictions_matrice.shape[0]
        n_estimators = predictions_matrice.shape[1]
        
        # 1. Poids initiaux
        # REMPLACE : "poids_sources = [1.0 for _ in range(n_estimators)]" 
        poids_sources = np.ones(n_estimators)

        for iteration in range(max_iter):
            # REMPLACE : La petite boucle "for m in range(n_estimators): anciens_poids[m] = poids_sources[m]" 
            # par le clonage de Numpy
            anciens_poids = poids_sources.copy()
            
            # ÉTAPE 1 : Trouver la classe gagnante
            # REMPLACE : La double-boucle + le dictionnaire "score_classes" manuel
            
            # 1. Création d'une grille NumPy rapide (Images x Modèles x Classes) remplie de 0 et de 1
            n_classes = len(self.classes_)
            grille_votes = np.eye(n_classes)[predictions_matrice] # eye = matrice  identité, ici prediction_matrice contient des entiers (0,1,etc correspondant aux classes), on utilise ces entiers pour créer une matrice où la position de la classe votée est à 1 et les autres à 0. La forme de grille_votes sera (n_samples, n_estimators, n_classes).
            
            # 2. On multiplie ces "1" par le poids actuel de chaque modèle 
            votes_ponderes_par_modele = grille_votes * poids_sources.reshape(1, n_estimators, 1)
            
            # 3. On additionne les voix pour chaque classe, puis on prend l'indice (argmax) de la classe gagnante
            # votes_ponderes_par_modele est de forme (n_samples, n_estimators, n_classes), on somme sur l'axe 1 (les votes des modèles) pour obtenir une matrice (n_samples, n_classes) qui contient le score total de chaque classe pour chaque échantillon. Ensuite, np.argmax(..., axis=1) nous donne l'indice de la classe gagnante pour chaque échantillon.      
            votes_finaux = votes_ponderes_par_modele.sum(axis=1).argmax(axis=1)

            # ÉTAPE 2 : Recalculer la fiabilité des modèles
            # REMPLACE : La double boucle if predictions_matrice[m][i] == votes_finaux[i]: bonnes_reponses += 1 ...
            # predictions_matrice == votes_finaux[:, None] compare TOUTE la grille aux résultats d'un coup 
            # np.mean(..., axis=0) fait la moyenne des True par colonne (par modèle) et donne directement le % de réussite
            if getattr(self, 'sf_metric', 'accuracy') == 'f1':
                from sklearn.metrics import f1_score
                new_poids = np.zeros(n_estimators)
                for m in range(n_estimators):
                    # le score f1 macro penalise fortement un modèle qui ignore la classe minoritaire (ex: les anomalies)
                    new_poids[m] = f1_score(votes_finaux, predictions_matrice[:, m], average='macro', zero_division=0)
                poids_sources = new_poids
            else:
                poids_sources = np.mean(predictions_matrice == votes_finaux[:, None], axis=0)

            # donc par exemple si le modèle 0 a 80% de bonnes réponses, poids_sources[0] vaudra 0.8, etc.
            # ex: predictions_matrice = [[0,1,0], [1,1,0], [0,0,1]] (3 échantillons, 3 modèles)
            #     votes_finaux = [0, 1, 0] (les classes gagnantes pour les 3 échantillons)
            #     predictions_matrice == votes_finaux[:, None] produira une matrice de booléens indiquant où les prédictions de chaque modèle correspondent aux votes finaux, et np.mean(..., axis=0) donnera le pourcentage de correspondance pour chaque modèle.
            # ainsi poid_sources contiendra directement les scores de fiabilité de chaque modèle par rapport au vote majoritaire.


            # ÉTAPE 3 : Vérifier la convergence
            # REMPLACE : La boucle diff = abs(poids_sources[m] - anciens_poids[m])
            # np.max(np.abs) trouve la modification la plus grande de poids lors de ce tour
            # np.abs renvoie la valeur absolue et ici on regarde si les poids ont changé de manière significative par rapport à l'itération précédente. Si le changement maximum est inférieur au seuil epsilon_arret, on considère que les poids ont convergé et on peut arrêter les itérations.
            max_diff = np.max(np.abs(poids_sources - anciens_poids))
            
            if max_diff < epsilon_arret:
                if self.verbose:
                    print(f"[{self.__class__.__name__}] Convergence S&F atteinte après {iteration} itérations.")
                break
                
        self.sf_weights_ = poids_sources

    def _fit_SF1(self, X_train, epsilon_arret=0.001, max_iter=100):
        """
        Méthode "Source et Faits" (S&F) itérative calculée lors de l'entraînement.
        permet de calculer la fiabilité de chaque modèle par rapport au vote majoritaire.
        - X_train : Données d'entraînement utilisées pour calculer les poids S&F
        - epsilon_arret : Seuil de convergence pour les poids (si les poids ne changent pas significativement, on s'arrête)
        - max_iter : Nombre maximum d'itérations pour éviter les boucles infinies en cas de non-convergence
        """
        # Obtenir toutes les prédictions
        predictions_matrice = [self._predict_encoded_safe(estimator, X_train) for estimator in self.estimators_]
        n_estimators = len(self.estimators_)
        n_samples = len(X_train)
        
        # 1. Poids initiaux (tous les modèles valent 1 au début)
        poids_sources = [1.0 for _ in range(n_estimators)]
        anciens_poids = [0.0 for _ in range(n_estimators)] # Pour vérifier la convergence à la fin de chaque tour
        votes_finaux = [0 for _ in range(n_samples)] # Initialisation des votes finaux (classe gagnante pour chaque donnée)
        
        for iteration in range(max_iter):
            # Copie de sécurité pour vérifier la convergence à la fin du tour
            for m in range(n_estimators):
                anciens_poids[m] = poids_sources[m]
                
            for i in range(n_samples):
                # On compte les votes selon le poids de chaque modèle
                score_classes = {}
                for m in range(n_estimators):
                    vote_du_modele_m = predictions_matrice[m][i]
                    poids_du_modele_m = poids_sources[m]
                    
                    if vote_du_modele_m in score_classes:
                        score_classes[vote_du_modele_m] += poids_du_modele_m
                    else:
                        score_classes[vote_du_modele_m] = poids_du_modele_m
                
                # Trouver la classe gagnante pour la donnée i
                meilleur_score = -1
                classe_gagnante = -1
                for classe, score in score_classes.items():
                    if score > meilleur_score:
                        meilleur_score = score
                        classe_gagnante = classe
                        
                votes_finaux[i] = classe_gagnante
                
            # ÉTAPE 2 : Recalculer la fiabilité des modèles
            for m in range(n_estimators):
                bonnes_reponses = 0
                for i in range(n_samples):
                    if predictions_matrice[m][i] == votes_finaux[i]:
                        bonnes_reponses += 1
                
                # Le nouveau poids est le pourcentage de bonnes réponses face au grand groupe
                nouveau_poids = bonnes_reponses / n_samples
                poids_sources[m] = nouveau_poids
                
            # ÉTAPE 3 : Vérifier si on s'arrête 
            convergence = True
            for m in range(n_estimators):
                diff = abs(poids_sources[m] - anciens_poids[m])
                if diff >= epsilon_arret:
                    convergence = False
                    break # Pas besoin de tout vérifier, un seul empêche de s'arrêter
                    
            if convergence and iteration > 0:
                if self.verbose:
                    print(f"[{self.__class__.__name__}] Convergence S&F atteinte après {iteration} itérations pendant l'entraînement.")
                break
                
        self.sf_weights_ = poids_sources
        # La fonction ne retourne rien, elle mémorise juste les poids dans l'objet
    