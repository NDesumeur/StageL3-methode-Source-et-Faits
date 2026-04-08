import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.MyVotingClassifier import MyVotingClassifier
from classes.utils.Trouve_params import Trouve_params
from classes.utils.Normaliseur import Normaliseur
from classes.utils.Evaluateur import Evaluateur
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

import numpy as np

mnist = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X, y = mnist

norm = Normaliseur(methode='minmax')
X = norm.fit_transform(X)

def main():
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    print("\n" + "="*80)
    print(" TEST UTILITAIRE : Trouve_params (sur un sous-échantillon)")
    print("="*80)

    dt_base = DecisionTreeClassifier(random_state=42)
    chercheur = Trouve_params(X_train[:1000], y_train[:1000])
    dt_optimise = chercheur.trouve_params_rapide(dt_base, n_iter=5)
    print(f"Meilleurs hyperparamètres trouvés : {dt_optimise.get_params()}")

    print("\n" + "="*80)
    print(" COMPARAISON : Modèles Individuels vs MyVotingClassifier (Soft, Hard, S&F)")
    print("="*80)

    clf_bases = [
        ('et', ExtraTreesClassifier(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('nb_g', GaussianNB()),
        ('knn_3', KNeighborsClassifier()),
        ('knn_7', KNeighborsClassifier()),
        ('logreg', LogisticRegression(max_iter=200, random_state=42)),
        ('sgd', SGDClassifier(random_state=42)),
        ('nb_m', MultinomialNB()),
        ('rf_faible', RandomForestClassifier(random_state=42)),
        ('et_faible', ExtraTreesClassifier(random_state=42)),
        ('bagging', BaggingClassifier(random_state=42)),
        ('ada', AdaBoostClassifier(random_state=42)),

        ('bn', BernoulliNB(alpha=1.0, binarize=0.5)),
        ('dt_depth10', DecisionTreeClassifier(max_depth=10, random_state=42)),
        ('etc_faible', ExtraTreeClassifier(max_depth=5, random_state=42))
    ]

    voting_custom = MyVotingClassifier(estimators=clf_bases, voting='S&F', verbose=True)

    print("\n--- Entraînement de la flotte (12 modèles) avec auto-optimisation et évaluation S&F ---")
    start = time.time()

    voting_custom.fit(X_train, y_train, auto_optimize='normal')
    print(f"-> Entraînement terminé en {time.time() - start:.2f} secondes")

    print("\n--- SCORES INDIVIDUELS DES 12 MODELES SUR LE TEST SET ---")
    for i in range(len(clf_bases)):
        name = clf_bases[i][0]
        estimator = voting_custom.estimators_[i]

        y_pred_ind_entiers = estimator.predict(X_test)
        y_pred_ind = voting_custom.le_.inverse_transform(y_pred_ind_entiers)

        acc = accuracy_score(y_test, y_pred_ind)
        print(f"  - {name:>10} : {acc*100:.2f}% de precision")

    print(f"\n   Poids 'Source & Faits' trouves par l'algorithme :")
    for i in range(len(clf_bases)):
        name = clf_bases[i][0]
        poids = voting_custom.sf_weights_[i]
        print(f"    - {name:>10} : {poids:.4f}")

    print("\n" + "="*80)
    print(" RESULTATS GLOBAUX DE L'ENSEMBLE (SOFT, HARD, S&F) ")
    print("="*80)

    voting_custom.voting = 'soft'
    eval_my_soft = Evaluateur(voting_custom, X_test, y_test)
    metrics_my_soft = eval_my_soft.evaluate()
    eval_my_soft.print_metrics(metrics_my_soft, nom_modele="MyVoting (SOFT)")

    voting_custom.voting = 'hard'
    eval_my_hard = Evaluateur(voting_custom, X_test, y_test)
    metrics_my_hard = eval_my_hard.evaluate()
    eval_my_hard.print_metrics(metrics_my_hard, nom_modele="MyVoting (HARD)")

    voting_custom.voting = 'S&F'
    eval_my_sf = Evaluateur(voting_custom, X_test, y_test)
    metrics_my_sf = eval_my_sf.evaluate()
    eval_my_sf.print_metrics(metrics_my_sf, nom_modele="MyVoting (Source & Faits)")

    print("\nAffichage de la matrice de confusion pour MyVoting (S&F)")
    eval_my_sf.plot_confusion_matrix(save_path="matrice_confusion_myvoting_sf.png")

    print("\n\n" + "="*80)
    print(" PREUVE DE ROBUSTESSE : Validation Croisée (3-Folds) sur 15 000 images")
    print("="*80)
    from sklearn.model_selection import StratifiedKFold

    np.random.seed(42)
    indices_cv = np.random.choice(len(X), 15000, replace=False)
    X_cv, y_cv = X[indices_cv], y[indices_cv]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores_cv = {'soft': [], 'hard': [], 'S&F': []}

    from sklearn.base import clone
    clf_bases_cv = [(nom, clone(mod)) for nom, mod in clf_bases]

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_cv, y_cv)):
        print(f"\n--- Itération {fold+1}/3 de la Validation Croisée ---")
        X_train_cv, y_train_cv = X_cv[train_idx], y_cv[train_idx]
        X_test_cv, y_test_cv = X_cv[test_idx], y_cv[test_idx]

        voting_cv = MyVotingClassifier(estimators=clf_bases_cv, voting='S&F', verbose=False)
        voting_cv.fit(X_train_cv, y_train_cv, auto_optimize='non')

        voting_cv.voting = 'soft'
        acc_soft = accuracy_score(y_test_cv, voting_cv.predict(X_test_cv))
        scores_cv['soft'].append(acc_soft)

        voting_cv.voting = 'hard'
        acc_hard = accuracy_score(y_test_cv, voting_cv.predict(X_test_cv))
        scores_cv['hard'].append(acc_hard)

        voting_cv.voting = 'S&F'
        acc_sf = accuracy_score(y_test_cv, voting_cv.predict(X_test_cv))
        scores_cv['S&F'].append(acc_sf)

        print(f"Scores du Split {fold+1} -> Soft: {acc_soft*100:.2f}% | Hard: {acc_hard*100:.2f}% | S&F: {acc_sf*100:.2f}%")

    print("\n--- MOYENNE GLOBALE (La statistique scientifique réelle) ---")
    moy_soft = np.mean(scores_cv['soft']) * 100
    moy_hard = np.mean(scores_cv['hard']) * 100
    moy_sf = np.mean(scores_cv['S&F']) * 100
    print(f"Soft Voting    : {moy_soft:.2f}%")
    print(f"Hard Voting    : {moy_hard:.2f}%")
    print(f"Source & Faits : {moy_sf:.2f}% (Victoire : {'Oui' if moy_sf > moy_soft and moy_sf > moy_hard else 'Non'})")

    print("\n\n--- ANALYSE DETAILLEE : SCORE DE CONFIANCE DU JURY ---")

    np.random.seed(42)
    indices = np.random.choice(len(X_test), 5, replace=False)
    X_echantillon = X_test[indices]
    y_echantillon_reel = y_test[indices]

    voting_custom.voting = 'S&F'
    voting_custom.set_params(weights=voting_custom.sf_weights_)

    bulletins_de_vote = voting_custom.score_confiance(X_echantillon)

    for idx in range(len(bulletins_de_vote)):
        resultat = bulletins_de_vote[idx]
        vraie_valeur = y_echantillon_reel[idx]

        print(f"\nImage Test #{idx+1} (Le vrai chiffre est : {vraie_valeur})")
        print(f" -> Prédiction retenue : {resultat['prediction_finale']}")

        confiance = resultat['taux_confiance']
        if confiance == 100.0:
            print(" -> État du jury :  UNANIME (100%)")
        elif confiance > 50.0:
            print(f" -> État du jury :  MAJORITAIRE ({confiance}%)")
        else:
            print(f" -> État du jury :  TRÈS DISPUTÉ ({confiance}%) - Mérite une vérification par un humain !")

        print(f" -> Détail complet des votes : {resultat['details_votes']}")
        print(f" -> Score comptable en voix : {resultat['score_voix']}")

if __name__ == "__main__":
    main()
