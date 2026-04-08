import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.MyVotingClassifier import MyVotingClassifier
from classes.utils.Trouve_params import Trouve_params
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB

import numpy as np

mnist = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X, y = mnist

X = (X / 255.0).astype(np.float32)

def test_mnist_data(X, y, num_images=1):

    print("Type des donnees (X):", type(X))
    print("Dimensions de X:", X.shape)
    print("Type des etiquettes (y):", type(y))
    print("Dimensions de y:", y.shape)

    print("\n ===  Premier exemple === ")
    print("Etiquette (chiffre attendu) :", y[num_images-1])
    print("\nValeurs des pixels (pixels de la premiere image) :")
    print(X[num_images-1][:784])

    print("\nAffichage de l'image")
    image = X[num_images-1].reshape(28, 28)
    plt.imshow(image, cmap="gray")
    plt.title(f"Chiffre : {y[num_images-1]}")
    plt.axis("off")
    plt.show()

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_modele(X_train, y_train, modele):
    modele.fit(X_train, y_train)

def evaluate_modele(X_test, y_test, modele, nom_modele):
    print(f"\nEvaluation du {nom_modele}")
    y_pred = modele.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"Accuracy sur l'ensemble de test  : {acc * 100:.2f}%")
    print(f"Precision sur l'ensemble de test : {precision * 100:.2f}%")
    print(f"Recall sur l'ensemble de test    : {recall * 100:.2f}%")
    print(f"F1-Score sur l'ensemble de test  : {f1 * 100:.2f}%")

def run_cross_validation(X, y, modele, nom_modele, cv=3):
    
    print(f"\n === Cross-Validation ({cv} folds) pour {nom_modele} === ")
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    scores = cross_validate(modele, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    print(f" Moyenne Accuracy  : {scores['test_accuracy'].mean() * 100:.2f}%")
    print(f" Moyenne Precision : {scores['test_precision_macro'].mean() * 100:.2f}%")
    print(f" Moyenne Recall    : {scores['test_recall_macro'].mean() * 100:.2f}%")
    print(f" Moyenne F1-Score  : {scores['test_f1_macro'].mean() * 100:.2f}%")

def optimize_svm_with_gridsearch(X_train, y_train):
    print("\n === Optimisation des hyperparametres du SVM (GridSearch CV) === ")
    print(" Recherche des meilleurs parametres C et gamma moyennant une Cross-Validation interne")

    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01],
        'kernel': ['rbf']
    }

    base_svm = SVC(max_iter=1000, random_state=42)

    grid_search = GridSearchCV(base_svm, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)

    print(f"\n Meilleurs parametres trouves : {grid_search.best_params_}")
    print(f" Meilleur score (F1-Macro) trouve en validation interne : {grid_search.best_score_ * 100:.2f}%")

    return grid_search.best_estimator_

def optimize_knn_with_gridsearch(X_train, y_train):
    print("\n === Optimisation des hyperparametres du KNN (GridSearch CV) === ")
    print(" Recherche du meilleur nombre de voisins 'n_neighbors'")

    param_grid = {
        'n_neighbors': [3, 5, 7]
    }

    base_knn = KNeighborsClassifier()
    grid_search = GridSearchCV(base_knn, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"\n Meilleurs parametres trouves pour KNN : {grid_search.best_params_}")
    print(f" Meilleur score (F1-Macro) trouve en validation interne : {grid_search.best_score_ * 100:.2f}%")

    return grid_search.best_estimator_

def optimize_ridge_with_gridsearch(X_train, y_train):
    print("\n === Optimisation des hyperparametres du Ridge Classifier (GridSearch CV) === ")
    print(" Recherche de la meilleure force de regularisation 'alpha'...")

    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    }

    base_ridge = RidgeClassifier(random_state=42)
    grid_search = GridSearchCV(base_ridge, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"\n Meilleurs parametres trouves pour Ridge : {grid_search.best_params_}")
    print(f" Meilleur score (F1-Macro) trouve en validation interne : {grid_search.best_score_ * 100:.2f}%")

    return grid_search.best_estimator_

def main():

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    modele_rf = RandomForestClassifier(n_estimators=50)
    modele_knn = KNeighborsClassifier(n_neighbors=3)
    modele_decision_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    modele_log_reg = LogisticRegression(max_iter=1000, random_state=42)
    modele_nbayes = GaussianNB()
    modele_svm = SVC(kernel='linear', max_iter=1000, random_state=42)
    modele_ridge = RidgeClassifier(random_state=42)

    print("\nLancement de la Cross-Validation pour tous les modeles (sur les donnees d'entrainement)")
    run_cross_validation(X_train, y_train, modele_rf, "Random Forest", cv=3)
    run_cross_validation(X_train, y_train, modele_knn, "KNN", cv=3)
    run_cross_validation(X_train, y_train, modele_decision_tree, "Decision Tree", cv=3)
    run_cross_validation(X_train, y_train, modele_log_reg, "Régression Logistique", cv=3)
    run_cross_validation(X_train, y_train, modele_nbayes, "Naive Bayes", cv=3)
    run_cross_validation(X_train, y_train, modele_svm, "SVM", cv=3)
    run_cross_validation(X_train, y_train, modele_ridge, "Ridge Classifier", cv=3)

    chercheur = Trouve_params(X_train, y_train)

    meilleur_modele_svm = chercheur.trouve_params(SVC(max_iter=1000, random_state=42))
    meilleur_modele_knn = chercheur.trouve_params(KNeighborsClassifier())
    meilleur_modele_ridge = chercheur.trouve_params(RidgeClassifier(random_state=42))

    print("\n === EVALUATION FINALE CIBLEE SUR LE TEST SET (20% CACHES) ===")
    evaluate_modele(X_test, y_test, meilleur_modele_svm, "SVM (Optimisé)")
    evaluate_modele(X_test, y_test, meilleur_modele_knn, "KNN (Optimisé)")
    evaluate_modele(X_test, y_test, meilleur_modele_ridge, "Ridge (Optimisé)")
    evaluate_modele(X_test, y_test, modele_rf, "Random Forest")
    evaluate_modele(X_test, y_test, modele_decision_tree, "Decision Tree ")
    evaluate_modele(X_test, y_test, modele_log_reg, "Régression Logistique ")
    evaluate_modele(X_test, y_test, modele_nbayes, "Naive Bayes ")

    print("\n" + "="*80)
    print(" COMPARAISON : MyVotingClassifier vs sklearn.ensemble.VotingClassifier")
    print("="*80)

    clf1 = DecisionTreeClassifier(max_depth=15, random_state=42)
    clf2 = RandomForestClassifier(n_estimators=30, random_state=42)
    clf3 = GaussianNB()

    estimators_sklearn = [('dt', clf1), ('rf', clf2), ('nb', clf3)]

    estimators_custom = [('dt', clone(clf1)), ('rf', clone(clf2)), ('nb', clone(clf3))]

    voting_sklearn = VotingClassifier(estimators=estimators_sklearn, voting='hard')
    voting_custom = MyVotingClassifier(estimators=estimators_custom, voting='hard', verbose=True)

    print("\n--- Entraînement sklearn.ensemble.VotingClassifier ---")
    start = time.time()
    voting_sklearn.fit(X_train, y_train)
    print(f"-> Terminé en {time.time() - start:.2f} secondes")

    print("\n--- Entraînement MyVotingClassifier ---")
    start = time.time()
    voting_custom.fit(X_train, y_train)
    print(f"-> Terminé en {time.time() - start:.2f} secondes (total global)")

    print("\n--- Phase de Prédictions ---")
    start = time.time()
    y_pred_sk = voting_sklearn.predict(X_test)
    print(f"Prédictions sklearn terminées en {time.time() - start:.2f} secondes")

    start = time.time()
    y_pred_my = voting_custom.predict(X_test)
    print(f"Prédictions MyVotingClassifier terminées en {time.time() - start:.2f} secondes")

    acc_sk = accuracy_score(y_test, y_pred_sk)
    acc_my = accuracy_score(y_test, y_pred_my)

    print("\n--- RÉSULTATS ---")
    print(f"Accuracy VotingClassifier (sklearn) : {acc_sk * 100:.2f}%")
    print(f"Accuracy MyVotingClassifier         : {acc_my * 100:.2f}%")

    if np.array_equal(y_pred_sk, y_pred_my):
         print("\n=> SUCCES TOTAL : Les deux classifieurs renvoient EXACTEMENT les mêmes prédictions !")
    else:
         diff = np.sum(y_pred_sk != y_pred_my)
         print(f"\n=> ATTENTION : Il y a des différences sur {diff} prédictions.")

if __name__ == "__main__":
    main()
