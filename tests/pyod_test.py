import numpy as np
from sklearn.linear_model import LinearRegression
from pyod.models.iforest import IForest
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold

np.random.seed(42)

temps_travail_train = np.random.uniform(2,30,200).reshape(-1, 1)
notes_eleves_train = (temps_travail_train * 0.7 + np.random.normal(3, 2, 200).reshape(-1, 1)).flatten()

temps_travail_train = np.clip(temps_travail_train, 0, 50)
notes_eleves_train = np.clip(notes_eleves_train, 0, 20)

print("--- 1. etape Scikit-Learn ---")
modele_sk = LinearRegression()
modele_sk.fit(temps_travail_train, notes_eleves_train)
print(f"La regle apprise est : Note = Temps * {modele_sk.coef_[0]:.2f} + {modele_sk.intercept_:.2f}")

temps_test = np.array([15, 3, 45, 25, 10, 20, 0, 30, 5, 12, 40, 50]).reshape(-1, 1)
notes_reelles = np.array([13, 19, 2, 18, 10, 15, 18, 6, 5, 11, 20, 1])

notes_predites = np.clip(modele_sk.predict(temps_test), 0, 20)

erreurs = np.abs(notes_reelles - notes_predites)

print("\n--- 2. etape PyOD  ---")

X_train_complet = np.column_stack((temps_travail_train, notes_eleves_train))

detecteur_pyod = IForest(contamination=0.05, random_state=42)
detecteur_pyod.fit(X_train_complet)

X_test = np.column_stack((temps_test, notes_reelles))

predictions_pyod = detecteur_pyod.predict(X_test)
scores_pyod = detecteur_pyod.decision_function(X_test)

print("\nResultats :")
for i in range(len(temps_test)):
    t = temps_test[i][0]
    n_reel = notes_reelles[i]
    n_predit = notes_predites[i]
    err = erreurs[i]
    statut = "[Anomalie]" if predictions_pyod[i] == 1 else "[Normal]"

    print(f"Eleve {i+1} [Temps: {t}h | Note reelle: {n_reel}/20]")
    print(f"  -> Modele Scikit-Learn : S'attendait a {n_predit:.1f}/20 (Erreur de {err:.1f} points)")
    print(f"  -> Analyse PyOD        : {statut} (Score d'anomalie : {scores_pyod[i]:.2f})")
    print("-" * 50)

print("\n--- 3. etape Evaluation des performances ---")

vrai_statut = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1])

print("Realite attendue :", vrai_statut)
print("Predictions PyOD :", predictions_pyod)

print("\n--- EXPLICATIONS DES SCORES ---")
print(f"PRECISION : {precision_score(vrai_statut, predictions_pyod, zero_division=0):.2f}")
print(" -> 'Parmi tous les eleves que PyOD a denonce, combien trichent vraiment ?'")
print(" -> Plus proche de 1 = moins de bonnes valeurs dites comme anomalies (Faux Positifs).\n")

print(f"RAPPEL    : {recall_score(vrai_statut, predictions_pyod, zero_division=0):.2f}")
print(" -> 'Parmi toutes les VRAIES anomalies qui existent, combien PyOD a reussi a attraper ?'")
print(" -> Plus proche de 1 = moins de mauvaises valeurs dites comme normales (Faux Negatifs).\n")

print(f"F1-SCORE  : {f1_score(vrai_statut, predictions_pyod, zero_division=0):.2f}")
print(" -> 'La moyenne globale de l'algorithme.'")
print(" -> Moyenne entre Precision et Rappel.\n")

print("\nRapport detaille :")
print(classification_report(vrai_statut, predictions_pyod, target_names=["Normal", "Anomalie"], zero_division=0))

print("\n--- 4. etape Cross Validation ---")
print("1. Scikit-learn calcule la marge d'erreur pour 100 nouveaux eleves.")
print("2. PyOD isole les erreurs les plus anormales.")
print("3. La cross validation certifie le score final.")

X_cv_temps_normaux = np.random.uniform(10, 40, 80).reshape(-1, 1)
notes_normales = (X_cv_temps_normaux * 0.7 + np.random.normal(3, 1.5, 80).reshape(-1, 1)).flatten()

temps_genies = np.random.uniform(0, 5, 10).reshape(-1, 1)
notes_genies = np.random.uniform(18, 20, 10)

temps_echec = np.random.uniform(45, 50, 10).reshape(-1, 1)
notes_echec = np.random.uniform(0, 4, 10)

X_cv_temps = np.vstack((X_cv_temps_normaux, temps_genies, temps_echec))
X_cv_notes = np.clip(np.concatenate((notes_normales, notes_genies, notes_echec)), 0, 20)

y_cv = np.concatenate((np.zeros(80), np.ones(20)))

notes_predites_cv = np.clip(modele_sk.predict(X_cv_temps), 0, 20)
erreurs_cv = np.abs(X_cv_notes - notes_predites_cv).reshape(-1, 1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
detecteur_cv = IForest(contamination=0.20, random_state=42)

scores_cv = []
for train_index, test_index in skf.split(erreurs_cv, y_cv):
    err_train, err_test = erreurs_cv[train_index], erreurs_cv[test_index]
    y_test_vrai = y_cv[test_index]

    detecteur_cv.fit(err_train)

    predictions_test = detecteur_cv.predict(err_test)
    scores_cv.append(f1_score(y_test_vrai, predictions_test, zero_division=0))

print("\nScores sur chaque bloc testé :")
for i, sc in enumerate(scores_cv):
    print(f" - Bloc {i+1} : {sc:.2f}")

print(f"\n=> Score final: {np.mean(scores_cv):.2f}")
