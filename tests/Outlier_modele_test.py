# Fichier pour tester les modeles d'outliers ellipsenvelope, isolation forest et local outlier factor
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings

# Ajouter la racine du projet au path pour pouvoir importer "classes" depuis le dossier "tests"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score, f1_score

from classes.utils.ChargeurDonnees import ChargeurDonnees
from classes.utils.ChargeurDonneesPourOutlier import ChargeurDonneesPourOutlier
from classes.utils.Trouve_params import Trouve_params
from classes.utils.Normaliseur import Normaliseur
from classes.MyVotingOutlier import MyVotingOutlier

def test_outlier_models():
    # 1. Chargement des données à l'aide du nouveau gestionnaire d'Outliers paramétrable
    X_custom, y_custom, _ = ChargeurDonneesPourOutlier.charger(
        nom_dataset="Digits", 
        classe_normale='8', 
        pourcentage_normaux=100, 
        nb_anomalies_par_classe=2
    )

    nb_normaux = np.sum(y_custom == 1)
    nb_anomalies = np.sum(y_custom == -1)
    print(f" Jeu de données custom généré : {nb_normaux} normaux, {nb_anomalies} anomalies.\n")

    # On supprime les warnings de matrice singulière
    warnings.simplefilter("ignore", UserWarning)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_custom, y_custom, test_size=0.2, random_state=42, stratify=y_custom)

    print(f" Répartition : Entraînement {len(y_train)} | Test {len(y_test)}\n")

    # Application du Normaliseur sur les données 
    # fit_transform sur l'entraînement, transform sur le test
    norm = Normaliseur(methode='auto')
    X_train_norm = norm.fit_transform(X_train)
    X_test_norm = norm.transform(X_test)
    print(f" Données normalisées via la méthode : {norm.methode}")

    from sklearn.decomposition import PCA
    # Réduction de dimension avec PCA : UNIQUEMENT pour Elliptic Envelope
    pca = PCA(n_components=0.95, random_state=42) # Garder 95% de l'information
    X_train_pca = pca.fit_transform(X_train_norm)
    X_test_pca = pca.transform(X_test_norm)
    print(f" Dimensions réduites avec PCA : de {X_custom.shape[1]} à {X_train_pca.shape[1]} composantes (Pour Elliptic Envelope uniquement).\n")

    # cv=3 pour la recherche d'hyperparamètres et verbose=0 pour cacher le spam de [CV] END ...
    chercheur_norm = Trouve_params(X_train_norm, y_train, cv=3, verbose=0)
    chercheur_pca = Trouve_params(X_train_pca, y_train, cv=3, verbose=0)

    print("--- Recherche pour Isolation Forest ---")
    IF_brut = IsolationForest(random_state=42)
    IF_opti = chercheur_norm.trouve_params(IF_brut)
    IF_pred = IF_opti.predict(X_test_norm)
    score_if = f1_score(y_test, IF_pred, average='macro') * 100
    print(f" Score IF : Accuracy = {accuracy_score(y_test, IF_pred)*100:.2f}% | F1-Score = {score_if:.2f}%")        

    print("\n--- Recherche pour Local Outlier Factor ---")
    LOF_brut = LocalOutlierFactor(novelty=True)
    LOF_opti = chercheur_norm.trouve_params(LOF_brut)
    LOF_pred = LOF_opti.predict(X_test_norm)
    score_lof = f1_score(y_test, LOF_pred, average='macro') * 100
    print(f" Score LOF : Accuracy = {accuracy_score(y_test, LOF_pred)*100:.2f}% | F1-Score = {score_lof:.2f}%")     

    print("\n--- Recherche pour Elliptic Envelope ---")
    EE_brut = EllipticEnvelope(random_state=42)
    EE_opti = chercheur_pca.trouve_params(EE_brut)
    
    EE_pred = EE_opti.predict(X_test_pca)
    score_ee = f1_score(y_test, EE_pred, average='macro') * 100
    print(f" Score EE : Accuracy = {accuracy_score(y_test, EE_pred)*100:.2f}% | F1-Score = {score_ee:.2f}%")

    print("\n--- Récapitulatif des Modèles (Min, Max, Avg) ---")
    scores = {
        "Isolation Forest": score_if,
        "Local Outlier Factor": score_lof,
        "Elliptic Envelope": score_ee
    }
    
    min_model = min(scores, key=scores.get)
    max_model = max(scores, key=scores.get)
    avg_score = sum(scores.values()) / len(scores)
    
    print(f" Modèle MIN (Le plus mauvais) : {min_model} avec {scores[min_model]:.2f}%")
    print(f" Modèle MAX (Le meilleur)     : {max_model} avec {scores[max_model]:.2f}%")
    print(f" Score AVG (Moyenne des 3)    : {avg_score:.2f}%\n")
    
    # === Test des 3 modes de MyVotingOutlier (HARD, SOFT, S&F) ===
    # /!\ ATTENTION: Etant donné que EE tourne sur la PCA et pas les autres, MyVotingOutlier va se retrouver avec une seule entrée (X_custom) au lieu de deux !
    # Pour simuler que chaque modèle recoit ses propres données, nous allons lui passer le X_test_norm. 
    # Mettre en place un wrapper sklearn Pipeline serait l'idéal pour l'EllipticEnvelope :
    from sklearn.pipeline import Pipeline
    EE_pipeline = Pipeline([
        ('pca', PCA(n_components=0.95, random_state=42)),
        ('ee', EE_opti)
    ])
    # Entrainement du pipeline complet
    EE_pipeline.fit(X_train_norm)

    print("--- 1. Test avec MyVotingOutlier (Mode HARD - Vote Majoritaire) ---")
    vote_classifieur_hard = MyVotingOutlier(
        estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], 
        voting='hard', verbose=False
    )
    vote_classifieur_hard.fit(X_train_norm, y_train)
    vote_pred_hard = vote_classifieur_hard.predict(X_test_norm)
    score_vote_hard = f1_score(y_test, vote_pred_hard, average='macro') * 100
    print(f" Score MyVotingOutlier HARD : Accuracy = {accuracy_score(y_test, vote_pred_hard)*100:.2f}% | F1-Score = {score_vote_hard:.2f}%")
    
    confiance = vote_classifieur_hard.score_confiance(X_test_norm)
    unanimite = sum(1 for c in confiance if c['taux_confiance'] == 100.0)
    print(f" -> Prédictions où les 3 modèles sont UNANIMES : {unanimite} sur {len(X_test_norm)}\n")


    print("--- 2. Test avec MyVotingOutlier (Mode SOFT - Conversion des scores en Probabilités) ---")
    vote_classifieur_soft = MyVotingOutlier(
        estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], 
        voting='soft', verbose=False
    )
    vote_classifieur_soft.fit(X_train_norm, y_train)
    vote_pred_soft = vote_classifieur_soft.predict(X_test_norm)
    score_vote_soft = f1_score(y_test, vote_pred_soft, average='macro') * 100
    print(f" Score MyVotingOutlier SOFT : Accuracy = {accuracy_score(y_test, vote_pred_soft)*100:.2f}% | F1-Score = {score_vote_soft:.2f}%\n")

    print("--- 3. Test avec MyVotingOutlier (Mode SOURCE & FAITS) ---")
    vote_classifieur_sf = MyVotingOutlier(
        estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], 
        voting='S&F', verbose=False, sf_metric='f1'
    )
    vote_classifieur_sf.fit(X_train_norm, y_train)
    vote_pred_sf = vote_classifieur_sf.predict(X_test_norm)
    score_vote_sf = f1_score(y_test, vote_pred_sf, average='macro') * 100
    print(f" Score MyVotingOutlier S&F : Accuracy = {accuracy_score(y_test, vote_pred_sf)*100:.2f}% | F1-Score = {score_vote_sf:.2f}%")

    print(f" -> Poids de 'Fiabilité' attribués aux modèles par l'algorithme S&F :")
    for i, (name, _) in enumerate(vote_classifieur_sf.estimators):
        print(f"     - {name:^3} : {vote_classifieur_sf.sf_weights_[i]*100:.1f}% de fiabilité")

test_outlier_models() 
