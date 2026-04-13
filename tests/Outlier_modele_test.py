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
from classes.MyT_SNE import MyTSNE

def evaluer_modele_sur_donnees(X_custom, y_custom, titre="", afficher_graphique=False):
    print(f"\n{'='*50}\n{titre}\n{'='*50}")
    nb_normaux = np.sum(y_custom == 1)
    nb_anomalies = np.sum(y_custom == -1)
    print(f"Jeu de données : {nb_normaux} normaux, {nb_anomalies} anomalies.\n")

    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_custom, y_custom, test_size=0.2, random_state=42, stratify=y_custom)

    norm = Normaliseur(methode='auto')
    X_train_norm = norm.fit_transform(X_train)
    X_test_norm = norm.transform(X_test)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_test_pca = pca.transform(X_test_norm)

    chercheur_norm = Trouve_params(X_train_norm, y_train, cv=3, verbose=0)
    chercheur_pca = Trouve_params(X_train_pca, y_train, cv=3, verbose=0)

    IF_opti = chercheur_norm.trouve_params(IsolationForest(random_state=42))
    IF_pred = IF_opti.predict(X_test_norm)
    score_if = f1_score(y_test, IF_pred, average='macro') * 100

    LOF_opti = chercheur_norm.trouve_params(LocalOutlierFactor(novelty=True))
    LOF_pred = LOF_opti.predict(X_test_norm)
    score_lof = f1_score(y_test, LOF_pred, average='macro') * 100

    EE_opti = chercheur_pca.trouve_params(EllipticEnvelope(random_state=42))
    EE_pred = EE_opti.predict(X_test_pca)
    score_ee = f1_score(y_test, EE_pred, average='macro') * 100

    print(f" Score Individuels => IF : {score_if:.2f}% | LOF : {score_lof:.2f}% | EE : {score_ee:.2f}%")

    from sklearn.pipeline import Pipeline
    EE_pipeline = Pipeline([
        ('pca', PCA(n_components=0.95, random_state=42)),
        ('ee', EE_opti)
    ])
    EE_pipeline.fit(X_train_norm)

    # VOTE HARD
    vote_hard = MyVotingOutlier(estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], voting='hard', verbose=False)
    vote_hard.fit(X_train_norm, y_train)
    vote_pred_hard = vote_hard.predict(X_test_norm)
    score_vote_hard = f1_score(y_test, vote_pred_hard, average='macro') * 100

    # VOTE SOFT
    vote_soft = MyVotingOutlier(estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], voting='soft', verbose=False)
    vote_soft.fit(X_train_norm, y_train)
    vote_pred_soft = vote_soft.predict(X_test_norm)
    score_vote_soft = f1_score(y_test, vote_pred_soft, average='macro') * 100

    # VOTE S&F
    vote_sf = MyVotingOutlier(estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], voting='S&F', verbose=False, sf_metric='f1')
    vote_sf.fit(X_train_norm, y_train)
    vote_pred_sf = vote_sf.predict(X_test_norm)
    score_vote_sf = f1_score(y_test, vote_pred_sf, average='macro') * 100

    print(f"\n --- Scores Voting Classifier ---")
    print(f" => Score Vote HARD : {score_vote_hard:.2f}%")
    print(f" => Score Vote SOFT : {score_vote_soft:.2f}%")
    print(f" => Score Vote S&F  : {score_vote_sf:.2f}%")
    print(f" [Poids S&F] IF={vote_sf.sf_weights_[0]*100:.1f}%, LOF={vote_sf.sf_weights_[1]*100:.1f}%, EE={vote_sf.sf_weights_[2]*100:.1f}%")
    
    if afficher_graphique:
        print("\n[Visualisation] Lancement de T-SNE sur les données de test...")
        # Application de MyTSNE pour observation visuelle
        tsne = MyTSNE(n_components=2, max_iter=500, perplexity=30.0, random_state=42)
        tsne.fit_transform(X_test_norm)
        
        # --- NOUVEAU : Création d'un dictionnaire avec toutes nos prédictions ---
        dictionnaire_pour_ui = {
            "Isolation Forest": IF_pred,
            "Local Outlier Factor": LOF_pred,
            "Elliptic Envelope": EE_pred,
            "Vote HARD": vote_pred_hard,
            "Vote SOFT": vote_pred_soft,
            "Vote S&F": vote_pred_sf
        }
        
        # Affichage interactif avec RadioButtons !
        print("[Visualisation] Ouverture de la fenêtre interactive T-SNE... Cliquez sur les boutons à gauche.")
        tsne.afficher_interactif_anomalies(dict_predictions=dictionnaire_pour_ui, y_true=y_test)


def test_outlier_models():
    print("Chargement de la grille complète d'Outliers...")
    grille_complete, _ = ChargeurDonneesPourOutlier.charger_grille_anomalies("Digits", random_state=42)

    configs_classe_0 = grille_complete['1']
    for nom_config, (X_custom, y_custom) in configs_classe_0.items():
        # On n'affiche le graphique QUE pour la première configuration (100pct) pour éviter d'inonder le testeur de fenêtres matplotlib
        doit_afficher = (nom_config == "100pct_2ano")
        evaluer_modele_sur_donnees(X_custom, y_custom, titre=f"TEST DE LA CLASSE '1' - Config: {nom_config}", afficher_graphique=doit_afficher)

if __name__ == "__main__":
    test_outlier_models()