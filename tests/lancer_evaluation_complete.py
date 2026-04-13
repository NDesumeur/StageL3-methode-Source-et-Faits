import os
import sys
import warnings
import numpy as np
import joblib
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from classes.utils.ChargeurDonneesPourOutlier import ChargeurDonneesPourOutlier
from classes.utils.Normaliseur import Normaliseur
from classes.utils.Trouve_params import Trouve_params
from classes.MyVotingOutlier import MyVotingOutlier
from classes.utils.Borda import CalculateurBorda

def extraire_metriques(y_true, y_pred):
    return {
        'f1': f1_score(y_true, y_pred, average='macro') * 100,
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'accuracy': accuracy_score(y_true, y_pred) * 100
    }

def lancer_evaluation():
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)

    dossier_sauvegarde = os.path.join(os.path.dirname(__file__), "..", "modeles_sauvegardes")
    os.makedirs(dossier_sauvegarde, exist_ok=True)

    print("1. Chargement en mémoire des 40 configurations...")
    grille_complete, _ = ChargeurDonneesPourOutlier.charger_grille_anomalies("Digits", random_state=42)
    
    noms_candidats = [
        'Isolation Forest', 'Local Outlier Factor', 'Elliptic Envelope', 
        'Modèle MIN', 'Modèle MAX', 'Modèle AVG', 
        'Vote HARD', 'Vote SOFT', 'Vote S&F'
    ]
    
    liste_scores_global = []
    historique_metriques_global = {c: {'f1': [], 'precision': [], 'recall': [], 'accuracy': []} for c in noms_candidats}
    
    liste_scores_par_config = {}
    historique_metriques_par_config = {}
    
    print("2. Lancement des entraînements et des évaluations...")
    total_iterations = len(grille_complete.keys()) * 4
    with tqdm(total=total_iterations, desc="Progression globale") as pbar:
        for classe_cible, configs in grille_complete.items():
            for nom_config, (X_custom, y_custom) in configs.items():
                
                if nom_config not in liste_scores_par_config:
                    liste_scores_par_config[nom_config] = []
                    historique_metriques_par_config[nom_config] = {c: {'f1': [], 'precision': [], 'recall': [], 'accuracy': []} for c in noms_candidats}
                
                X_train, X_test, y_train, y_test = train_test_split(X_custom, y_custom, test_size=0.2, random_state=42, stratify=y_custom)
                
                norm = Normaliseur(methode='standard')
                X_train_norm = norm.fit_transform(X_train)
                X_test_norm = norm.transform(X_test)
                
                pca = PCA(n_components=0.95, random_state=42)
                X_train_pca = pca.fit_transform(X_train_norm)
                X_test_pca = pca.transform(X_test_norm)
                
                chercheur_norm = Trouve_params(X_train_norm, y_train, cv=3, verbose=0)
                chercheur_pca = Trouve_params(X_train_pca, y_train, cv=3, verbose=0)
                
                IF_opti = chercheur_norm.trouve_params(IsolationForest(random_state=42))
                IF_pred = IF_opti.predict(X_test_norm)
                m_if = extraire_metriques(y_test, IF_pred)
                
                LOF_opti = chercheur_norm.trouve_params(LocalOutlierFactor(novelty=True))
                LOF_pred = LOF_opti.predict(X_test_norm)
                m_lof = extraire_metriques(y_test, LOF_pred)
                
                EE_base_opti = chercheur_pca.trouve_params(EllipticEnvelope(random_state=42))
                EE_pipeline = Pipeline([('pca', PCA(n_components=0.95, random_state=42)), ('ee', EE_base_opti)])
                EE_pipeline.fit(X_train_norm)
                EE_pred = EE_pipeline.predict(X_test_norm)
                m_ee = extraire_metriques(y_test, EE_pred)
                
                m_min = {k: min(m_if[k], m_lof[k], m_ee[k]) for k in m_if}
                m_max = {k: max(m_if[k], m_lof[k], m_ee[k]) for k in m_if}
                m_avg = {k: (m_if[k] + m_lof[k] + m_ee[k]) / 3 for k in m_if}
                
                vote_hard = MyVotingOutlier(estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], voting='hard', verbose=False)
                vote_hard.fit(X_train_norm, y_train)
                m_hard = extraire_metriques(y_test, vote_hard.predict(X_test_norm))
                
                vote_soft = MyVotingOutlier(estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], voting='soft', verbose=False)
                vote_soft.fit(X_train_norm, y_train)
                m_soft = extraire_metriques(y_test, vote_soft.predict(X_test_norm))
                
                vote_sf = MyVotingOutlier(estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], voting='S&F', verbose=False, sf_metric='f1')
                vote_sf.fit(X_train_norm, y_train)
                m_sf = extraire_metriques(y_test, vote_sf.predict(X_test_norm))
                
                resultats_courants = {
                    'Isolation Forest': m_if, 'Local Outlier Factor': m_lof, 'Elliptic Envelope': m_ee,
                    'Modèle MIN': m_min, 'Modèle MAX': m_max, 'Modèle AVG': m_avg,
                    'Vote HARD': m_hard, 'Vote SOFT': m_soft, 'Vote S&F': m_sf
                }
                
                scores_actuels = {k: metriques['f1'] for k, metriques in resultats_courants.items()}
                
                # Mise à jour globale
                liste_scores_global.append(scores_actuels)
                for k, metriques in resultats_courants.items():
                    for metrique_nom, valeur in metriques.items():
                        historique_metriques_global[k][metrique_nom].append(valeur)
                        
                # Mise à jour par configuration précise
                liste_scores_par_config[nom_config].append(scores_actuels)
                for k, metriques in resultats_courants.items():
                    for metrique_nom, valeur in metriques.items():
                        historique_metriques_par_config[nom_config][k][metrique_nom].append(valeur)
                
                modeles_a_sauver = {'IF': IF_opti, 'LOF': LOF_opti, 'EE': EE_pipeline, 'Voting_SF': vote_sf}
                chemin_fichier = os.path.join(dossier_sauvegarde, f"modeles_classe{classe_cible}_{nom_config}.joblib")
                joblib.dump(modeles_a_sauver, chemin_fichier)
                
                pbar.update(1)

    print("\n3. Calcul des classements par la méthode de Borda...\n")
    
    # Affichage pour chaque ratio / configuration spécifique
    for config_nom, scores_liste in liste_scores_par_config.items():
        classement = CalculateurBorda.calculer(scores_liste)
        
        moyennes = {}
        for candidat, mesures in historique_metriques_par_config[config_nom].items():
            moyennes[candidat] = {k: np.mean(v) for k, v in mesures.items()}
            
        print(f"\n ---> RÉSULTATS POUR LA CONFIGURATION : {config_nom.upper()} <---")
        CalculateurBorda.afficher_classement(classement, nb_confrontations=len(scores_liste), moyennes_metriques=moyennes)

    # Affichage global
    print("\n ---> RÉSULTATS GLOBAUX (TOUTES CONFIGURATIONS CONFONDUES) <---")
    classement_global = CalculateurBorda.calculer(liste_scores_global)
    moyennes_globales = {}
    for candidat, mesures in historique_metriques_global.items():
        moyennes_globales[candidat] = {k: np.mean(v) for k, v in mesures.items()}
        
    CalculateurBorda.afficher_classement(classement_global, nb_confrontations=len(liste_scores_global), moyennes_metriques=moyennes_globales)

if __name__ == "__main__":
    lancer_evaluation()