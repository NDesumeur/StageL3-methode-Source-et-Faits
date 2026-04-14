import streamlit as st
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

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

CACHE_FILE = os.path.join(os.path.dirname(__file__), "cache_evaluations.joblib")

def extraire_metriques(y_true, y_pred):
    return {
        'f1': f1_score(y_true, y_pred, average='macro') * 100,
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'accuracy': accuracy_score(y_true, y_pred) * 100
    }

def executer_evaluation(classes_a_tester, configs_a_tester):
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)
    
    if os.path.exists(CACHE_FILE):
        cache_resultats = joblib.load(CACHE_FILE)
    else:
        cache_resultats = {}
        
    grille_complete, _ = ChargeurDonneesPourOutlier.charger_grille_anomalies("Digits", random_state=42)
    
    resultats_finaux = []
    combinaisons_a_faire = []
    
    for c in classes_a_tester:
        for conf in configs_a_tester:
            cle = f"{c}_{conf}"
            if cle in cache_resultats:
                resultats_finaux.append(cache_resultats[cle])
            else:
                combinaisons_a_faire.append((c, conf, cle))
                
    if combinaisons_a_faire:
        barre = st.progress(0)
        texte_statut = st.empty()
        total = len(combinaisons_a_faire)
        
        for i, (classe_cible, nom_config, cle) in enumerate(combinaisons_a_faire):
            texte_statut.text(f"Calcul en direct : Classe {classe_cible} | Config {nom_config} ({i+1}/{total})")
            
            configs = grille_complete.get(classe_cible, {})
            if nom_config not in configs:
                continue
                
            X_custom, y_custom = configs[nom_config]
            
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
            m_if = {
                'train': extraire_metriques(y_train, IF_opti.predict(X_train_norm)),
                'test': extraire_metriques(y_test, IF_opti.predict(X_test_norm))
            }
            
            LOF_opti = chercheur_norm.trouve_params(LocalOutlierFactor(novelty=True))
            m_lof = {
                'train': extraire_metriques(y_train, LOF_opti.predict(X_train_norm)),
                'test': extraire_metriques(y_test, LOF_opti.predict(X_test_norm))
            }
            
            EE_base_opti = chercheur_pca.trouve_params(EllipticEnvelope(random_state=42))
            EE_pipeline = Pipeline([('pca', PCA(n_components=0.95, random_state=42)), ('ee', EE_base_opti)])
            EE_pipeline.fit(X_train_norm)
            m_ee = {
                'train': extraire_metriques(y_train, EE_pipeline.predict(X_train_norm)),
                'test': extraire_metriques(y_test, EE_pipeline.predict(X_test_norm))
            }
            
            m_min = {'train':{}, 'test':{}}
            m_max = {'train':{}, 'test':{}}
            m_avg = {'train':{}, 'test':{}}
            for phase in ['train', 'test']:
                for k in ['f1', 'precision', 'recall', 'accuracy']: 
                    vals = [m_if[phase][k], m_lof[phase][k], m_ee[phase][k]]
                    m_min[phase][k] = min(vals)
                    m_max[phase][k] = max(vals)
                    m_avg[phase][k] = sum(vals)/3

            vote_hard = MyVotingOutlier(estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], voting='hard', verbose=False)
            vote_hard.fit(X_train_norm, y_train)
            m_hard = {
                'train': extraire_metriques(y_train, vote_hard.predict(X_train_norm)),
                'test': extraire_metriques(y_test, vote_hard.predict(X_test_norm))
            }
            
            vote_soft = MyVotingOutlier(estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], voting='soft', verbose=False)
            vote_soft.fit(X_train_norm, y_train)
            m_soft = {
                'train': extraire_metriques(y_train, vote_soft.predict(X_train_norm)),
                'test': extraire_metriques(y_test, vote_soft.predict(X_test_norm))
            }
            
            vote_sf = MyVotingOutlier(estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], voting='S&F', verbose=False, sf_metric='f1')
            vote_sf.fit(X_train_norm, y_train)
            m_sf = {
                'train': extraire_metriques(y_train, vote_sf.predict(X_train_norm)),
                'test': extraire_metriques(y_test, vote_sf.predict(X_test_norm))
            }
            
            res = {
                'classe': classe_cible,
                'config': nom_config,
                'metriques': {
                    'Isolation Forest': m_if, 'Local Outlier Factor': m_lof, 'Elliptic Envelope': m_ee,
                    'Modèle MIN': m_min, 'Modèle MAX': m_max, 'Modèle AVG': m_avg,
                    'Vote HARD': m_hard, 'Vote SOFT': m_soft, 'Vote S&F': m_sf
                }
            }
            
            cache_resultats[cle] = res
            resultats_finaux.append(res)
            barre.progress((i + 1) / total)
            
        joblib.dump(cache_resultats, CACHE_FILE)
        barre.empty()
        texte_statut.empty()
        
    return resultats_finaux

def main():
    st.set_page_config(page_title="Evaluation d'Anomalies Borda", layout="wide")
    st.title("Comparatif des Modèles (Train vs Test)")
    
    etat_cache = "Données sauvegardées localement" if os.path.exists(CACHE_FILE) else "Cache vide (Calculs à venir)"
    st.markdown(f"Ajustez vos filtres dans le menu à gauche, puis cliquez sur **Lancer l'évaluation**. *({etat_cache})*")

    classes_possibles = [str(i) for i in range(10)]
    configs_possibles = ['100pct_2ano', '50pct_2ano', '25pct_1ano', '10pct_1ano']

    st.sidebar.header("Configuration du Test")
    choix_classe = st.sidebar.selectbox("Sélection de la Classe à tester :", ["Toutes les Classes"] + classes_possibles)
    choix_config = st.sidebar.selectbox("Sélection de la Configuration (Ratio) :", ["Toutes les Configurations"] + configs_possibles)
    
    bouton_lancer = st.sidebar.button("Lancer l'évaluation", type="primary")
    bouton_vider = st.sidebar.button("Vider les données (Nettoyer cache)")
    
    if bouton_vider:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            st.sidebar.success("Toutes les sauvegardes ont été purgées !")
            st.rerun()

    if bouton_lancer:
        classes_a_tester = classes_possibles if choix_classe == "Toutes les Classes" else [choix_classe]
        configs_a_tester = configs_possibles if choix_config == "Toutes les Configurations" else [choix_config]
        
        donnees_entrainees = executer_evaluation(classes_a_tester, configs_a_tester)
        
        if not donnees_entrainees:
            st.error("Génération des données échouée.")
            return

        noms_candidats = list(donnees_entrainees[0]['metriques'].keys())
        liste_scores_config_pour_borda = []
        historique = {c: {'train': {'f1': [], 'precision': [], 'recall': [], 'accuracy': []}, 
                          'test': {'f1': [], 'precision': [], 'recall': [], 'accuracy': []}} for c in noms_candidats}
        
        for passage in donnees_entrainees:
            dict_borda = {}
            for candidat, ses_metriques in passage['metriques'].items():
                dict_borda[candidat] = ses_metriques['test']['f1']
                
                historique[candidat]['train']['f1'].append(ses_metriques['train']['f1'])
                historique[candidat]['train']['precision'].append(ses_metriques['train']['precision'])
                historique[candidat]['train']['recall'].append(ses_metriques['train']['recall'])
                historique[candidat]['train']['accuracy'].append(ses_metriques['train']['accuracy'])
                
                historique[candidat]['test']['f1'].append(ses_metriques['test']['f1'])
                historique[candidat]['test']['precision'].append(ses_metriques['test']['precision'])
                historique[candidat]['test']['recall'].append(ses_metriques['test']['recall'])
                historique[candidat]['test']['accuracy'].append(ses_metriques['test']['accuracy'])
                
            liste_scores_config_pour_borda.append(dict_borda)

        classement_final = CalculateurBorda.calculer(liste_scores_config_pour_borda)
        
        df_rows = []
        position = 1
        for candidat, points in classement_final.items():
            df_rows.append({
                "Position": str(position) + ("er" if position == 1 else "ème"),
                "Modèle": candidat,
                "Points Borda": points,
                "F1 Train (%)": round(np.mean(historique[candidat]['train']['f1']), 1),
                "F1 Test (%)": round(np.mean(historique[candidat]['test']['f1']), 1),
                "Pré Train (%)": round(np.mean(historique[candidat]['train']['precision']), 1),
                "Pré Test (%)": round(np.mean(historique[candidat]['test']['precision']), 1),
                "Rap Train (%)": round(np.mean(historique[candidat]['train']['recall']), 1),
                "Rap Test (%)": round(np.mean(historique[candidat]['test']['recall']), 1),
                "Acc Train (%)": round(np.mean(historique[candidat]['train']['accuracy']), 1),
                "Acc Test (%)": round(np.mean(historique[candidat]['test']['accuracy']), 1),
            })
            position += 1

        df_borda = pd.DataFrame(df_rows)
        df_borda.set_index("Position", inplace=True)
        
        st.subheader(f"Tableau de Résultats - {len(donnees_entrainees)} Itérations Évaluées")
        
        st.dataframe(df_borda.style.highlight_max(subset=["Points Borda", "F1 Test (%)"], color='#c6e0b4')
                                   .highlight_min(subset=["Points Borda", "F1 Test (%)"], color='#f8cbcb')
                                   .format(precision=1),
                     use_container_width=True)

        csv = df_borda.to_csv(sep=';', decimal=',').encode('utf-8')
        st.download_button(
            label="Télécharger ce tableau en format Excel / CSV",
            data=csv,
            file_name=f'Extraction_Metriques_{choix_classe}_{choix_config}.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()