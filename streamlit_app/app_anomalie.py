import streamlit as st
import os
import sys
import warnings
import numpy as np
import pandas as pd

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

# Le décorateur cache_data mémorise les résultats pour ne pas recalculer 2 fois la même chose
@st.cache_data(show_spinner=False)
def executer_evaluation(classes_a_tester, configs_a_tester):
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)
    
    grille_complete, _ = ChargeurDonneesPourOutlier.charger_grille_anomalies("Digits", random_state=42)
    
    resultats = []
    total_iterations = len(classes_a_tester) * len(configs_a_tester)
    
    barre = st.progress(0)
    texte_statut = st.empty()
    
    compteur = 0
    for classe_cible in classes_a_tester:
        if classe_cible not in grille_complete:
            continue
        configs = grille_complete[classe_cible]
        
        for nom_config in configs_a_tester:
            if nom_config not in configs:
                continue
                
            texte_statut.text(f"Entraînement en cours : Classe {classe_cible} | Config {nom_config} ({compteur+1}/{total_iterations})")
            
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
            
            resultats.append({
                'classe': classe_cible,
                'config': nom_config,
                'metriques': resultats_courants
            })
            
            compteur += 1
            barre.progress(compteur / total_iterations)
            
    barre.empty()
    texte_statut.empty()
    return resultats

def main():
    st.set_page_config(page_title="Evaluation d'Anomalies Borda", layout="wide")
    st.title("Classement Borda : Modèles de Détection d'Anomalies")
    st.markdown("Ajustez vos filtres dans le menu à gauche, puis cliquez sur **Lancer l'évaluation**.")

    # Options statiques connues
    classes_possibles = [str(i) for i in range(10)]
    configs_possibles = ['100pct_2ano', '50pct_2ano', '25pct_1ano', '10pct_1ano']

    # 1. Menu latéral fixe (affiché instantanément)
    st.sidebar.header("Configuration du Test")
    choix_classe = st.sidebar.selectbox("Sélection de la Classe à tester :", ["Toutes les Classes"] + classes_possibles)
    choix_config = st.sidebar.selectbox("Sélection de la Configuration (Ratio) :", ["Toutes les Configurations"] + configs_possibles)
    
    bouton_lancer = st.sidebar.button("Lancer l'évaluation", type="primary")

    # 2. Action déclenchée manuellement
    if bouton_lancer:
        # Filtrage ciblé (on ne calcule que ce qui est demandé)
        classes_a_tester = classes_possibles if choix_classe == "Toutes les Classes" else [choix_classe]
        configs_a_tester = configs_possibles if choix_config == "Toutes les Configurations" else [choix_config]
        
        # Les listes sont castées en tuple pour le cache Streamlit
        donnees_entrainees = executer_evaluation(tuple(classes_a_tester), tuple(configs_a_tester))
        
        if not donnees_entrainees:
            st.error("Génération des données échouée.")
            return

        # 3. Borda & Pandas Dataframe
        noms_candidats = list(donnees_entrainees[0]['metriques'].keys())
        liste_scores_config_pour_borda = []
        historique_des_metriques = {c: {'f1': [], 'precision': [], 'recall': [], 'accuracy': []} for c in noms_candidats}
        
        for passage in donnees_entrainees:
            dict_reconstruit = {}
            for candidat, ses_metriques in passage['metriques'].items():
                dict_reconstruit[candidat] = ses_metriques['f1']
                historique_des_metriques[candidat]['f1'].append(ses_metriques['f1'])
                historique_des_metriques[candidat]['precision'].append(ses_metriques['precision'])
                historique_des_metriques[candidat]['recall'].append(ses_metriques['recall'])
                historique_des_metriques[candidat]['accuracy'].append(ses_metriques['accuracy'])
                
            liste_scores_config_pour_borda.append(dict_reconstruit)

        classement_final = CalculateurBorda.calculer(liste_scores_config_pour_borda)
        
        df_rows = []
        position = 1
        for candidat, points in classement_final.items():
            df_rows.append({
                "Position": str(position) + ("er" if position == 1 else "ème"),
                "Modèle": candidat,
                "Points Borda": points,
                "F1-Score (%)": round(np.mean(historique_des_metriques[candidat]['f1']), 2),
                "Précision (%)": round(np.mean(historique_des_metriques[candidat]['precision']), 2),
                "Rappel (%)": round(np.mean(historique_des_metriques[candidat]['recall']), 2),
                "Accuracy (%)": round(np.mean(historique_des_metriques[candidat]['accuracy']), 2)
            })
            position += 1

        df_borda = pd.DataFrame(df_rows)
        df_borda.set_index("Position", inplace=True)
        
        st.subheader(f" Tableau de Résultats - {len(donnees_entrainees)} Confrontations Evaluées")
        st.write(f"- Classe Ciblée : {choix_classe}\n- Difficulté Sélectionnée : {choix_config}")
        
        st.dataframe(df_borda.style.highlight_max(subset=["Points Borda", "F1-Score (%)"], color='#c6e0b4')
                                   .highlight_min(subset=["Points Borda", "F1-Score (%)"], color='#f8cbcb'),
                     use_container_width=True)

if __name__ == "__main__":
    main()