import streamlit as st
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classes.MyT_SNE import MyTSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler
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
            texte_statut.text(f"Calcul : Classe {classe_cible} | Config {nom_config} ({i+1}/{total})")
            
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
            
            # Un équilibre parfait entre trop peu de dimensions (perte de la forme)
            # et trop (matrice infinie pour EE) : 85% de la variance conservée.
            pca_medium = PCA(n_components=0.85, random_state=42)
            X_train_pca_medium = pca_medium.fit_transform(X_train_norm)
            X_test_pca_medium = pca_medium.transform(X_test_norm)
            
            norm_mm = MinMaxScaler()
            X_train_mm = norm_mm.fit_transform(X_train)
            X_test_mm = norm_mm.transform(X_test)
            
            chercheur_norm = Trouve_params(X_train_norm, y_train, cv=3, verbose=0)
            chercheur_pca = Trouve_params(X_train_pca, y_train, cv=3, verbose=0)
            chercheur_pca_medium = Trouve_params(X_train_pca_medium, y_train, cv=3, verbose=0)
            chercheur_mm = Trouve_params(X_train_mm, y_train, cv=3, verbose=0)
            
            IF_opti = chercheur_norm.trouve_params(IsolationForest(random_state=42))
            pred_if_train = IF_opti.predict(X_train_norm)
            pred_if_test = IF_opti.predict(X_test_norm)
            m_if = {
                'train': extraire_metriques(y_train, pred_if_train),
                'test': extraire_metriques(y_test, pred_if_test)
            }
            
            LOF_opti = chercheur_norm.trouve_params(LocalOutlierFactor(novelty=True))
            lof_pour_train = LocalOutlierFactor(n_neighbors=LOF_opti.n_neighbors, contamination=LOF_opti.contamination)
            pred_lof_train = lof_pour_train.fit_predict(X_train_norm)
            pred_lof_test = LOF_opti.predict(X_test_norm)
            m_lof = {
                'train': extraire_metriques(y_train, pred_lof_train),
                'test': extraire_metriques(y_test, pred_lof_test)
            }
            
            EE_base_opti = chercheur_pca_medium.trouve_params(EllipticEnvelope(random_state=42))
            EE_pipeline = Pipeline([('pca', PCA(n_components=0.85, random_state=42)), ('ee', EE_base_opti)])
            EE_pipeline.fit(X_train_norm)
            pred_ee_train = EE_pipeline.predict(X_train_norm)
            pred_ee_test = EE_pipeline.predict(X_test_norm)
            m_ee = {
                'train': extraire_metriques(y_train, pred_ee_train),
                'test': extraire_metriques(y_test, pred_ee_test)
            }
            
            # One-Class SVM Linear
            OCSVM_lin_base = chercheur_mm.trouve_params(OneClassSVM(kernel='linear'))
            # Pas de PCA pour le SVM Linéaire ! Le PCA centre les données sur 0,
            # forçant le SVM de type linéaire à couper le cluster normal en plein milieu.
            OCSVM_lin_pipeline = Pipeline([('scaler', MinMaxScaler()), ('ocsvm', OCSVM_lin_base)])
            OCSVM_lin_pipeline.fit(X_train_norm)
            pred_ocsvm_lin_train = OCSVM_lin_pipeline.predict(X_train_norm)
            pred_ocsvm_lin_test = OCSVM_lin_pipeline.predict(X_test_norm)
            m_ocsvm_lin = {
                'train': extraire_metriques(y_train, pred_ocsvm_lin_train),
                'test': extraire_metriques(y_test, pred_ocsvm_lin_test)
            }

            # One-Class SVM RBF
            OCSVM_rbf_base = chercheur_mm.trouve_params(OneClassSVM(kernel='rbf'))
            # Comme pour le SVM Linéaire, l'espace d'origine est crucial.
            # L'approche MinMax donne toutes les données brutes spatiales [0, 1] à l'algorithme.
            OCSVM_rbf_pipeline = Pipeline([('scaler', MinMaxScaler()), ('ocsvm', OCSVM_rbf_base)])
            OCSVM_rbf_pipeline.fit(X_train)
            pred_ocsvm_rbf_train = OCSVM_rbf_pipeline.predict(X_train)
            pred_ocsvm_rbf_test = OCSVM_rbf_pipeline.predict(X_test)
            m_ocsvm_rbf = {
                'train': extraire_metriques(y_train, pred_ocsvm_rbf_train),
                'test': extraire_metriques(y_test, pred_ocsvm_rbf_test)
            }
            
            m_min = {'train':{}, 'test':{}}
            m_max = {'train':{}, 'test':{}}
            m_avg = {'train':{}, 'test':{}}
            for phase in ['train', 'test']:
                for k in ['f1', 'precision', 'recall', 'accuracy']: 
                    vals = [m_if[phase][k], m_lof[phase][k], m_ee[phase][k], m_ocsvm_lin[phase][k], m_ocsvm_rbf[phase][k]]
                    m_min[phase][k] = min(vals)
                    m_max[phase][k] = max(vals)
                    m_avg[phase][k] = sum(vals)/5

            # -------- VOTE 3 (IF, LOF, EE) --------
            # Voting Hardware 3
            vote_hard_3 = MyVotingOutlier(estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], voting='hard', verbose=False)
            vote_hard_3.fit(X_train_norm, y_train)
            
            pred_hard_train_3 = vote_hard_3.predict(X_train_norm)
            pred_hard_test_3 = vote_hard_3.predict(X_test_norm)
            m_hard_3 = {'train': extraire_metriques(y_train, pred_hard_train_3), 'test': extraire_metriques(y_test, pred_hard_test_3)}
            
            # Voting Software 3
            vote_soft_3 = MyVotingOutlier(estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], voting='soft', verbose=False)
            vote_soft_3.fit(X_train_norm, y_train)
            
            pred_soft_train_3 = vote_soft_3.predict(X_train_norm)
            pred_soft_test_3 = vote_soft_3.predict(X_test_norm)
            m_soft_3 = {'train': extraire_metriques(y_train, pred_soft_train_3), 'test': extraire_metriques(y_test, pred_soft_test_3)}
            
            # Voting S&F 3
            vote_sf_3 = MyVotingOutlier(estimators=[('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline)], voting='S&F', verbose=False)
            vote_sf_3.fit(X_train_norm, y_train)
            
            pred_sf_train_3 = vote_sf_3.predict(X_train_norm)
            pred_sf_test_3 = vote_sf_3.predict(X_test_norm)
            m_sf_3 = {'train': extraire_metriques(y_train, pred_sf_train_3), 'test': extraire_metriques(y_test, pred_sf_test_3)}

            # -------- VOTE 5 (IF, LOF, EE, OCSVM Lin, OCSVM RBF) --------
            estimators_5 = [('if', IF_opti), ('lof', LOF_opti), ('ee', EE_pipeline), ('ocsvm_lin', OCSVM_lin_pipeline), ('ocsvm_rbf', OCSVM_rbf_pipeline)]
            
            # Voting Hardware 5
            vote_hard_5 = MyVotingOutlier(estimators=estimators_5, voting='hard', verbose=False)
            vote_hard_5.fit(X_train_norm, y_train)
            pred_hard_train_5 = vote_hard_5.predict(X_train_norm)
            pred_hard_test_5 = vote_hard_5.predict(X_test_norm)
            m_hard_5 = {'train': extraire_metriques(y_train, pred_hard_train_5), 'test': extraire_metriques(y_test, pred_hard_test_5)}
            
            # Voting Software 5
            vote_soft_5 = MyVotingOutlier(estimators=estimators_5, voting='soft', verbose=False)
            vote_soft_5.fit(X_train_norm, y_train)
            pred_soft_train_5 = vote_soft_5.predict(X_train_norm)
            pred_soft_test_5 = vote_soft_5.predict(X_test_norm)
            m_soft_5 = {'train': extraire_metriques(y_train, pred_soft_train_5), 'test': extraire_metriques(y_test, pred_soft_test_5)}
            
            # Voting S&F 5
            vote_sf_5 = MyVotingOutlier(estimators=estimators_5, voting='S&F', verbose=False)
            vote_sf_5.fit(X_train_norm, y_train)
            pred_sf_train_5 = vote_sf_5.predict(X_train_norm)
            pred_sf_test_5 = vote_sf_5.predict(X_test_norm)
            m_sf_5 = {'train': extraire_metriques(y_train, pred_sf_train_5), 'test': extraire_metriques(y_test, pred_sf_test_5)}
            
            res = {
                'classe': classe_cible,
                'config': nom_config,
                                'metriques': {
                    'Isolation Forest': m_if, 'Local Outlier Factor': m_lof, 'Elliptic Envelope': m_ee,
                    'OCSVM Linéaire': m_ocsvm_lin, 'OCSVM RBF': m_ocsvm_rbf,
                    'Modèle MIN': m_min, 'Modèle MAX': m_max, 'Modèle AVG': m_avg,
                    'Vote HARD 3': m_hard_3, 'Vote SOFT 3': m_soft_3, 'Vote S&F 3': m_sf_3,
                    'Vote HARD 5': m_hard_5, 'Vote SOFT 5': m_soft_5, 'Vote S&F 5': m_sf_5
                },
                'details_visu': {
                    'X_train_norm': X_train_norm,
                    'X_test_norm': X_test_norm,
                    'X_train_raw': X_train,
                    'X_test_raw': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'preds_train': {
                        'Isolation Forest': pred_if_train,
                        'Local Outlier Factor': pred_lof_train,
                        'Elliptic Envelope': pred_ee_train,
                        'OCSVM Linéaire': pred_ocsvm_lin_train,
                        'OCSVM RBF': pred_ocsvm_rbf_train,
                        'Vote HARD 3': pred_hard_train_3,
                        'Vote SOFT 3': pred_soft_train_3,
                        'Vote S&F 3': pred_sf_train_3,
                        'Vote HARD 5': pred_hard_train_5,
                        'Vote SOFT 5': pred_soft_train_5,
                        'Vote S&F 5': pred_sf_train_5
                    },
                    'preds_test': {
                        'Isolation Forest': pred_if_test,
                        'Local Outlier Factor': pred_lof_test,
                        'Elliptic Envelope': pred_ee_test,
                        'OCSVM Linéaire': pred_ocsvm_lin_test,
                        'OCSVM RBF': pred_ocsvm_rbf_test,
                        'Vote HARD 3': pred_hard_test_3,
                        'Vote SOFT 3': pred_soft_test_3,
                        'Vote S&F 3': pred_sf_test_3,
                        'Vote HARD 5': pred_hard_test_5,
                        'Vote SOFT 5': pred_soft_test_5,
                        'Vote S&F 5': pred_sf_test_5
                    }
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
    st.set_page_config(page_title="Evaluation d'Anomalies Borda & T-SNE", layout="wide")
    st.title("Comparatif des Modèles d'Anomalies - Classement Borda & Visualisation T-SNE")
    st.info("Pour afficher les résultats, il faut selectionner une classe et une configuration dans le menu à gauche, puis cliquer sur 'Lancer l'évaluation'.")
    
    if "donnees_eval" not in st.session_state:
        st.session_state.donnees_eval = None

    etat_cache = "Données sauvegardées localement" if os.path.exists(CACHE_FILE) else "Cache vide (Calculs à venir)"

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
            st.session_state.donnees_eval = None
            st.sidebar.success("Toutes les sauvegardes ont été purgées !")
            st.rerun()

    if bouton_lancer:
        classes_a_tester = classes_possibles if choix_classe == "Toutes les Classes" else [choix_classe]
        configs_a_tester = configs_possibles if choix_config == "Toutes les Configurations" else [choix_config]
        donnees = executer_evaluation(classes_a_tester, configs_a_tester)
        if not donnees:
            st.error("Génération des données échouée.")
            st.session_state.donnees_eval = None
        else:
            st.session_state.donnees_eval = donnees

    if st.session_state.donnees_eval:
        donnees_entrainees = st.session_state.donnees_eval

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
                "Précision Train (%)": round(np.mean(historique[candidat]['train']['precision']), 1),
                "Précision Test (%)": round(np.mean(historique[candidat]['test']['precision']), 1),
                "Rappel Train (%)": round(np.mean(historique[candidat]['train']['recall']), 1),
                "Rappel Test (%)": round(np.mean(historique[candidat]['test']['recall']), 1),
                "Accuracy Train (%)": round(np.mean(historique[candidat]['train']['accuracy']), 1),
                "Accuracy Test (%)": round(np.mean(historique[candidat]['test']['accuracy']), 1),
            })
            position += 1

        df_borda = pd.DataFrame(df_rows)
        df_borda.set_index("Position", inplace=True)
        
        # Création des onglets
        tab_borda, tab_tsne = st.tabs(["Tableau des résultats (Borda)", "My-TSNE"])
        
        with tab_borda:
            st.subheader(f"Basé sur {len(donnees_entrainees)} itération(s) évaluée(s)")
            
            st.dataframe(df_borda.style.highlight_max(subset=["Points Borda", "F1 Test (%)"], color='#c6e0b4')
                                       .highlight_min(subset=["Points Borda", "F1 Test (%)"], color='#f8cbcb')
                                       .format(precision=1), width='stretch')

            colA, colB = st.columns([1, 2])
            with colA:
                csv = df_borda.to_csv(sep=';', decimal=',').encode('utf-8')
                st.download_button(label="Télécharger ce tableau en format Excel/CSV", data=csv,
                                   file_name=f'Extraction_Metriques_{choix_classe}_{choix_config}.csv', mime='text/csv')

        with tab_tsne:
            if len(donnees_entrainees) == 1:
                st.subheader(f"Affichage pour la Classe {donnees_entrainees[0]['classe']} | Configuration {donnees_entrainees[0]['config']}")
                
                passage = donnees_entrainees[0]
                details = passage.get('details_visu', None)
            
                if details is None or 'X_train_raw' not in details:
                    st.warning(" Les données ont été mises à jour pour incorporer les images. Veuillez 'Vider les données' à gauche et relancer.")
                else:
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1:
                        modele_visu = st.selectbox("Choisir le Modèle :", list(details['preds_test'].keys()))
                    with c2:
                        choix_donnees = st.radio("Cible du T-SNE :", ["Test Uniquement", "Entraînement Uniquement", "Complet (Train + Test)"], index=0, horizontal=True)
                    
                    if '10pct' in passage['config']:
                        if choix_donnees == "Test Uniquement": def_perp = 5
                        elif choix_donnees == "Entraînement Uniquement": def_perp = 15
                        else: def_perp = 15
                    elif '25pct' in passage['config']:
                        if choix_donnees == "Test Uniquement": def_perp = 5
                        elif choix_donnees == "Entraînement Uniquement": def_perp = 20
                        else: def_perp = 25
                    else:
                        if choix_donnees == "Test Uniquement": def_perp = 15
                        elif choix_donnees == "Entraînement Uniquement": def_perp = 40
                        else: def_perp = 45
                        
                    with c3:
                        perplexity_val = st.slider("Perplexité (T-SNE) :", min_value=2, max_value=50, value=def_perp, step=1)
                    
    
                    # Extraction ciblée
                    if choix_donnees == "Test Uniquement":
                        X_norm = details['X_test_norm']
                        X_raw = details['X_test_raw']
                        y_true = details['y_test']
                        y_pred = details['preds_test'][modele_visu]
                    elif choix_donnees == "Entraînement Uniquement":
                        X_norm = details['X_train_norm']
                        X_raw = details['X_train_raw']
                        y_true = details['y_train']
                        y_pred = details['preds_train'][modele_visu]
                    else: # Complet
                        X_norm = np.vstack((details['X_train_norm'], details['X_test_norm']))
                        X_raw = np.vstack((details['X_train_raw'], details['X_test_raw']))
                        y_true = np.concatenate((details['y_train'], details['y_test']))
                        y_pred = np.concatenate((details['preds_train'][modele_visu], details['preds_test'][modele_visu]))
    
                    @st.cache_data
                    def compute_tsne(X_data, perp):
                        n_samples = X_data.shape[0]
                        # Protection mathématique 
                        safe_perp = min(perp, max(1, n_samples - 1))
                        return MyTSNE(n_components=2, perplexity=safe_perp, max_iter=1000).fit_transform(X_data)
                        
                    with st.spinner("Calcul des coordonnées MyTSNE en cours.."):
                        X_tsne = compute_tsne(X_norm, perplexity_val)
                    
                    fig, ax = plt.subplots(figsize=(11, 7))
                    
                    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                    from matplotlib.lines import Line2D
                        
                    for i in range(len(X_tsne)):
                            # Reconstruction du chiffre 8x8
                            img_data = X_raw[i].reshape(8, 8)
                            imagebox = OffsetImage(img_data, cmap=plt.cm.gray_r, zoom=1.4)
                            
                            is_anom_pred = (y_pred[i] == -1)
                            is_anom_true = (y_true[i] == -1)
                            
                            if is_anom_pred and is_anom_true:
                                color = '#d62728' # Vrai Positif (Bien Trouvé) = ROUGE
                                lw = 3
                            elif is_anom_pred and not is_anom_true:
                                color = '#ff7f0e' # Faux Positif (Fausse Alarme) = ORANGE
                                lw = 3
                            elif not is_anom_pred and is_anom_true:
                                color = '#9467bd' # Faux Négatif (Anomalie Ratée) = VIOLET
                                lw = 3
                            else:
                                color = '#1f77b4' # Vrai Négatif (Sain Normal) = BLEU
                                lw = 1
                                
                            ab = AnnotationBbox(imagebox, (X_tsne[i, 0], X_tsne[i, 1]), frameon=True,
                                                bboxprops=dict(edgecolor=color, linewidth=lw, facecolor='white', alpha=0.8))
                            ax.add_artist(ab)
                            
                        # Fixer les limites manuellement quand on utilise add_artist
                    ax.set_xlim(X_tsne[:, 0].min() - 5, X_tsne[:, 0].max() + 5)
                    ax.set_ylim(X_tsne[:, 1].min() - 5, X_tsne[:, 1].max() + 5)
                        
                    legend_elements = [
                            Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='#1f77b4', markersize=10, label='Vrai Négatif (Non anomalie bien identifiée)'),
                            Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='#d62728', markeredgewidth=2, markersize=10, label='Vrai Positif (Anomalie Bien Trouvée)'),
                            Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='#ff7f0e', markeredgewidth=2, markersize=10, label='Faux Positif (Anomalie Faussement Détectée)'),
                            Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='#9467bd', markeredgewidth=2, markersize=10, label='Faux Négatif (Anomalie Ratée)')
                        ]
                    ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left')
    
                    ax.set_title(f"Aperçu T-SNE - {modele_visu} ({choix_donnees} / P={perplexity_val})", fontsize=14, pad=15)
                    ax.grid(True, linestyle=":", alpha=0.4)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
            else:
                st.info("La zone de projection My-TSNE est masquée car vous avez sélectionné plusieurs classes ou configurations en même temps. \n\nPour réafficher et interagir avec l'espace 2D, veuillez cibler **une seule classe** et **une seule configuration** dans le menu de gauche puis relancez l'évaluation.")

if __name__ == "__main__":
    main()

