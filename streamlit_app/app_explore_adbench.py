import streamlit as st
import numpy as np
import pandas as pd
import os

def main():
    # st.set_page_config(page_title="Explorateur ADBench", layout="wide")
    st.title(" Explorateur de Datasets ADBench")
    
    dossier_data = os.path.join(os.getcwd(), "data", "adbench")
    
    if not os.path.exists(dossier_data):
        st.error(f"Le dossier {dossier_data} n'existe pas. Téléchargez d'abord les données.")
        return
        
    fichiers_npz = [f for f in os.listdir(dossier_data) if f.endswith('.npz')]
    
    if not fichiers_npz:
        st.warning("Aucun fichier .npz trouvé dans le dossier.")
        return
        
    # Barre latérale pour choisir le dataset
    dataset_choisi = st.selectbox("Sélectionnez un dataset à analyser :", sorted(fichiers_npz))
    
    if dataset_choisi:
        chemin_fichier = os.path.join(dossier_data, dataset_choisi)
        
        # Chargement du fichier .npz
        data = np.load(chemin_fichier, allow_pickle=True)
        X = data['X']
        y = np.array(data['y'], dtype=int)
        
        # Calculs de base
        nb_lignes, nb_colonnes = X.shape
        nb_anomalies = np.sum(y == 1)
        nb_normaux = np.sum(y == 0)
        taux_contamination = (nb_anomalies / nb_lignes) * 100
        
        # Création des métriques en haut de page
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Échantillons (Lignes)", nb_lignes)
        col2.metric("Variables (Features)", nb_colonnes)
        col3.metric("Anomalies (Label=1)", nb_anomalies, f"{taux_contamination:.2f}% du total", delta_color="inverse")
        col4.metric("Points Normaux (Label=0)", nb_normaux)
        
        st.markdown("---")
        
        # Conversion en DataFrame Pandas pour un bel affichage
        df = pd.DataFrame(X)
        df.columns = [f"Feature_{i}" for i in range(nb_colonnes)]
        
        # On met le label à la toute fin
        df['LABEL (Anomalie ?)'] = y
        
        st.subheader("Aperçu des 50 premières lignes du tableau")
        # On stylise pour mettre en rouge les lignes qui sont des anomalies
        def coloriser_anomalies(row):
            if row['LABEL (Anomalie ?)'] == 1:
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)
            
        st.dataframe(df.head(50).style.apply(coloriser_anomalies, axis=1), use_container_width=True)

if __name__ == "__main__":
    main()
