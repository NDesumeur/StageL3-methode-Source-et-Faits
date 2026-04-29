import streamlit as st
import os

# Configuration globale pour toutes les pages (doit être la première commande Streamlit)
st.set_page_config(
    page_title="Hub - Stage L3",
    layout="wide"
)

# Obtenir le chemin de base du dossier 'streamlit_app'
base_dir = os.path.dirname(__file__)

# --- Définition des différentes pages du Hub ---
page_classification = st.Page(
    os.path.join(base_dir, "app.py"),
    title="1. Classification & Vote")

page_borda_tsne = st.Page(
    os.path.join(base_dir, "app_anomalie.py"),
    title="2. Détection d'Anomalies Images (T-SNE et Borda)"
)

page_adbench = st.Page(
    os.path.join(base_dir, "app_explore_adbench.py"),
    title="3. Explorateur ADBench"
)

# --- Configuration du menu de navigation ---
pg = st.navigation(
    {
        "Scikit-Learn": [page_classification, page_borda_tsne],
        "PyOD": [page_adbench],
    }
)

# Exécution de la navigation
pg.run()
