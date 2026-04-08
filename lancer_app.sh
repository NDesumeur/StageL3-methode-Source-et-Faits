#!/bin/bash
echo "========================================================"
echo "        Lancement de l'application Streamlit            "
echo "========================================================"
echo ""

# Vérifier si Streamlit est installé avant le lancement
if ! command -v streamlit &> /dev/null
then
    echo "[ERREUR] Streamlit n'est pas installé."
    echo "Installation des dépendances en cours avec pip..."
    pip3 install -r requirements.txt
fi

echo "Démarrage du serveur local Streamlit..."
streamlit run streamlit_app/app.py
