@echo off
echo ========================================================
echo         Lancement de l'application Streamlit        
echo ========================================================
echo.

:: Vérifier si Streamlit est installé avant le lancement
python -m streamlit --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Streamlit n'est pas installe.
    echo Installation des dependances en cours avec pip...
    pip install -r requirements.txt
)

echo Demarrage du serveur local Streamlit...
streamlit run streamlit_app\app.py

pause
