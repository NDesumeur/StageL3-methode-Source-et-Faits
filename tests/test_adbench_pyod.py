import numpy as np
import pandas as pd
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Importation de quelques modèles PyOD demandés
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

# Utilitaire pour générer des données tabulaires (Simulation d'un dataset ADBench)
from pyod.utils.data import generate_data

def evaluer_modele(nom_modele, modele, X_train, y_train, X_test, y_test):
    print(f"\n--- Entraînement de {nom_modele} ---")
    
    # Entraînement
    modele.fit(X_train)
    
    # Prédictions (0 = inlier, 1 = outlier pour PyOD)
    # PyOD utilise 0 et 1, contrairement à scikit-learn (1 et -1)
    y_test_pred = modele.predict(X_test)
    
    # Scores d'anomalies bruts (plus c'est élevé, plus c'est anormal)
    y_test_scores = modele.decision_function(X_test)
    
    # Calcul des métriques
    roc_auc = roc_auc_score(y_test, y_test_scores)
    f1 = f1_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Précision: {prec:.4f}")
    print(f"Rappel   : {rec:.4f}")
    
    return roc_auc, f1

def charger_dataset_adbench(nom_fichier="2_annthyroid.npz"):
    """Charge un dataset classique depuis le dossier local data/adbench."""
    dossier = "data/adbench"
    chemin_fichier = os.path.join(dossier, nom_fichier)
    
    if not os.path.exists(chemin_fichier):
        print(f" Erreur : Le fichier {chemin_fichier} n'existe pas. Veuillez lancer download_adbench.py d'abord.")
        return None, None
            
    print(f" Chargement des données depuis {chemin_fichier}...")
    data = np.load(chemin_fichier, allow_pickle=True)
    
    # Dans ADBench, les .npz contiennent généralement 'X' et 'y'
    X = data['X']
    y = data['y']
    
    # y peut parfois contenir des strings ou floats, on s'assure d'avoir des entiers
    y = np.array(y, dtype=int) 
    
    return X, y

def main():
    print("Démarrage de l'évaluation ADBench via PyOD...")
    # Lister les fichiers .npz dans data/adbench/
    dossier = "data/adbench"
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.npz')]
    
    if not fichiers:
        print(" Aucun fichier .npz trouvé dans data/adbench/.")
        return
        
    # Choisir le premier dataset ou en sélectionner un au hasard/spécifique
    dataset_cible = fichiers[0] # Ou "2_annthyroid.npz" si présent
    print(f"Dataset sélectionné : {dataset_cible}")
    
    X, y = charger_dataset_adbench(dataset_cible)
    
    if X is None or y is None:
        print("Arrêt : Données introuvables.")
        return
        
    contamination = max(0.01, sum(y) / len(y))  # Ratio d'anomalies réel du dataset

    
    # Séparation Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Taille Train : {X_train.shape} | Anomalies : {sum(y_train)}")
    print(f"Taille Test  : {X_test.shape} | Anomalies : {sum(y_test)}")
    
    # Standardisation (Cruciale pour KNN, OCSVM, PCA, etc.)
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    # Dictionnaire des modèles à tester
    modeles = {
        "Isolation Forest (iForest)": IForest(contamination=contamination, random_state=42),
        "Local Outlier Factor (LOF)": LOF(contamination=contamination),
        "K-Nearest Neighbors (KNN)": KNN(contamination=contamination),
        "One-Class SVM (OCSVM)": OCSVM(contamination=contamination),
        "Principal Component Analysis (PCA)": PCA(contamination=contamination, random_state=42)
    }
    
    resultats = {}
    
    for nom, mod in modeles.items():
        roc, f1 = evaluer_modele(nom, mod, X_train_norm, y_train, X_test_norm, y_test)
        resultats[nom] = {'ROC-AUC': roc, 'F1-Score': f1}
        
    print("\n=== RÉSUMÉ DES PERFORMANCES ===")
    df_res = pd.DataFrame(resultats).T
    df_res = df_res.sort_values(by="ROC-AUC", ascending=False)
    print(df_res)

if __name__ == "__main__":
    main()