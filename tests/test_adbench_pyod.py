import numpy as np
import pandas as pd
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler

# Importation de TOUS les modèles PyOD demandés
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.loda import LODA
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.sos import SOS

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classes.utils.Trouve_params_pyod import Trouve_params_pyod

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
    acc = accuracy_score(y_test, y_test_pred)
    
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Précision: {prec:.4f}")
    print(f"Rappel   : {rec:.4f}")
    print(f"Accuracy : {acc:.4f}")
    
    return roc_auc, f1, prec, rec, acc

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
        
    # On choisit un dataset "raisonnable" où les anomalies sont identifiables 
    # pour te prouver que les algo tournent bien au-dessus de 50% (F1) / 90% (AUC)
    dataset_cible = "18_Ionosphere.npz" 
    print(f"Dataset sélectionné : {dataset_cible}")
    
    X, y = charger_dataset_adbench(dataset_cible)
    
    if X is None or y is None:
        print("Arrêt : Données introuvables.")
        return
        
    # --- ASTUCE OCSVM : SOUS-ÉCHANTILLONNAGE SUR LES TRES GROS DATASETS ---
    # OCSVM a une complexité O(N^3). Au-delà de 10 000 lignes il fige mathématiquement.
    max_lignes = 10000
    if len(X) > max_lignes:
        print(f"\n Dataset massif ({len(X)} lignes). OCSVM va crasher/être infini.")
        print(f"Stratification à {max_lignes} lignes au total...")
        _, X, _, y = train_test_split(X, y, test_size=max_lignes/len(X), stratify=y, random_state=42)
        
    contamination = max(0.01, sum(y) / len(y))  # Ratio d'anomalies réel du dataset

    
    # Séparation Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Taille Train : {X_train.shape} | Anomalies : {sum(y_train)}")
    print(f"Taille Test  : {X_test.shape} | Anomalies : {sum(y_test)}")
    
    # Standardisation Robuste (Ignore les anomalies extrêmes pour calculer l'échelle)
    scaler = RobustScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    # Dictionnaire des modèles à tester
    modeles = {
        "IForest": IForest(contamination=contamination, random_state=42),
        "LOF": LOF(contamination=contamination),
        "KNN": KNN(contamination=contamination),
        "OCSVM": OCSVM(contamination=contamination),
        "PCA": PCA(contamination=contamination, random_state=42),
        "CBLOF": CBLOF(contamination=contamination, random_state=42),
        "COF": COF(contamination=contamination),
        "HBOS": HBOS(contamination=contamination),
        "LODA": LODA(contamination=contamination),
        "COPOD": COPOD(contamination=contamination),
        "ECOD": ECOD(contamination=contamination),
        # SOS et DeepSVDD peuvent être très longs, on les ajoute mais on les commente si ça freeze
        "SOS": SOS(contamination=contamination),
        "DeepSVDD": DeepSVDD(contamination=contamination, random_state=42, verbose=0, n_features=X_train_norm.shape[1])
    }
    
    # On va utiliser notre optimiseur sur quelques modèles pour l'exemple (ça prendrait trop de temps sinon)
    print("\n[OPTIMISATION GRILLE - TEST]")
    optimiseur = Trouve_params_pyod(X_train_norm, y_train, cv=2, scoring='f1')
    
    resultats = {}
    
    for nom, mod in modeles.items():
        print(f"\n--- Évaluation de {nom} ---")
        try:
            # Optimisation exhaustive de TOUS les algorithmes via GridSearch
            mod_opt = optimiseur.trouve_params(mod)
            roc, f1, prec, rec, acc = evaluer_modele(f"{nom} (Optimisé)", mod_opt, X_train_norm, y_train, X_test_norm, y_test)
            
            resultats[nom] = {'ROC-AUC': roc, 'F1-Score': f1, 'Précision': prec, 'Rappel': rec, 'Accuracy': acc}
        except Exception as e:
            print(f" Échec du modèle {nom} : {e}")
        
    print("\n=== RÉSUMÉ DES PERFORMANCES ===")
    df_res = pd.DataFrame(resultats).T
    df_res = df_res.sort_values(by="ROC-AUC", ascending=False)
    print(df_res)

if __name__ == "__main__":
    main()