import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from classes.utils.ChargeurDonnees import ChargeurDonnees
from classes.utils.Normaliseur import Normaliseur
from classes.MyVotingOutlier import MyVotingOutlier
from classes.MyT_SNE import MyTSNE

def tester_tsne_complet_interactif():
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)

    print("1. Chargement de TOUT le dataset Digits (10 classes)...")
    X, y, _, _ = ChargeurDonnees.charger_scikit("Digits")
    X_np = X.to_numpy() if hasattr(X, 'to_numpy') else X
    
    # y est un dataframe ou serie, on convertit en int
    y_str = y.to_numpy().astype(str) if hasattr(y, 'to_numpy') else y.astype(str)
    y_true = y_str.astype(int)

    # Pour que ce soit rapide à observer dans l'application, 
    # on va prendre 1000 points aléatoires (au lieu de 1797)
    np.random.seed(42)
    indices = np.random.choice(len(X_np), min(1000, len(X_np)), replace=False)
    X_sample = X_np[indices]
    y_sample = y_true[indices]

    print("2. Normalisation / PCA...")
    norm = Normaliseur(methode='auto')
    X_norm = norm.fit_transform(X_sample)
    
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_norm)

    print("3. Entraînement des modèles d'outliers non supervisés sur l'ensemble global...")
    
    # On force TOUS les modèles à trouver exactement 5% d'anomalies (50 points sur 1000)
    # Cela permet de comparer "QUELS" points ils choisissent, à égalité.
    taux_contamination = 0.05
    
    IF = IsolationForest(random_state=42, contamination=taux_contamination)
    IF.fit(X_norm)
    pred_IF = IF.predict(X_norm)
    
    LOF = LocalOutlierFactor(novelty=False, contamination=taux_contamination)
    pred_LOF = LOF.fit_predict(X_norm)
    
    EE_pipeline = Pipeline([
        ('pca', PCA(n_components=0.95, random_state=42)),
        ('ee', EllipticEnvelope(random_state=42, contamination=taux_contamination))
    ])
    EE_pipeline.fit(X_norm)
    pred_EE = EE_pipeline.predict(X_norm)

    print("4. Projection T-SNE de TOUTES les classes...")
    tsne = MyTSNE(n_components=2, max_iter=500, perplexity=30.0, random_state=42)
    tsne.fit_transform(X_norm)

    dict_predictions = {
        "Isolation Forest": pred_IF,
        "Local Outlier Factor": pred_LOF,
        "Elliptic Envelope": pred_EE
    }

    print("5. Ouverture de l'interface T-SNE ! (Cliquez sur les boutons à gauche)")
    tsne.afficher_interactif_anomalies(dict_predictions=dict_predictions, y_true=y_sample)

if __name__ == "__main__":
    tester_tsne_complet_interactif()
