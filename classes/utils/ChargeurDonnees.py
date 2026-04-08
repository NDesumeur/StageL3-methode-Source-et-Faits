import pandas as pd
from sklearn import datasets
import numpy as np

class ChargeurDonnees:
    """
    Classe utilitaire conçue pour fournir facilement des données à l'application Streamlit.
    Gère les datasets natifs de Scikit-Learn fournis par l'utilisateur.
    - lister_datasets_scikit() : Retourne la liste des datasets disponibles pour la sélection.
    - charger_scikit(nom_choisi) : Charge le dataset correspondant au nom choisi et retourne X, y, les noms des colonnes et les noms des classes.

    """
    
    @staticmethod 
    def lister_datasets_scikit():
        """
        Retourne la liste des datasets pertinents pour de la classification.
        """
        return ["Iris (Classification de fleurs)", 
                "Wine (Classification de vins)", 
                "Breast Cancer (Détection de cancer)", 
                "Digits (Petits chiffres 8x8)",
                "MNIST (Grands chiffres 28x28)",
                "Fashion MNIST (Vêtements)",
                "Olivetti Faces (Visages)",
                "Lunes",
                "Cercles",
                "Nuages de points"]

    @staticmethod
    def charger_scikit(nom_choisi):
        """
        Charge un dataset scikit-learn en fonction du nom choisi dans l'interface.
        Renvoie X, y, les noms des colonnes et les noms des cibles.
        """
        if "Iris" in nom_choisi:
            data = datasets.load_iris()
            X, y = data.data, data.target
        elif "Wine" in nom_choisi:
            data = datasets.load_wine()
            X, y = data.data, data.target
        elif "Breast" in nom_choisi:
            data = datasets.load_breast_cancer()
            X, y = data.data, data.target
        elif "Digits" in nom_choisi:
            data = datasets.load_digits()
            X, y = data.data, data.target
        elif "Fashion MNIST" in nom_choisi:
            data = datasets.fetch_openml('Fashion-MNIST', version=1, parser='auto')
            X, y = data.data.values if hasattr(data.data, 'values') else data.data, data.target.astype(int).values if hasattr(data.target, 'values') else data.target.astype(int)
        elif "MNIST" in nom_choisi:
            data = datasets.fetch_openml('mnist_784', version=1, parser='auto')
            X, y = data.data.values if hasattr(data.data, 'values') else data.data, data.target.astype(int).values if hasattr(data.target, 'values') else data.target.astype(int)
        elif "Olivetti" in nom_choisi:
            data = datasets.fetch_olivetti_faces()
            X, y = data.data, data.target
        elif "Lunes" in nom_choisi:
            X, y = datasets.make_moons(n_samples=1000, noise=0.2, random_state=42)
            data = type('DummyResult', (object,), {'feature_names': ['Axe X', 'Axe Y'], 'target_names': ['Bleu', 'Rouge']})()
        elif "Cercles" in nom_choisi:
            X, y = datasets.make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
            data = type('DummyResult', (object,), {'feature_names': ['Axe X', 'Axe Y'], 'target_names': ['Intérieur', 'Extérieur']})()
        elif "Nuages" in nom_choisi:
            X, y = datasets.make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=3, random_state=42)
            data = type('DummyResult', (object,), {'feature_names': [f"Feature_{i}" for i in range(10)], 'target_names': ['Classe 0', 'Classe 1', 'Classe 2']})()
        else:
            raise ValueError("Dataset inconnu")
        
        # Sécurisation des noms de colonnes et de classes
        # Si le dataset a des noms de features, on les utilise, sinon on génère des noms génériques
        noms_features = data.feature_names if hasattr(data, 'feature_names') else [f"Pixel_{i}" if "MNIST" in nom_choisi else f"Col_{i}" for i in range(X.shape[1])]
        

        # Si MNIST ou Olivetti, on génère des noms de classes basés sur les labels numériques, sinon on essaie d'utiliser target_names s'ils sont pertinents, sinon on génère des noms génériques
        # On vérifie que target_names existe, qu'il a plus d'une classe, et que ce n'est pas juste une liste de 'class' générique (certains datasets utilisent 'class' comme nom de cible sans donner les vrais noms)
        # sinon on se rabat sur les classes uniques présentes dans y
        if "MNIST" in nom_choisi or "Olivetti" in nom_choisi:
            noms_classes = [str(c) for c in np.unique(y)]
        elif hasattr(data, 'target_names') and data.target_names is not None and len(data.target_names) > 1 and data.target_names[0] != 'class':
            noms_classes = [str(c) for c in data.target_names]
        else:
            noms_classes = [str(c) for c in np.unique(y)]
        
        return X, y, noms_features, noms_classes

  