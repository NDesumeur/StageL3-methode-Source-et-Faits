import numpy as np
from .ChargeurDonnees import ChargeurDonnees

class ChargeurDonneesPourOutlier:
    """
    Classe utilitaire pour generer des jeux de donnees d'Outliers (Anomalies) 
    a partir des datasets standards de scikit-learn.
    1  -> Donnee normale
    -1 -> Anomalie
    """
    
    @staticmethod
    def charger(nom_dataset, classe_normale, pourcentage_normaux=100, nb_anomalies_par_classe=2, random_state=42):
        # 1. Charger dataset existant grace a ChargeurDonnees
        X, y, noms_features, _ = ChargeurDonnees.charger_scikit(nom_dataset)
        
        X_np = X.to_numpy() if hasattr(X, 'to_numpy') else X
        y_np = y.to_numpy() if hasattr(y, 'to_numpy') else y
        
        y_encoded = y_np.astype(str)
        classe_normale_str = str(classe_normale)
        
        # 2. Filtrer les normaux
        indices_normaux = np.where(y_encoded == classe_normale_str)[0]
        nb_normaux_a_garder = int(len(indices_normaux) * (pourcentage_normaux / 100.0))
        
        np.random.seed(random_state)
        indices_normaux_gardes = np.random.permutation(indices_normaux)[:nb_normaux_a_garder]
        
        X_normaux = X_np[indices_normaux_gardes]
        y_normaux = np.ones(len(X_normaux), dtype=int)
        
        # 3. Filtrer les anomalies
        classes_anormales = [c for c in np.unique(y_encoded) if c != classe_normale_str]
        
        X_anomalies_list = []
        for c in classes_anormales:
            indices_anormaux_classe = np.where(y_encoded == c)[0]
            np.random.shuffle(indices_anormaux_classe)
            indices_selectionnes = indices_anormaux_classe[:nb_anomalies_par_classe]
            
            if len(indices_selectionnes) > 0:
                X_anomalies_list.append(X_np[indices_selectionnes])
                
        if len(X_anomalies_list) > 0:
            X_anomalies = np.vstack(X_anomalies_list)
            y_anomalies = -np.ones(len(X_anomalies), dtype=int)
        else:
            X_anomalies = np.empty((0, X_np.shape[1]))
            y_anomalies = np.empty((0,), dtype=int)
            
        # 4. Concat et melange
        X_custom = np.vstack((X_normaux, X_anomalies))
        y_custom = np.concatenate((y_normaux, y_anomalies))
        
        indices_melange_final = np.random.permutation(len(X_custom))
        X_custom = X_custom[indices_melange_final]
        y_custom = y_custom[indices_melange_final]
        
        return X_custom, y_custom, noms_features

    @staticmethod
    def charger_grille_anomalies(nom_dataset, random_state=42):
        """
        Pour chaque classe du dataset original, génère les 4 configurations demandées :
        - 100% de données normales, 2 anomalies par classe
        - 50% de données normales, 2 anomalies par classe
        - 25% de données normales, 1 anomalie par classe
        - 10% de données normales, 1 anomalie par classe
        
        Renvoie un dictionnaire structuré : 
        {
            'classe_0': {
                '100_2': (X, y),
                '50_2': (X, y),
                ...
            },
            ...
        }
        """
        X, y, noms_features, _ = ChargeurDonnees.charger_scikit(nom_dataset)
        
        y_np = y.to_numpy() if hasattr(y, 'to_numpy') else y
        y_encoded = y_np.astype(str)
        classes_uniques = np.unique(y_encoded)
        
        configurations = [
            (100, 2),
            (50, 2),
            (25, 1),
            (10, 1)
        ]
        
        resultats_complets = {}
        
        for classe in sorted(classes_uniques):
            resultats_complets[classe] = {}
            for config in configurations:
                pourcentage_normaux, nb_anomalies = config
                
                Cle_nom = f"{pourcentage_normaux}pct_{nb_anomalies}ano"
                
                # On réutilise la méthode charger classique
                X_cust, y_cust, _ = ChargeurDonneesPourOutlier.charger(
                    nom_dataset=nom_dataset,
                    classe_normale=classe,
                    pourcentage_normaux=pourcentage_normaux,
                    nb_anomalies_par_classe=nb_anomalies,
                    random_state=random_state
                )
                
                resultats_complets[classe][Cle_nom] = (X_cust, y_cust)
                
        return resultats_complets, noms_features
