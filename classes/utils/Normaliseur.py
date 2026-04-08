from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
class Normaliseur:
    """
    Classe de normalisation des données. Permet de choisir entre plusieurs méthodes.
    - minmax : Normalise entre 0 et 1.
    - standard : Centre sur 0 avec écart-type de 1.
    - robust : Comme Standard, mais résistant aux valeurs extrêmes.
    - maxabs : Divise par la valeur absolue max, sans décaler autour de 0.
    - auto : Devine automatiquement la meilleure méthode de normalisation selon les données (lors du fit).
    - quand utiliser ?
        - minmax : données entre 2 valeurs connues, pas de valeurs extrêmes.
        - standard : données avec distribution normale, ou quand on veut centrer sur 0.
        - robust : données avec valeurs extrêmes, ou quand on veut une normalisation plus robuste.
        - maxabs : données déjà centrées sur 0, mais avec différentes échelles.
    """
    def __init__(self, methode='minmax'):
        self.methode = methode
        self.scaler = None
        self._initialiser_scaler()
        
    def _initialiser_scaler(self):
        if self.methode == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.methode == 'standard':
            self.scaler = StandardScaler()
        elif self.methode == 'robust':
            self.scaler = RobustScaler()
        elif self.methode == 'maxabs':
            self.scaler = MaxAbsScaler()
        elif self.methode == 'auto':
            self.scaler = None 
        else:
            raise ValueError("methode doit être 'minmax', 'standard', 'robust', 'maxabs' ou 'auto'")

    @staticmethod
    def deviner_meilleure_methode(X):
        """
        Analyse  la distribution des données pour en déduire 
        la meilleure méthode de normalisation par défaut.

        """
        # On convertit en numpy array au cas où c'est un DataFrame ou autre format
        X_array = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)
        
        # 1. Vérifier si on a un format type 'image' (valeurs entre 0 et 255 environ)
        if (np.min(X_array) >= 0.0) and (np.max(X_array) <= 255.0) and (np.max(X_array) > 1.0):
            return "minmax"
            
        # 2. Vérification des anomalies / écarts extrêmes dans les données
        # percentile 1 donne pour une liste triée la valeur en dessous de laquelle se trouvent 1% des données
        # percentile 99 donne la valeur en dessous de laquelle se trouvent 99% des données
        p99 = np.percentile(X_array, 99)
        p1 = np.percentile(X_array, 1)
        mediane = np.median(X_array)
        
        #  Si la distance entre le maximum et le 99e percentile 
        # est plus grande que toute l'étendue globale des données "normales" (entre 1% et 99%)
        # permet de vérifier s'il y a des valeurs très extrêmes qui pourraient fausser une normalisation standard
        if (np.max(X_array) - p99) > (p99 - p1) or (p1 - np.min(X_array)) > (p99 - p1):
            return "robust"
            
        # 3. Cas général
        return "standard"

    def fit(self, X):
        if self.methode == 'auto':
            self.methode = self.deviner_meilleure_methode(X)
            self._initialiser_scaler()
            
        self.scaler.fit(X)
        return self

    def transform(self, X):
        if self.scaler is None:
            raise ValueError("Le scaler n'a pas été initialisé. Effectuez un fit() d'abord.")
        return self.scaler.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)