import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate


class Evaluateur:
    """
    Classe d'évaluation des modèles permettant soit une évaluation classique sur Test Set, 
    soit une validation croisée automatique (cross-validation).
    """
    def __init__(self, model, X, y, cross_val=False, cv_folds=5):
        """
        - Si cross_val=False : X et y doivent être X_test et y_test.
        - Si cross_val=True  : X et y doivent être le jeu de données entier (ou X_train).
        """
        self.model = model
        self.X = X
        self.y = y
        self.cross_val = cross_val
        self.cv_folds = cv_folds
        
        self.y_pred = None
        self.metrics = None

    def evaluate(self):
        """
        Calcule les métriques en déduisant automatiquement la méthode 
        (Classique ou Cross-Validation) grâce au paramètre du constructeur.
        """
        if self.cross_val:
            return self._evaluate_cv()
        else:
            return self._evaluate_standard()

    def _evaluate_standard(self):
        self.y_pred = self.model.predict(self.X)
        self.metrics = {
            'accuracy': accuracy_score(self.y, self.y_pred),
            'precision': precision_score(self.y, self.y_pred, average='macro', zero_division=0),
            'recall': recall_score(self.y, self.y_pred, average='macro', zero_division=0),
            'f1_score': f1_score(self.y, self.y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(self.y, self.y_pred),
            'classification_report': classification_report(self.y, self.y_pred)
        }
        return self.metrics

    def _evaluate_cv(self):
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        self.metrics = cross_validate(self.model, self.X, self.y, cv=self.cv_folds, scoring=scoring, n_jobs=-1)
        return self.metrics
    
    def print_metrics(self, metrics=None, nom_modele=None):
        """
        Affiche les métriques. S'adapte automatiquement au type d'évaluation (CV ou Classique).
        """
        if metrics is not None:
            self.metrics = metrics
            
        if self.metrics is None:
            print("Veuillez d'abord appeler la méthode evaluate() pour calculer les métriques.")
            return
            
        nom = nom_modele if nom_modele else type(self.model).__name__

        if self.cross_val:
            print(f"\n=== ÉVALUATION CROISÉE (CV={self.cv_folds}) : {nom} ===")
            print(f"Accuracy moyenne  : {np.mean(self.metrics['test_accuracy']) * 100:.2f}% (+/- {np.std(self.metrics['test_accuracy']) * 100:.2f}%)")
            print(f"Precision moyenne : {np.mean(self.metrics['test_precision_macro']) * 100:.2f}%")
            print(f"Recall moyen      : {np.mean(self.metrics['test_recall_macro']) * 100:.2f}%")
            print(f"F1-Score moyen    : {np.mean(self.metrics['test_f1_macro']) * 100:.2f}%")
        else:
            print(f"\n=== ÉVALUATION CLASSIQUE (Test Set) : {nom} ===")
            print(f"Accuracy  : {self.metrics['accuracy'] * 100:.2f}%")
            print(f"Precision : {self.metrics['precision'] * 100:.2f}%")
            print(f"Recall    : {self.metrics['recall'] * 100:.2f}%")
            print(f"F1-Score  : {self.metrics['f1_score'] * 100:.2f}%")
            print("\nClassification Report :")
            print(self.metrics['classification_report'])
            
    def plot_confusion_matrix(self, save_path=None):
        """
        Affiche la matrice de confusion. Uniquement disponible en mode évaluation standard.
        - save_path : Chemin où sauvegarder l'image (par ex: 'matrice.png'). Si None, affiche juste l'image.
        """
        if self.cross_val:
            print("La matrice de confusion n'est disponible qu'en évaluation classique, pas en cross-validation.")
            return
            
        if self.metrics is None or 'confusion_matrix' not in self.metrics:
            print("Veuillez d'abord appeler la méthode evaluate() en évaluation standard.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=self.metrics['confusion_matrix'])
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        
        plt.title(f"Matrice de confusion - {type(self.model).__name__}")
        plt.ylabel("Vraies étiquettes")
        plt.xlabel("Prédictions")
        
        if save_path:
            plt.savefig(save_path)
            print(f"Graphique sauvegardé sous {save_path}")
        
        plt.show()