import sys
import os
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.MyT_SNE import MyTSNE

def main():
    print("--- Test comparatif : MyTSNE vs Sklearn TSNE ---")

    print("Chargement des données Digits (scikit-learn)")
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    X = X[:500]
    y = y[:500]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Format des données : {X_scaled.shape}")

    print("Début de l'entraînement de MyTSNE")
    my_tsne = MyTSNE(n_components=2, perplexity=30.0, learning_rate=200.0, max_iter=500, random_state=42, init='pca')
    X_2d_custom = my_tsne.fit_transform(X_scaled)

    print("Début de l'entraînement du TSNE de scikit-learn")

    sklearn_tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, max_iter=500, random_state=42, init='pca')
    X_2d_sklearn = sklearn_tsne.fit_transform(X_scaled)

    print("\nCalcul de la Trustworthiness")
    from sklearn.manifold import trustworthiness

    score_custom = my_tsne.score_voisinage(X_scaled, n_neighbors=5)
    silhouette_custom = my_tsne.calculer_silhouette_score(y)

    score_sklearn = trustworthiness(X_scaled, X_2d_sklearn, n_neighbors=5)

    print(f" -> Score Trustworthiness MyTSNE       : {score_custom*100:.2f}%")
    print(f" -> Score Silhouette MyTSNE            : {silhouette_custom:.3f}")
    print(f" -> Score Trustworthiness Scikit-Learn : {score_sklearn*100:.2f}%\n")

    print("Test de la méthode afficher()...")

    my_tsne.afficher(X=X_scaled, y=y, afficher_score=True, save_path="rendu_my_tsne.png")

    print("\nTest de la méthode afficher_comparatif_init()...")

    my_tsne.afficher_comparatif_init(X_scaled, y, save_path="comparatif_init.png")

if __name__ == "__main__":
    main()
