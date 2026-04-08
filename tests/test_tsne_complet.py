import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.MyT_SNE import MyTSNE

def tester_tsne_complet():
    print(" Génération d'un petit jeu de données (3 groupes de points)")

    X, y = make_blobs(n_samples=150, n_features=10, centers=3, random_state=42)

    print(f" Taille des données : {X.shape} (150 points, 10 dimensions)")
    print("\n-----------------------------------------")

    print("  Lancement de NOTRE MyTSNE... (1000 itérations)")

    tsne_perso = MyTSNE(n_components=2, learning_rate=200.0, max_iter=1000, random_state=42, init='pca')
    embed_perso = tsne_perso.fit_transform(X)

    print("  Lancement de t-SNE Scikit-Learn officiel (1000 itérations)")
    tsne_officiel = TSNE(n_components=2, learning_rate=200.0, max_iter=1000, random_state=42, init='pca')
    embed_officiel = tsne_officiel.fit_transform(X)

    print("\n Terminé ! Calcul des scores d'évaluation (Trustworthiness)")

    score_perso = tsne_perso.score_voisinage(X)

    from sklearn.manifold import trustworthiness
    score_officiel = trustworthiness(X, embed_officiel)

    print(f"-> Score MyTSNE   : {score_perso*100:.2f}% de fidélité locale.")
    print(f"-> Score Officiel : {score_officiel*100:.2f}% de fidélité locale.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(embed_perso[:, 0], embed_perso[:, 1], c=y, cmap='tab10', alpha=0.7)
    ax1.set_title(f"Notre t-SNE (MyTSNE)\nScore de confiance: {score_perso:.3f}", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.scatter(embed_officiel[:, 0], embed_officiel[:, 1], c=y, cmap='tab10', alpha=0.7)
    ax2.set_title(f"t-SNE Officiel (Scikit-Learn)\nScore de confiance: {score_officiel:.3f}", fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Comparaison de notre algorithme avec Scikit-Learn", fontsize=14)
    plt.show()

if __name__ == "__main__":
    tester_tsne_complet()
