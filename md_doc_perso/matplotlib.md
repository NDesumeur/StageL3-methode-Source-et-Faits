# Matplotlib 

Le standard pour dessiner des graphiques en Python. Toujours importé comme ça : `import matplotlib.pyplot as plt`.

## La base absolue
```python
plt.plot([1, 2], [3, 4]) # Tracer une ligne X, Y
plt.show() # Affiche la fenêtre (attention ça bloque le code tant qu'on ferme pas)
```

## La méthode (Figure et Axes)
on prépare la toile (`fig`) et le cadre de dessin (`ax`).
```python
fig, ax = plt.subplots(figsize=(8, 6)) # Toile de 8x6
ax.set_title("Titre")
ax.set_xlabel("Axe X")
```

## Tracer le t-SNE (Nuage de points)
Le t-SNE nous donne des X et Y, et on veut colorier chaque point selon sa classe. On utilise `scatter` :
```python
# c = la liste des réponses mathématiques (ex: 0, 1, 2) pour colorier
# cmap='tab10' = palette magique avec 10 couleurs super distinctes
graph = ax.scatter(x=points_x, y=points_y, c=labels, cmap='tab10', s=5, alpha=0.6)

fig.colorbar(graph, label="Classes") # La petite barre de couleur sur le côté
```

## Autres commandes à retenir
- `plt.imshow(matrice)` : Affiche un tableau Numpy sous forme d'image (ex: MNIST).
- `plt.savefig("image.png")` : Sauvegarde au lieu d'afficher.
- `plt.gcf()` : "Get Current Figure". Très utile pour attraper l'image en cours et l'envoyer à Streamlit sans faire bugger le terminal avec `plt.show()` 