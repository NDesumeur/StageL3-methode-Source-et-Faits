# Streamlit 
Lancement de l'application via le terminal : `streamlit run app.py`

**RÈGLE FONDAMENTALE DE FONCTIONNEMENT**
À chaque interaction de l'utilisateur (clic, saisie, etc.), Streamlit **réexécute l'intégralité du script Python de haut en bas**.

## 1. Affichage de données
- `st.title("Titre")`, `st.header("Sous-titre")`, `st.markdown("**Gras**")` : Pour le balisage textuel.
- `st.write(variable)` : Fonction polyvalente qui s'adapte au type de la donnée fournie (texte, liste, dictionnaire, etc.) pour l'afficher graphiquement.
- `st.dataframe(df)` : Affiche un format tableau interactif (DataFrame Pandas complet).
- `st.pyplot(fig)` : Affiche un graphique Matplotlib pre-généré (idéal pour le t-SNE ou les matrices).

## 2. Widgets d'Interaction
Toute interaction avec ces éléments déclenche l'actualisation de la page.
- `st.button("Valider")` : Renvoie `True` uniquement au moment précis de l'activation.
- `st.selectbox("Modèle :", ["KNN", "SVM"])` : Menu déroulant pour une sélection de valeur unique.
- `st.slider("Valeur :", 0, 10)` : Curseur de sélection numérique.

## 3. Mise en page 
Fonctions permettant de structurer l'affichage plutôt que de tout empiler verticalement.
- `col1, col2 = st.columns(2)` : Divise l'espace de la page en colonnes (s'utilise avec le mot-clé `with`).
- `tab1, tab2 = st.tabs(["Graphiques", "Données"])` : Sépare les éléments dans des onglets de navigation.
- `st.sidebar.` : Modificateur pour placer n'importe quel contenu dans un menu latéral à gauche (exemple : `st.sidebar.button()`).

## 4. Gestion des performances et de l'état
Étant donné que la page se recharge à chaque clic, il est impératif d'utiliser des mécanismes de sauvegarde pour les processus coûteux.

**A) Le Cache (`@st.cache_data`)**
Un décorateur de fonction qui conserve le retour de celle-ci en mémoire. Le code interne ne sera exécuté qu'une seule fois tant que les arguments d'entrée ne changent pas.
```python
@st.cache_data
def charger_csv():
    # Utilisé pour éviter de recharger un gros dataset à chaque interaction
    return pd.read_csv("data.csv")
```

**B) L'état de session (`st.session_state`)**
Permet de définir des variables persistantes qui survivront au rechargement de la page. Indispensable pour contrôler la séquence ou l'affichage de menus conditionnels.
```python
# Initialisation de la variable si elle n'existe pas encore
if "etape_finie" not in st.session_state:
    st.session_state.etape_finie = False

# Mise à jour conditionnelle
if st.button("Valider"):
    st.session_state.etape_finie = True # La valeur sera conservée lors du rechargement
```