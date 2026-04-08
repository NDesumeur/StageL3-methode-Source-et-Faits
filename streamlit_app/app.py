import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classes.utils.ChargeurDonnees import ChargeurDonnees
from classes.utils.Normaliseur import Normaliseur
from classes.utils.Trouve_params import Trouve_params
from classes.utils.Evaluateur import Evaluateur
from classes.MyVotingClassifier import MyVotingClassifier
import numpy as np
from classes.MyT_SNE import MyTSNE
from sklearn.datasets import * 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC


@st.cache_data
def charger_donnees_cachees(nom_choisi):
    return ChargeurDonnees.charger_scikit(nom_choisi)




def configurer_page():
    """Initialise le titre et les paramètres de la page Streamlit."""
    st.set_page_config(page_title="MyVotingClassifier App", layout="centered")
    st.title("Bienvenue sur l'application de visualisation des performances de MyVotingClassifier !" )
    st.text("Cette application vous permet de charger différents datasets, d'explorer leurs caractéristiques de montrer leur modélisation avec MyT-SNE et de visualiser les résultats de MyVotingClassifier.")
    st.write("---")


def selectionner_dataset():
    """Gère le menu déroulant et le chargement des données. 
    Arrête l'exécution si rien n'est sélectionné."""
    
    st.write("### Sélection du dataset")
    liste_ds = ChargeurDonnees.lister_datasets_scikit()
        
    choix = st.selectbox("Choisissez le dataset à charger :", liste_ds, index=None, placeholder="--- Sélectionnez un dataset ---")
    # Sécurité si rien n'est sélectionné, on arrête l'exécution et on affiche un message d'information
    if choix is None:
        st.info("Veuillez sélectionner un dataset dans le menu déroulant")
        st.stop()
        
    with st.spinner(f"Chargement de {choix} en cours"):
        X, y, noms_colonnes, noms_classes = charger_donnees_cachees(choix)
        
    return choix, X, y, noms_colonnes, noms_classes

def afficher_exemple_visuel(choix, X, y, noms_colonnes):
    """Affiche une image ou un extrait de tableau selon le type de dataset."""
    st.write("### Un exemple visuel du dataset")
    
    # Initialisation de la mémoire pour ce bouton 
    if 'afficher_img' not in st.session_state:
        st.session_state.afficher_img = False

    # Bouton qui inverse l'état à chaque clic
    if st.button("Afficher / Masquer l'exemple"):
        st.session_state.afficher_img = not st.session_state.afficher_img

    if st.session_state.afficher_img:
        # Affichage d'une image pour les datasets d'images, sinon affichage d'un extrait de tableau
        img_data = X[0]
        if img_data.max() > 0:
            img_data = img_data / img_data.max()

        if "MNIST" in choix:
            st.image(img_data.reshape(28, 28), width=150, caption=f"Exemple d'une image ({choix}) avec le label {y[0]}")
        elif "Digits" in choix:
            st.image(img_data.reshape(8, 8), width=150, caption=f"Exemple d'une image (Digits) avec le label {y[0]}")
        elif "Olivetti" in choix:
            st.image(img_data.reshape(64, 64), width=150, caption=f"Exemple de Visage avec le label {y[0]}")
        else:
            st.write(f"Exemple de la 1ère ligne (Classe cible : {y[0]})")
            
            exemple_dict = {}
            for i in range(min(5, len(noms_colonnes))):
                exemple_dict[noms_colonnes[i]] = X[0][i]
                
            st.json(exemple_dict) 

    st.write("---")

def afficher_statistiques(X, noms_classes):
    """Affiche les metriques et les noms des classes."""
    with st.expander("Voir toutes les Classes (y)"):
        st.write(noms_classes)
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Nombre d'exemples (Lignes)", X.shape[0])
    col_stat2.metric("Nombre de Variables (Colonnes)", X.shape[1])
    col_stat3.metric("Nombre de Classes uniques à prédire", len(noms_classes))

    st.write("---")

def executer_tsne(X_matrix, perplexity, max_iter, init, methode_norm):
    """
    Exécution du modèle t-SNE sans cache.
    """
    # Normalisation
    norm = Normaliseur(methode=methode_norm)
    # On laisse la méthode de normalisation deviner automatiquement la meilleure méthode selon les données
    X_scale = norm.fit_transform(X_matrix)
    
    # t-SNE
    tsne = MyTSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, init=init)
    # On utilise la méthode fit_transform pour calculer la réduction de dimension et obtenir les coordonnées 2D
    X_2d = tsne.fit_transform(X_scale)
    
    return tsne, X_2d

def section_tsne(X, y):
    """Gère la section permettant la réduction de dimension via MyT-SNE."""
    st.write("### Réduction 2D avec MyT_SNE")
    st.write("Le t-SNE permet de compresser nos données complexes vers 2 dimensions pour les visualiser.")
    
    # Paramètres interactifs
    col1, col2, col3 = st.columns(3)
    max_donnees = X.shape[0]
    
    # On force une limite pour éviter les erreurs et un temps d'éxécution trop long 
    limite_absolue = min(max_donnees, 10000) 
    valeur_par_defaut = min(max_donnees, 800)
    # On propose à l'utilisateur de choisir combien de données il veut utiliser pour le t-SNE 
    echantillon = col1.slider("Nb données ( Limité à 10000 pour le temps d'éxécution)", min_value=50, max_value=limite_absolue, value=valeur_par_defaut, step=50)
    perplexite = col2.slider("Perplexité", min_value=5.0, max_value=50.0, value=30.0, step=1.0)
    max_iter = col3.slider("Itérations", min_value=250, max_value=1000, value=400, step=50)
    # Choix de l'initialisation de départ (PCA ou aléatoire)
    col4, col5 = st.columns(2)
    init = col4.radio("Initialisation de départ", ["pca", "random"], horizontal=True)
    # On laisse la méthode de normalisation deviner automatiquement la meilleure méthode selon les données
    methode_auto = Normaliseur.deviner_meilleure_methode(X.to_numpy() if hasattr(X, "to_numpy") else X)

    if st.button("Calculer le t-SNE"):
        # On prélève un échantillon pour l'entraînement
        X_echantillon = X[:echantillon]
        y_echantillon = y[:echantillon]
        
        with st.spinner("Calcul en cours (cela peut prendre quelques secondes)"):
            tsne, X_2d = executer_tsne(X_echantillon, perplexite, max_iter, init, methode_auto)
            
            # Affichage de la figure générée par MyT_SNE en lui disant de ne pas "show()"
            fig = tsne.afficher(X=X_echantillon, y=y_echantillon, afficher_score=True, en_ligne=True)
            
            # On stocke la figure pour qu'elle survive aux rechargements de page
            st.session_state.fig_tsne = fig
            
    if "fig_tsne" in st.session_state:
        # On utilise clear_figure=False sinon on détruit la figure pyplot stockée
        st.pyplot(st.session_state.fig_tsne, clear_figure=False)
        
        # --- EXPLICATIONS DES SCORES ---
        st.write("") # Espace
        with st.expander(" Comprendre les scores de Confiance et de Silhouette"):
            st.markdown("""
            **1. Le Score de Confiance (Trustworthiness)**
            - *À quoi ça sert ?* Il vérifie si t-SNE a fait du bon travail en ne cassant pas les voisinages initiaux purs.
            - *Comment le lire ?* C'est une note entre 0 et 1. 
            - **Très bon : > 0.90**. 
            - Si le score est faible, cela signifie que deux points très éloignés en haute dimension (la réalité) se retrouvent affichés artificiellement proches sur le dessin en 2D ("Crowding Problem").
            
            **2. Le Score de Silhouette**
            - *À quoi ça sert ?* Il évalue la séparation spatiale visuelle des groupes (clusters) créés sur le dessin 2D final. Il compare la distance d'un point avec sa propre classe par rapport à sa distance avec la classe la plus proche.
            - *Comment le lire ?* C'est une note entre -1 et 1.
            - **Bon : Plus c'est proche de +1, plus c'est distinct et coloré sous forme d'îles bien séparées**. 
            - **Mauvais : Proche de 0 ou négatif**, cela veut dire que les points de différentes classes sont totalement mélangés au centre et que l'algorithme n'a pas pu distinguer de groupe logique.
            """)
            
    st.write("---")

def section_voting_classifier(X_train, y_train, X_test, y_test):
    """Gère la section de visualisation des performances de MyVotingClassifier."""
    st.write(" ### Visualisation des performances de MyVotingClassifier et des autres modèles")
    
    # Dictionnaire des modèles de base disponibles
    modeles_disponibles = {
        "KNN (K-Nearest Neighbors)": KNeighborsClassifier(n_jobs=-1, n_neighbors=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=15, min_samples_leaf=2),
        "Arbre de Décision": DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_leaf=4),
        "SVM (Support Vector Machine)": SVC(probability=True, random_state=42),
        "Régression Logistique": LogisticRegression(max_iter=2000, random_state=42),
        "Ridge Classifier": RidgeClassifier(random_state=42),
        "Naïf Bayes": GaussianNB(),
        "Extra Trees": ExtraTreesClassifier(random_state=42, n_jobs=-1, max_depth=15, min_samples_leaf=2),
        "Bagging Classifier": BaggingClassifier(random_state=42, n_jobs=-1),
        "Linear SVC": LinearSVC(random_state=42, max_iter=1000)
    }
    
    # 2. Menu de sélection 
    noms_modeles_choisis = st.multiselect(
        "1. Choisissez les modèles de base à entraîner et à faire voter :",
        options=list(modeles_disponibles.keys()),
        default=[
            "KNN (K-Nearest Neighbors)", 
            "Random Forest", 
            "Arbre de Décision", 
            "SVM (Support Vector Machine)", 
            "Régression Logistique", 
            "Ridge Classifier", 
            "Naïf Bayes"
        ]
    )
    
    # Sécurité si rien n'est sélectionné
    if not noms_modeles_choisis:
        st.warning("Veuillez sélectionner au moins un algorithme pour continuer.")
        return
        
    st.info(f"{len(noms_modeles_choisis)} modèle(s) sélectionné(s) pour l'entraînement")
    
    # Menu de choix des métriques d'évaluation
    st.write("---")
    st.write("#### Choix des métriques d'évaluation")
    metriques_choisies = st.multiselect(
        "Quelles métriques voulez-vous calculer dans le tableau final ?",
        options=["Accuracy", "Précision", "Rappel", "F1-Score"],
        default=["Accuracy", "F1-Score"]
    )
    
    if not metriques_choisies:
        st.warning("Veuillez sélectionner au moins une métrique (ex: Accuracy).")
        return
    
    # On stocke ici la liste des modèles sous forme de (nom, instance) pour MyVotingClassifier
    estimateurs_choisis = [(nom, modeles_disponibles[nom]) for nom in noms_modeles_choisis]
    
    st.write("---")
    st.write("#### Options d'optimisation")
    choix_optimisation = st.radio(
        "Niveau d'optimisation automatique des modèles (GridSearch) :", 
        ["non", "rapide", "normal"], 
        index=0, 
        horizontal=True,
        help="'non' est le plus rapide. 'rapide' cherche de bons paramètres mais ce n'est pas parfait mais assez rapide. 'normal' cherche le test parfait mais peut prendre plusieurs minutes"
    )
    
    st.write("#### Options de rapidité de calcul")
    col1, col2 = st.columns(2)
    max_train_samples = col1.number_input(
        "Nombre max de données d'entraînement :", 
        min_value=100, 
        max_value=len(X_train), 
        value=min(len(X_train), 5000), 
        help="Limiter le nombre de données accélère considérablement l'entraînement (surtout pour de gros datasets comme MNIST)."
    )
    
    st.write("#### Lancement de l'Entraînement de l'ensemble")
    
    if st.button("Lancer MyVotingClassifier"):
        with st.spinner("Entraînement des modèles et des différents modes de Vote en cours..."):
            
            # Échantillonnage pour gagner du temps
            if len(X_train) > max_train_samples:
                indices = np.random.choice(len(X_train), max_train_samples, replace=False)
                X_train_use = X_train.iloc[indices] if hasattr(X_train, "iloc") else X_train[indices]
                y_train_use = y_train.iloc[indices] if hasattr(y_train, "iloc") else y_train[indices]
            else:
                X_train_use = X_train
                y_train_use = y_train
            
            resultats_tableau = []
            
            # Fonction utilitaire pour éviter de répéter le code de métrique
            def enregistrer_score(nom_algo, y_pred_train, y_pred_test):
                resultat = {"Algorithme": nom_algo}
                if "Accuracy" in metriques_choisies:
                    resultat["Accuracy (Train)"] = f"{accuracy_score(y_train_use, y_pred_train) * 100:.2f}%"
                    resultat["Accuracy (Test)"] = f"{accuracy_score(y_test, y_pred_test) * 100:.2f}%"
                if "Précision" in metriques_choisies:
                    resultat["Précision (Train)"] = f"{precision_score(y_train_use, y_pred_train, average='macro', zero_division=0) * 100:.2f}%"
                    resultat["Précision (Test)"] = f"{precision_score(y_test, y_pred_test, average='macro', zero_division=0) * 100:.2f}%"
                if "Rappel" in metriques_choisies:
                    resultat["Rappel (Train)"] = f"{recall_score(y_train_use, y_pred_train, average='macro', zero_division=0) * 100:.2f}%"
                    resultat["Rappel (Test)"] = f"{recall_score(y_test, y_pred_test, average='macro', zero_division=0) * 100:.2f}%"
                if "F1-Score" in metriques_choisies:
                    resultat["F1-Score (Train)"] = f"{f1_score(y_train_use, y_pred_train, average='macro', zero_division=0) * 100:.2f}%"
                    resultat["F1-Score (Test)"] = f"{f1_score(y_test, y_pred_test, average='macro', zero_division=0) * 100:.2f}%"
                return resultat

            # ---- Normalisation des données ----
            # On laisse la méthode de normalisation deviner automatiquement la meilleure méthode selon les données
            # On convertit en numpy array au cas où c'est un DataFrame ou autre format, car la méthode de devinage attend un format numpy
            methode_auto = Normaliseur.deviner_meilleure_methode(X_train_use.to_numpy() if hasattr(X_train_use, "to_numpy") else X_train_use)
            normaliseur = Normaliseur(methode=methode_auto)
            X_train_scale = normaliseur.fit_transform(X_train_use)
            X_test_scale = normaliseur.transform(X_test)

            # ---- MyVotingClassifier (Mode HARD) ---
            # Ajout du multithreading n_jobs=-1
            mvc_hard = MyVotingClassifier(estimators=estimateurs_choisis, voting='hard', n_jobs=-1)
            # L'étape fit pour mvc_hard va utiliser X_train_use (l'échantillon ou données complètes)
            mvc_hard.fit(X_train_scale, y_train_use, auto_optimize=choix_optimisation)
            predictions_hard_train = mvc_hard.predict(X_train_scale)
            predictions_hard_test = mvc_hard.predict(X_test_scale)
            resultats_tableau.append(enregistrer_score("MyVotingClassifier (Vote HARD)", predictions_hard_train, predictions_hard_test))
            # Les estimateurs optimisés par la classe Trouve_params sont contenus dans mvc_hard.estimators
            estimateurs_deja_optimises = mvc_hard.estimators
            
            # ---- MyVotingClassifier (Mode SOFT) ---
            try:
                # Mode Soft : On ne garde en amont que les estimateurs qui possèdent la méthode predict_proba necessaire pour soft 
                estimateurs_soft = [(nom, mod) for nom, mod in estimateurs_deja_optimises if hasattr(mod, "predict_proba")]
                
                if not estimateurs_soft:
                    raise Exception("Aucun des modèles sélectionnés ne supporte predict_proba.")
                    
                mvc_soft = MyVotingClassifier(estimators=estimateurs_soft, voting='soft', n_jobs=-1)
                # On met auto_optimize='non' car déjà optimisés 
                mvc_soft.fit(X_train_scale, y_train_use, auto_optimize='non')
                predictions_soft_train = mvc_soft.predict(X_train_scale)
                predictions_soft_test = mvc_soft.predict(X_test_scale)
                resultats_tableau.append(enregistrer_score("MyVotingClassifier (Vote SOFT)", predictions_soft_train, predictions_soft_test))
                
            except Exception as e:
                erreur_soft = {"Algorithme": "MyVotingClassifier (Vote SOFT)"}
                for m in metriques_choisies:
                    erreur_soft[f"{m} (Train)"] = f"Impossible : {str(e)}"
                    erreur_soft[f"{m} (Test)"] = f"Impossible : {str(e)}"
                resultats_tableau.append(erreur_soft)
                
            #  ---- MyVotingClassifier (Mode Source & Faits) ---
            mvc_sf = MyVotingClassifier(estimators=estimateurs_deja_optimises, voting='S&F', n_jobs=-1)
            mvc_sf.fit(X_train_scale, y_train_use, auto_optimize='non')
            predictions_sf_train = mvc_sf.predict(X_train_scale)
            predictions_sf_test = mvc_sf.predict(X_test_scale)
            resultats_tableau.append(enregistrer_score("MyVotingClassifier (Source & Faits)", predictions_sf_train, predictions_sf_test))
            
            #  ---- Les Modèles Individuels ----
            # Les estimateurs ont été entraînés dans mvc_hard, on s'en sert pour afficher le mode individuel
            for nom_modele_base, modele_entraine in mvc_hard.named_estimators_.items():
                predictions_base_train = modele_entraine.predict(X_train_scale)
                predictions_base_test = modele_entraine.predict(X_test_scale)
                resultats_tableau.append(enregistrer_score(f"{nom_modele_base}", predictions_base_train, predictions_base_test))

            # Affichage du tableau
            st.success("Modèles entraînés et évalués avec succès sur les données d'entraînement (80%) et de test (20%) !")
            st.dataframe(resultats_tableau, width='stretch')
            
            st.write("") 
            with st.expander("Comprendre les métriques d'évaluation"):
                st.markdown("""
                **1. Accuracy (Exactitude)**
                - *À quoi ça sert ?* C'est le pourcentage global de prédictions correctes. C'est la métrique la plus intuitive.
                - *Attention :* Elle peut être très trompeuse si les classes sont déséquilibrées (ex: 99% de photos de chiens et 1% de chats, dire "chien" à chaque fois donne 99% d'accuracy).
                
                **2. Précision**
                - *À quoi ça sert ?* Elle permet d'éviter les **fausses alertes** (les "Faux Positifs").
                - *En pratique :* Sur toutes les fois où le modèle a prédit une certaine classe, combien de fois avait-il vraiment raison ? 

                **3. Rappel (Recall ou Sensibilité)**
                - *À quoi ça sert ?* Il permet d'éviter de **rater des choses importantes** (les "Faux Négatifs").
                - *En pratique :* Sur tous les vrais cas qui existent dans la réalité, quel pourcentage le modèle a-t-il réussi à détecter ? 

                **4. F1-Score**
                - *À quoi ça sert ?* C'est la **moyenne juste** (moyenne harmonique) entre la Précision et le Rappel.
                - *En pratique :* Très strict ! Il exige que la Précision ET le Rappel soient tous les deux excellents pour avoir une bonne note globale. C'est souvent la métrique sur laquelle on se base en priorité.
                """)


def main():
    configurer_page()
    choix, X, y, noms_colonnes, noms_classes = selectionner_dataset()
    
    # Division du jeu de données : 80% pour l'entraînement/visualisation, 20% gardés pour plus tard (les tests)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # On affiche et utilise uniquement les données d'entraînement (80%)
    afficher_exemple_visuel(choix, X_train, y_train, noms_colonnes)
    afficher_statistiques(X_train, noms_classes)
    
    section_tsne(X_train, y_train)
    
    section_voting_classifier(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()