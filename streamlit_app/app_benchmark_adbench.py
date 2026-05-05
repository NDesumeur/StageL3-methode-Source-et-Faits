import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classes.MyVotingPyOD import MyVotingPyOD
from classes.utils.Trouve_params_pyod import Trouve_params_pyod
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sos import SOS


def list_adbench_datasets(adbench_dir: str) -> list[str]:
    if not os.path.exists(adbench_dir):
        return []
    return sorted([f for f in os.listdir(adbench_dir) if f.endswith(".npz")])


def charger_dataset_adbench(adbench_dir: str, nom_fichier: str):
    chemin = os.path.join(adbench_dir, nom_fichier)
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Dataset introuvable: {chemin}")

    data = np.load(chemin, allow_pickle=True)
    X = data["X"]
    y = np.array(data["y"], dtype=int)
    return X, y


def compute_metrics(y_true, y_pred, y_scores, prefix: str) -> dict[str, float]:
    return {
        f"{prefix} Accuracy": accuracy_score(y_true, y_pred),
        f"{prefix} ROC-AUC": roc_auc_score(y_true, y_scores),
        f"{prefix} PR-AUC": average_precision_score(y_true, y_scores),
        f"{prefix} F1-Score": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix} Précision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix} Rappel": recall_score(y_true, y_pred, zero_division=0),
    }


def evaluer_modele(modele, X_train, y_train, X_test, y_test):
    modele.fit(X_train)

    y_pred_train = modele.predict(X_train)
    y_scores_train = modele.decision_function(X_train)

    y_pred = modele.predict(X_test)
    y_scores = modele.decision_function(X_test)

    metrics = {}
    metrics.update(compute_metrics(y_train, y_pred_train, y_scores_train, "Train"))
    metrics.update(compute_metrics(y_test, y_pred, y_scores, "Test"))
    return metrics


def build_models(contamination: float, n_features: int):
    modeles = {
        "IForest": IForest(contamination=contamination, random_state=42),
        "LOF": LOF(contamination=contamination),
        "KNN": KNN(contamination=contamination),
        "OCSVM": OCSVM(contamination=contamination),
        "PCA": PCA(contamination=contamination, random_state=42),
        "CBLOF": CBLOF(contamination=contamination, random_state=42),
        "COF": COF(contamination=contamination),
        "HBOS": HBOS(contamination=contamination),
        "LODA": LODA(contamination=contamination),
        "COPOD": COPOD(contamination=contamination),
        "ECOD": ECOD(contamination=contamination),
        "SOS": SOS(contamination=contamination),
        "DeepSVDD": DeepSVDD(
            contamination=contamination,
            random_state=42,
            verbose=0,
            n_features=n_features,
        ),
    }

    return modeles


def run_benchmark(
    adbench_dir: str,
    dataset_name: str,
    cv: int,
    test_size: float,
    optim_scoring: str,
    vote_metric: str,
    selected_models: list[str],
    selected_strategies: list[str],
):
    logs: list[str] = []
    t0 = time.time()

    X, y = charger_dataset_adbench(adbench_dir, dataset_name)
    logs.append(f"Dataset {dataset_name}: X={X.shape}, anomalies={int(np.sum(y))}")

    contamination = max(0.01, float(np.sum(y) / len(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    scaler = RobustScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    logs.append(
        f"Train={X_train_norm.shape}, Test={X_test_norm.shape}, contamination~{contamination:.4f}"
    )

    modeles = build_models(contamination, X_train_norm.shape[1])
    modeles = {k: v for k, v in modeles.items() if k in selected_models}

    optimiseur = Trouve_params_pyod(X_train_norm, y_train, cv=cv, scoring=optim_scoring)

    resultats = {}
    modeles_optimises_pour_voting = []

    for nom, mod in modeles.items():
        try:
            logs.append(f"Optimisation {nom}...")
            mod_opt = optimiseur.trouve_params(mod)
            metrics = evaluer_modele(mod_opt, X_train_norm, y_train, X_test_norm, y_test)
            resultats[nom] = metrics
            modeles_optimises_pour_voting.append((nom, mod_opt))
            logs.append(
                f"{nom}: Test F1={metrics['Test F1-Score']:.4f}, Test Recall={metrics['Test Rappel']:.4f}, Test Acc={metrics['Test Accuracy']:.4f}"
            )
        except Exception as e:
            logs.append(f"ECHEC {nom}: {e}")

    if modeles_optimises_pour_voting:
        # Les poids d'initialisation de l'ensemble doivent venir du train pour éviter toute fuite du test.
        poids_train = [
            max(resultats[nom]["Train Accuracy"], 1e-6)
            for nom, _ in modeles_optimises_pour_voting
        ]

        for strat in selected_strategies:
            nom_vote = f"Ensemble_{strat}"
            try:
                vote_model = MyVotingPyOD(
                    estimators=modeles_optimises_pour_voting,
                    voting=strat,
                    weights=poids_train,
                    verbose=False,
                    vote_metric=vote_metric,
                    threshold_metric="accuracy",
                )
                vote_model.fit(X_train_norm, y_train)
                y_pred = vote_model.predict(X_test_norm)

                if strat == "hard":
                    y_pred_train = vote_model.predict(X_train_norm)
                    y_scores_train = y_pred_train.astype(float)
                    y_scores = y_pred.astype(float)
                else:
                    y_pred_train = vote_model.predict(X_train_norm)
                    y_scores_train = vote_model.decision_function(X_train_norm)
                    y_scores = vote_model.decision_function(X_test_norm)

                metrics_vote = {}
                metrics_vote.update(compute_metrics(y_train, y_pred_train, y_scores_train, "Train"))
                metrics_vote.update(compute_metrics(y_test, y_pred, y_scores, "Test"))
                resultats[nom_vote] = metrics_vote
                logs.append(
                    f"{nom_vote}: Test F1={resultats[nom_vote]['Test F1-Score']:.4f}, Test Recall={resultats[nom_vote]['Test Rappel']:.4f}, Test Acc={resultats[nom_vote]['Test Accuracy']:.4f}"
                )
            except Exception as e:
                logs.append(f"ECHEC {nom_vote}: {e}")

    if not resultats:
        raise RuntimeError("Aucun résultat produit. Vérifier les modèles sélectionnés et les données.")

    df_res = pd.DataFrame(resultats).T
    ordered_cols = [
        "Train Accuracy",
        "Train ROC-AUC",
        "Train PR-AUC",
        "Train F1-Score",
        "Train Précision",
        "Train Rappel",
        "Test Accuracy",
        "Test ROC-AUC",
        "Test PR-AUC",
        "Test F1-Score",
        "Test Précision",
        "Test Rappel",
    ]
    df_res = df_res[[c for c in ordered_cols if c in df_res.columns]]

    modeles_base = [nom for nom in df_res.index if not nom.startswith("Ensemble_")]
    if modeles_base:
        df_base = df_res.loc[modeles_base]
        df_res.loc["[STAT] Pire Modèle (MIN)"] = df_base.min()
        df_res.loc["[STAT] Moyenne Globale (AVG)"] = df_base.mean()
        df_res.loc["[STAT] Meilleur Modèle (MAX)"] = df_base.max()

    df_res = df_res.sort_values(by="Test F1-Score", ascending=False)
    elapsed = time.time() - t0
    logs.append(f"Terminé en {elapsed:.1f}s")

    return df_res, logs


def main():
    st.title("Benchmark ADBench PyOD + Ensembles")
    st.caption("Lance les modèles PyOD optimisés et compare hard/soft/S&F avec un réglage simple.")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    adbench_dir = os.path.join(project_root, "data", "adbench")

    datasets = list_adbench_datasets(adbench_dir)
    if not datasets:
        st.error(f"Aucun dataset .npz trouvé dans {adbench_dir}")
        return

    st.sidebar.header("Configuration")
    dataset_name = st.sidebar.selectbox("Dataset", datasets, index=min(18, len(datasets) - 1))
    cv = st.sidebar.slider("CV folds", min_value=2, max_value=5, value=2)
    test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.5, value=0.3, step=0.05)

    optim_scoring = st.sidebar.selectbox("Scoring optimiseur (Trouve_params_pyod)", ["accuracy", "f1", "roc_auc"], index=0)
    vote_metric = st.sidebar.selectbox("Métrique d'optimisation des poids S&F", ["accuracy", "f1"], index=0)

    all_models = list(build_models(0.1, 10).keys())
    selected_models = st.multiselect("Modèles à lancer", all_models, default=all_models)

    selected_strategies = st.multiselect(
        "Stratégies ensemble",
        ["hard", "soft", "S&F"],
        default=["hard", "soft", "S&F"],
    )

    run = st.button("Lancer le benchmark", type="primary", use_container_width=True)

    if not run:
        st.info("Choisis les options puis clique sur 'Lancer le benchmark'.")
        return

    if not selected_models:
        st.warning("Sélectionne au moins un modèle.")
        return

    if not selected_strategies:
        st.warning("Sélectionne au moins une stratégie d'ensemble.")
        return

    with st.spinner("Benchmark en cours... cela peut prendre plusieurs minutes."):
        try:
            df_res, logs = run_benchmark(
                adbench_dir=adbench_dir,
                dataset_name=dataset_name,
                cv=cv,
                test_size=test_size,
                optim_scoring=optim_scoring,
                vote_metric=vote_metric,
                selected_models=selected_models,
                selected_strategies=selected_strategies,
            )
        except Exception as e:
            st.error(f"Erreur pendant le benchmark: {e}")
            st.code(traceback.format_exc())
            return

    st.success("Benchmark terminé.")


    st.subheader("Résumé des performances")
    st.info(
        """
        Comment lire les métriques:
        - Accuracy: sur 100 exemples, combien sont correctement classés. Attention: en dataset très déséquilibré, elle peut paraître haute même si peu d'anomalies sont détectées.
        - ROC-AUC: mesure la qualité du classement des scores (anomalies au-dessus des normaux) sur tous les seuils. 0.5 = aléatoire, 1.0 = parfait.
        - PR-AUC: synthèse précision/rappel selon le seuil, généralement plus informative que ROC-AUC en détection d'anomalies rares.
        - F1-Score: équilibre entre précision et rappel pour un seuil fixe. Monte seulement si les deux progressent.
        - Précision: parmi les points prédits anomalies, quelle fraction est vraiment anormale.
        - Rappel: parmi les anomalies réelles, quelle fraction est retrouvée.

        Train vs Test:
        - Colonnes Train = performance sur les données d'entraînement.
        - Colonnes Test = performance sur des données jamais vues.
        - Si Train est très supérieur à Test, il y a risque de surapprentissage.
        """
    )
    st.dataframe(df_res.style.format("{:.6f}"), use_container_width=True)


if __name__ == "__main__":
    main()
