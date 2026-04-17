class CalculateurBorda:
    """
    Classe utilitaire pour appliquer la méthode de Borda modifiée sur un ensemble de résultats.
    Règle appliquée pour chaque duel (A contre B) dans une configuration donnée :
    - Si Score(A) > Score(B) : A gagne 2 points.
    - Si Score(A) == Score(B) : A et B gagnent 1 point chacun.
    - Si Score(A) < Score(B) : A gagne 0 point.
    """

    @staticmethod
    def calculer(liste_scores_configurations):
        """
        liste_scores_configurations : liste de dictionnaires.
        Chaque dictionnaire représente un "Cas" (ex: la grille '0' en 50pct_2ano)
        et contient les scores (ex: F1-Score) des 9 candidats.
        
        Exemple :
        [
            {'IF': 100, 'LOF': 90, 'EE': 45, 'Min': 45, 'Max': 100, 'Avg': 78, 'Hard': 100, 'Soft': 45, 'S&F': 100},
            {'IF': 80, 'LOF': 80, ...},
            ... (x40)
        ]
        
        Retourne :
        Dictionnaire des scores finaux de Borda, trié du gagnant absolu au perdant.
        """
        if not liste_scores_configurations:
            return {}

        # 1. Identifier tous les candidats depuis la première configuration
        # on prend la première configuration pour extraire les noms des candidats (ex: 'IF', 'LOF', etc.)
        candidats = list(liste_scores_configurations[0].keys())
        # Initialiser le score de Borda pour chaque candidat à 0
        scores_borda = {c: 0 for c in candidats}

        # 2. Parcourir toutes les configurations (tous les classements ou jeux de données)
        for config_scores in liste_scores_configurations:
            
            # Comparer chaque candidat avec tous les autres dans ce classement précis
            for i in range(len(candidats)):
                candidat_A = candidats[i]
                score_A = config_scores.get(candidat_A, 0) # Sécurité si un score est manquant
                
                for j in range(i + 1, len(candidats)):
                    candidat_B = candidats[j]
                    score_B = config_scores.get(candidat_B, 0)
                    
                    # Attribution des points selon les règles de Borda modifiées
                    if score_A > score_B:
                        scores_borda[candidat_A] += 2
                    elif score_A < score_B:
                        scores_borda[candidat_B] += 2
                    else:
                        # Égalité
                        scores_borda[candidat_A] += 1
                        scores_borda[candidat_B] += 1

        # 3. Trier le dictionnaire du vainqueur au perdant
        classement_trie = dict(sorted(scores_borda.items(), key=lambda item: item[1], reverse=True))
        
        return classement_trie

    @staticmethod
    def afficher_classement(classement_trie, nb_confrontations):
        """
        Affiche proprement le classement final dans la console.
        """
        print("\n" + "="*50)
        print(f" CLASSEMENT FINAL BORDA (Basé sur {nb_confrontations} configurations)")
        print("="*50)
        
        position = 1
        for candidat, points in classement_trie.items():
            print(f" {position}er | {candidat:<15} : {points} pts")
            position += 1
        print("="*50 + "\n")
