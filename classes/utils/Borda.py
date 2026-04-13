class CalculateurBorda:
    @staticmethod
    def calculer(liste_scores_configurations):
        if not liste_scores_configurations:
            return {}
        candidats = list(liste_scores_configurations[0].keys())
        scores_borda = {c: 0 for c in candidats}
        for config_scores in liste_scores_configurations:
            for i in range(len(candidats)):
                candidat_A = candidats[i]
                score_A = config_scores.get(candidat_A, 0)
                for j in range(i + 1, len(candidats)):
                    candidat_B = candidats[j]
                    score_B = config_scores.get(candidat_B, 0)
                    if score_A > score_B:
                        scores_borda[candidat_A] += 2
                    elif score_A < score_B:
                        scores_borda[candidat_B] += 2
                    else:
                        scores_borda[candidat_A] += 1
                        scores_borda[candidat_B] += 1
        classement_trie = dict(sorted(scores_borda.items(), key=lambda item: item[1], reverse=True))
        return classement_trie

    @staticmethod
    def afficher_classement(classement_trie, nb_confrontations, moyennes_metriques=None):
        print("\n" + "="*85)
        print(f" CLASSEMENT FINAL BORDA (Basé sur {nb_confrontations} configurations)")
        print("="*85)
        position = 1
        for candidat, points in classement_trie.items():
            ligne = f" {position}er | {candidat:<20} : {points} pts"
            if moyennes_metriques and candidat in moyennes_metriques:
                m = moyennes_metriques[candidat]
                ligne += f" | F1: {m['f1']:.1f}% | Précision: {m['precision']:.1f}% | Rappel: {m['recall']:.1f}% | Acc: {m['accuracy']:.1f}%"
            print(ligne)
            position += 1
        print("="*85 + "\n")