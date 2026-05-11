# RAPPORT DE STAGE L3 INFORMATIQUE

---

## Informations administratives

**Stagiaire :** Nolan Desumeur  
**Formation :** Licence 3 Informatique  
**Université :** Université d'Artois - Faculté des Sciences Jean Perrin  
**Période du stage :** 23 mars 2026 - 15 mai 2026 (8 semaines)  
**Structure d'accueil :** CRIL (Centre de Recherche en Informatique de Lens)  
**Encadrant académique :** Mr Delorme  

**Sujet du stage :**  
*Utilisation de la méthode 'Sources & Faits' pour la détection d'anomalies*

**Tâches principales :**  
Développement d'un module logiciel mettant en œuvre une méthode de vote pour les méthodes ensemblistes de détection automatique d'anomalies.

---

Logos

---

## Table des matières

*Cette table sera générée automatiquement lors de la conversion en LibreOffice Writer*

1. Introduction
2. Contexte du stage
3. Travail effectué
4. Bilan personnel
5. Conclusion

---

I. Introduction

La détection d'anomalies est un domaine important du machine learning qui vise à identifier des observations anormales dans des données sans étiquettes préalables. Contrairement à la classification supervisée classique, on ne dispose pas toujours de labels fiables ; on cherche simplement à isoler ce qui s'écarte significativement de la normale.

Ce problème est fondamental dans de nombreuses applications : détection de fraude bancaire, surveillance de systèmes critiques, identification de défauts en fabrication, ou détection d'intrusions informatiques. Dans tous ces contextes, les anomalies sont **rares et critiques**.

Le défi principal est que **aucun algorithme n'est universellement meilleur**. Chaque détecteur (Isolation Forest, LOF, One-Class SVM, etc.) capture différents types d'anomalies : certains excellent sur anomalies isolées, d'autres sur anomalies de groupe. D'où l'idée naturelle de **combiner plusieurs détecteurs** par vote d'ensemble.

Les méthodes traditionnelles présentent toutefois des limites. Le **hard voting** (prise de vote unique) ignore l'information de confiance des détecteurs : un modèle fiable à 95% pèse autant qu'un modèle à 51%. Le **soft voting** (moyenne des scores) reste simple : on additionne les scores sans tenir compte de la qualité réelle de chaque modèle. Aucune des deux approches n'adapte les contributions en fonction des performances observées.

Ce stage explore une approche spécifique : la méthode **Sources & Faits**, qui apprend automatiquement des poids adaptatifs pour chaque détecteur. Au lieu de poids fixes, chaque modèle reçoit une pondération basée sur sa performance observée, ajustée itérativement jusqu'à convergence. Cette étude m'a permis de **découvrir concrètement les fondamentaux du machine learning** (validation croisée, c'est-a-dire un découpage répété des données en sous-ensembles pour entraîner et tester de façon robuste, ensemble learning, gestion de données déséquilibrées), ce qui est essentiel pour préparer un master en IA. Les résultats de mes recherches indiquent généralement une amélioration de Sources & Faits par rapport aux approches classiques, d'où l'intérêt d'une validation sérieuse.

L'objectif concret du stage est de réaliser une application permettant de tester l'efficacité de la méthode Sources & Faits, puis de la comparer aux approches hard et soft voting sur plusieurs jeux de données. Le travail consiste à tester plusieurs méthodes de détection, à produire des résultats mesurables et à les analyser. Le rapport est organisé comme suit : une présentation du contexte du stage, une description du travail effectué, un bilan personnel, puis une conclusion.

II. Contexte du stage

### 1. Comment j'ai obtenu le stage

Ma recherche de stage a débuté en novembre avec l'envoi de candidatures massives. Après environ trois mois (mi-février), j'avais reçu **50 réponses négatives et aucune positive**. 

À ce moment critique, à la fin d'un cours, M. Delorme a mentionné qu'il pourrait éventuellement accueillir un stagiaire. J'ai immédiatement envoyé un mail directement après ce cours. Ce contact académique s'est avéré décisif car, contrairement aux candidatures entreprises, il était directement lié à mon projet personnel : je souhaite poursuivre avec un master en IA, et un stage sur ce sujet me fournissait des bases solides en machine learning. M. Delorme m'a proposé un stage de 8 semaines au CRIL, débutant le 23 mars 2026.

### 2. La structure d'accueil

Le stage s'est déroulé au CRIL (Centre de Recherche en Informatique de Lens), rattaché à la Faculté Jean Perrin de l'Université d'Artois. Je travaillais principalement en autonomie sous la supervision de M. Delorme, avec des points réguliers (réunions hebdomadaires ou bihebdomadaires) pour présenter l'avancement et recevoir des retours.

**Environnement physique et ressources :**
- Lieu de travail : salle de classe utilisée par le master Jeux Vidéo
- Matériel : un ordinateur personnel mis à disposition pour l'intégralité du stage
- Accès aux ressources : utilisation de la documentation en ligne de scikit-learn et PyOD, aucune autre ressource matérielle requise

### 3. Le sujet proposé

Le sujet proposé est « Utilisation de la méthode 'Sources & Faits' pour la détection d'anomalies ». La méthode Sources & Faits est une stratégie d'agrégation qui apprend des coefficients de pondération adaptatifs pour chaque modèle : elle commence par un vote simple, puis optimise itérativement les poids en fonction des performances jusqu'à convergence. On a appliqué cette méthode à des modèles de détection d'anomalies et on l'a comparée aux approches de vote classiques (hard et soft voting).

III. Travail Effectué

### 1. Presentation detaillee du projet auquel j'ai participe et mise en perspective

Le projet consistait a mettre en place, tester et comparer la méthode Sources & Faits dans un cadre concret. Pour y parvenir, j'ai developpe trois mini-applications successives. Toutes ont été realisees avec Streamlit pour obtenir rapidement une interface web simple, lisible et utile a chaque etape. Cette progression m'a permis d'apprendre les bases, d'experimenter, puis d'arriver a une application finale directement alignee avec l'objectif du stage.

**Application 1 : classification (scikit-learn).**
Cette premiere application avait un but clair : me familiariser avec le monde du machine learning, comprendre les métriques, et construire une premiere version d'un vote d'ensemble. La premiere semaine, je n'avais pas d'instruction claire, mis a part me familiariser avec le machine learning. J'ai donc travaille la documentation scikit-learn et numpy, ainsi que les notions indispensables (classification, validation croisee, decoupage train/test, précision, rappel, F1-score), tout en realisant des tests sur les modeles.
Cette phase m'a permis de me mettre a niveau rapidement : j'ai observe le comportement des algorithmes, compare leur stabilite selon les paramètres, et appris a lire les résultats sans me limiter a une seule métrique. J'ai aussi compris l'importance d'avoir un protocole de test stable, afin que les comparaisons restent cohérentes d'une execution a l'autre.
La semaine suivante, j'ai commence a creer la classe **MyVotingClassifier**, qui reproduit le fonctionnement de scikit-learn en y ajoutant la méthode Sources & Faits. J'ai aussi cree **MyTsne** pour visualiser les données, et des classes utilitaires comme **Trouve_params**, qui prend un modele et cherche les meilleurs paramètres via validation croisee. Cette classe m'a permis d'explorer une recherche de grille et une recherche plus rapide, ce qui était essentiel pour tester plusieurs modeles sans y passer trop de temps.

J'ai utilise plusieurs modeles de classification fournis par mon tuteur ou ajoutes pour avoir une liste plus large :
- KNN (K-Nearest Neighbors)
- Random Forest
- Arbre de Décision
- SVM (Support Vector Machine)
- Regression Logistique
- Ridge Classifier
- Naïf Bayes
- Extra Trees
- Bagging Classifier
- Linear SVC

Les données provenaient de plusieurs jeux scikit-learn, dont MNIST. J'ai mis en place les métriques classiques (Accuracy, Précision, Rappel, F1-Score). L'accuracy indique la proportion totale de bonnes predictions. La précision indique, parmi les predictions positives, la part correcte. Le rappel indique, parmi les vrais positifs, la part détectée. Le F1-Score combine précision et rappel pour donner un score unique. J'ai du me familiariser avec la cross validation pour obtenir des résultats stables. Cette partie m'a oblige a comprendre la difference entre précision et rappel, l'équilibre mesure par le F1-Score, et l'impact du choix de métrique sur le classement des modeles, notamment pourquoi l'accuracy seule n'est pas suffisante.
En pratique, j'ai constate que deux modeles pouvaient avoir une accuracy proche mais des comportements tres différents sur les erreurs. L'ajout de précision et rappel m'a permis de comprendre si un modele était plutot prudent ou plutot agressif, et le F1-Score m'a donne un compromis plus lisible.
L'application affichait un classement comparant les modeles individuels et le **MyVotingClassifier** dans ses trois versions (hard, soft, Sources & Faits). Le vote hard correspondait a une majorite simple, le vote soft reposait sur la moyenne des scores, et Sources & Faits ajoutait une ponderation adaptee a la performance observee. Cette comparaison m'a donne un premier apercu de l'intérêt d'un vote adaptatif.
J'ai également integre une visualisation t-SNE via **MyTsne** et travaille sur les scores de confiance ainsi que le score de silhouette pour interpréter les projections. Le score de silhouette mesure si des points proches appartiennent bien au meme groupe et si les groupes sont bien séparés : plus il est eleve, plus la séparation est nette. J'ai aussi du comprendre le fonctionnement de matplotlib pour afficher correctement les projections. Le t-SNE est une méthode de reduction de dimension qui projette des données de grande dimension en 2D ou 3D tout en conservant les voisinages locaux, ce qui aide a voir des groupes ou des points atypiques. Concrètement, **MyVotingClassifier** centralisait la logique de vote et fournissait une comparaison uniforme, tandis que **MyTsne** servait a comprendre visuellement la structure des données et les séparations obtenues. Cela m'a aide a relier les chiffres des métriques a des observations visuelles.

Exemple de résultat (MNIST) :

| Algorithme | Accuracy | Précision | Rappel | F1-Score |
|-----------|----------|-----------|--------|----------|
| SVM (Support Vector Machine) | 97.19% | 97.18% | 97.17% | 97.18% |
| MyVotingClassifier (Sources & Faits) | 96.32% | 96.33% | 96.29% | 96.31% |
| MyVotingClassifier (Vote SOFT) | 96.16% | 96.20% | 96.10% | 96.14% |
| KNN (K-Nearest Neighbors) | 96.16% | 96.24% | 96.09% | 96.14% |
| Random Forest | 96.11% | 96.09% | 96.09% | 96.09% |
| MyVotingClassifier (Vote HARD) | 95.91% | 95.95% | 95.86% | 95.90% |
| Regression Logistique | 91.51% | 91.42% | 91.39% | 91.40% |
| Ridge Classifier | 85.24% | 85.48% | 84.98% | 84.99% |
| Arbre de Décision | 85.16% | 84.92% | 84.93% | 84.91% |
| Naïf Bayes | 53.96% | 65.49% | 53.02% | 47.96% |

Dans cet exemple, Sources & Faits se comporte bien et reste au-dessus des votes hard et soft. L'amélioration est toutefois limitee car le nombre de modeles reste reduit, ce qui m'a montre qu'une vraie diversite de détecteurs est nécessaire pour que la méthode exprime tout son potentiel.
Ce résultat m'a confirme que l'approche S&F est pertinente, mais qu'elle reste sensible a la richesse de l'ensemble. Avec un nombre plus important et plus varie de modeles, l'effet du vote adaptatif devrait être plus visible.
Cette premiere application a aussi servi de base technique : elle m'a permis de mettre en place une organisation propre du code, de structurer les résultats en tableaux lisibles, et de verifier que les differences observees étaient cohérentes d'une execution a l'autre.
Elle m'a également permis de decouvrir scikit-learn et le fonctionnement de Streamlit, qui n'étaient pas forcement evidents au premier abord.
Finalement, cette etape m'a appris a produire un premier niveau de documentation interne : clarifier le role de chaque classe, noter les hypotheses de test, et formaliser les choix de métriques. Cela m'a servi pour la suite du projet. 

**Application 2 : détection d'anomalies (scikit-learn).**
L'objectif de cette application était de passer a la détection d'anomalies en restant sur des modeles simples. Les modeles utilisés étaient :

- Isolation Forest
- Local Outlier Factor
- Elliptic Envelope
- OCSVM lineaire
- OCSVM RBF
- Vote HARD 3, Vote SOFT 3, Vote S&F 3
- Vote HARD 5, Vote SOFT 5, Vote S&F 5

Cette application faisait la transition entre la classification supervisée et la détection d'anomalies. Elle m'a oblige a changer de logique, car l'objectif n'était plus de classer correctement toutes les données, mais de détecter un petit nombre de points anormaux au milieu d'une majorite normale.
Ici, les données normales sont simplement les images "habituelles" d'une classe (par exemple les vrais 0 de MNIST). Les anomalies sont des images d'autres classes qui ne devraient pas être la (par exemple des 1 ou des 7 ajoutes dans la classe 0).
Sur cette partie, je cherchais surtout a comprendre comment les modeles se comportaient quand la taille de l'echantillon et la proportion d'anomalies changeaient. Les memes modeles peuvent donner des résultats tres différents selon la taille des données, et le but était de mesurer l'efficacite de Sources & Faits sur des tailles variees.

Les données étaient basees sur MNIST (environ 1400 données) pour garder des temps de calcul raisonnables. Pour chaque classe, je prenais un pourcentage des images de la classe (100%, 50%, 25%, 10%) et j'ajoutais des anomalies prises dans les autres classes : 2 anomalies pour 100% et 50%, 1 anomalie pour 25% et 10%. Par exemple, pour la classe 0 : 100% des 0 + 2 images d'autres classes, et pareil pour chaque classe. L'idee était d'evaluer les modeles dans plusieurs situations de déséquilibre, et non sur un seul cas. Ce choix était intéressant pour voir les résultats avec peu de données, mais aussi pour visualiser les points un a un dans le t-SNE. Je pouvais alors afficher les erreurs (faux positifs et faux negatifs), ce qui était utile pour l'analyse.
Chaque configuration était evaluee de la meme maniere afin de pouvoir comparer les modeles a egalite. Cela m'a permis de construire un protocole de test repetable, avec les memes métriques et la meme logique de comparaison.
Cette etape était importante car la détection d'anomalies est naturellement déséquilibrée : les anomalies sont rares, et un modele peut sembler performant sur une configuration mais echouer sur une autre. En variant le nombre d'anomalies, j'ai pu observer la stabilite de chaque méthode et voir comment le vote reagissait selon le contexte.
Le classement des modeles était base sur un score de type Borda : a chaque comparaison, 2 points si un modele est superieur, 1 point si les performances sont equivalentes. Les métriques étaient calculees en train et en test (F1, Précision, Rappel, Accuracy). Lorsque l'utilisateur choisissait une seule configuration, l'application affichait une visualisation t-SNE avec la classe de chaque point pour voir comment les modeles separaient les données.
La lecture train/test me permettait de verifier si un modele restait cohérent et ne produisait pas des résultats artificiellement bons sur l'entrainement (surapprentissage). Cela ajoutait une verification simple avant d'interpréter le classement final.
Les versions 3 et 5 correspondaient a des ensembles de taille différente (3 ou 5 modeles). Cela m'a permis d'observer si augmenter le nombre de modeles ameliorait la stabilite du vote, ou si au contraire cela apportait peu de gain.
Cette comparaison entre 3 et 5 modeles était aussi un moyen simple d'evaluer l'impact de la taille de l'ensemble, sans changer le reste du protocole.
Le choix du score Borda permettait d'eviter qu'un modele soit juge uniquement sur un cas particulier. Un modele qui est regulierement bon sur plusieurs configurations obtient un score global meilleur qu'un modele qui gagne une seule fois mais est faible ailleurs. Cette logique est plus proche de l'objectif du stage, qui cherchait des performances robustes et pas seulement ponctuelles.

Exemple de classement Borda :

| Position | Modele | Points Borda | F1 Train (%) | F1 Test (%) | Précision Test (%) | Rappel Test (%) | Accuracy Test (%) |
|---------|--------|--------------|--------------|-------------|-------------------|----------------|-------------------|
| 1er | Modele MAX | 879 | 85.4 | 91.1 | 93.2 | 93.1 | 94.4 |
| 2eme | Local Outlier Factor | 674 | 79.5 | 82.5 | 82.9 | 85.5 | 89.1 |
| 3eme | Vote HARD 5 | 671 | 79.5 | 82.7 | 86.0 | 83.3 | 89.6 |
| 4eme | Vote S&F 5 | 671 | 79.5 | 82.7 | 86.0 | 83.3 | 89.6 |
| 8eme | Vote SOFT 5 | 548 | 78.0 | 73.6 | 73.8 | 76.1 | 86.3 |
| 13eme | OCSVM lineaire | 317 | 60.6 | 66.3 | 68.3 | 69.2 | 77.9 |
| 14eme | Modele MIN | 94 | 57.9 | 52.2 | 50.5 | 56.2 | 72.3 |

Dans ce contexte, a cause du faible nombre de données et de modeles, Sources & Faits donnait parfois exactement les memes résultats que le vote hard. Cela m'a montre les limites d'une experimentation trop restreinte. Quand les jeux étaient trop gros, j'ai utilise un echantillonnage pour garder des temps de calcul raisonnables.
J'ai donc considere cette application comme une etape d'apprentissage et de validation du protocole, plutot que comme un résultat definitif sur la performance de Sources & Faits.
L'echantillonnage était nécessaire pour garder un temps de calcul compatible avec les tests multiples. Sans cela, la boucle de comparaison devenait trop lente et rendait difficile l'analyse iterative.
L'application 2 m'a aussi permis de comprendre les limites d'une simple transposition des techniques de classification vers l'anomalie. Les modeles n'ayant pas tous la meme sensibilite, le classement changeait selon la proportion d'anomalies. Cette phase m'a servi de transition entre une approche supervisée et une approche non supervisée plus réaliste.

En pratique, j'ai vu que les classements pouvaient varier fortement selon la configuration choisie et que certains modeles étaient plus ou moins efficaces selon l'echantillon. Globalement, sur l'ensemble des tests, la méthode Sources & Faits restait plutot efficace, souvent devant le vote soft et devant la plupart des modeles.

**Application 3 : détection d'anomalies (PyOD).**
La troisieme application correspond directement au sujet du stage : utiliser et verifier Sources & Faits sur des données réelles de détection d'anomalies. L'application testait des jeux de données choisis parmi les 40 du benchmark ADBench. Les métriques suivaient le meme schema en train et en test : Accuracy, ROC-AUC, PR-AUC, F1-Score, Précision et Rappel. Le ROC-AUC mesure la capacite d'un modele a classer les anomalies avant les données normales pour tous les seuils possibles. Le PR-AUC mesure la qualite des détections positives quand la classe anormale est rare, ce qui est crucial en détection d'anomalies.
Cette application s'appuyait sur un jeu de données choisi dans ADBench a chaque execution. Cela donnait un contexte beaucoup plus proche de la réalité, avec des données variees et des structures d'anomalies différentes. L'objectif était de verifier si Sources & Faits restait fiable sur un ensemble de données plus large, et pas seulement sur MNIST.
PyOD propose de nombreux détecteurs. j'utilisais les modèles CBLOF, KNN, OCSVM, ECOD, LODA, Isolation Forest, PCA, LOF, DeepSVDD, COPOD, HBOS, SOS ou COF. Cela donnait un ensemble hétérogène, ce qui est justement utile pour tester un vote adaptatif.
J'ai dû m'adapter à PyOD qui est un peu différent de Sklearn, on a utilisé PyOD justement car c'est fait spécifiquement pour la détection d'anomalies, et ces modèles étant des modèles à apprentissage non supervisé j'ai dû gérer les choses différement et créer d'autres classes notamment un myVotingPyOD spécifiquement pour gérer les changements de résultats que me donnent les modèles PyOD qui ne sont pas comme Sklearn, et j'ai dû créer une nouvelle classe pour trouver les paramètres qui étaient plus nombreux en général puisque les modèles demandent une plus grosse configuration.
Pour cette application, les métriques étaient essentielles : l'accuracy restait informative mais insuffisante, et les indicateurs comme PR-AUC et F1-Score donnaient une vision plus réaliste de la capacite a détecter les anomalies rares. J'ai donc compare les modeles sur plusieurs indicateurs et non sur un seul chiffre.

| Modele | Train Accuracy | Train ROC-AUC | Train PR-AUC | Train F1 | Test Accuracy | Test ROC-AUC | Test PR-AUC | Test F1 |
|--------|----------------|---------------|--------------|---------|---------------|--------------|-------------|--------|
| [STAT] Meilleur Modele (MAX) | 0.9343 | 0.9464 | 0.6878 | 0.6185 | 0.9295 | 0.9322 | 0.6236 | 0.5967 |
| Ensemble_S&F | 0.9168 | 0.9297 | 0.5429 | 0.3618 | 0.9178 | 0.9303 | 0.5497 | 0.3982 |
| Ensemble_soft | 0.9176 | 0.9293 | 0.5395 | 0.4591 | 0.9141 | 0.9300 | 0.5460 | 0.5053 |
| Ensemble_hard | 0.9158 | 0.6123 | 0.2265 | 0.3484 | 0.9184 | 0.6359 | 0.2562 | 0.4000 |
| [STAT] Moyenne Globale (AVG) | 0.9055 | 0.8579 | 0.4560 | 0.3611 | 0.9023 | 0.8700 | 0.4597 | 0.3950 |

Sur certains jeux, les résultats restaient moyens : soit parce que les données étaient trop complexes, soit parce que certains modeles étaient trop lourds pour la machine disponible. Cette application m'a permis d'entrer dans une évaluation plus réaliste et de confronter la méthode a des jeux de données plus proches d'un contexte de recherche.
Cette phase m'a également oblige a manipuler des résultats plus nombreux et plus difficiles a interpréter. Le fait d'avoir plusieurs métriques a la fois m'a permis de voir qu'un modele pouvait être correct en Accuracy mais faible en F1 ou en PR-AUC. C'est une difference majeure avec la classification classique et cela a oriente toute l'analyse.
J'ai aussi constate que la performance pouvait beaucoup varier selon les jeux, ce qui renforcait l'idee de multiplier les tests. Sur certains jeux, l'ensemble S&F n'était pas le meilleur en valeur absolue, mais il restait souvent plus stable que les votes simples, ce qui est un aspect important pour un usage réel.

**Mise en perspective.**
Ces trois applications suivent une progression logique : classification pour apprendre les bases et le vote, anomalies sur MNIST pour valider le protocole, puis PyOD/ADBench pour une évaluation plus proche du réel. Ce cheminement m'a permis de comparer les approches de vote dans plusieurs contextes et de situer Sources & Faits dans un cadre plus large.

Sur le plan technique, je suis passé du supervisé à l'anomalie en gardant le vote comme fil conducteur. Sur le plan méthodologique, je suis passé d'expériences simples à des données plus difficiles, ce qui m'a obligé à vérifier la cohérence des résultats et à ajuster mes choix.

Au final, cette progression m'a aidé à distinguer un prototype d'une évaluation plus rigoureuse : tester, vérifier, corriger, puis valider avant de passer à l'étape suivante.

### 2. Choix de conceptions

Le choix de Streamlit s'imposait pour les trois applications : c'est une solution simple, gratuite et rapide pour construire une interface web sans lourdeur technique. Il suffisait de mettre le code sur GitHub, puis sur Streamlit de choisir le fichier correspondant à l'application. Cela permettait de tester plusieurs configurations, d'afficher des tableaux de résultats et d'explorer les modèles sans passer par une application plus lourde.
Ce choix était aussi pratique pour le suivi : je pouvais montrer rapidement l'état d'avancement, envoyer un lien puisqu'il s'agit d'un site web, comparer des résultats, et vérifier visuellement que les sorties étaient cohérentes. Cela a facilité les discussions avec mon tuteur et les ajustements à faire.
J'ai aussi choisi d'utiliser numpy pour avoir des calculs plus scientifiques et plus optimisés, en cohérence avec les pratiques de scikit-learn et PyOD.
Pour les tests, j'ai retenu un découpage 80/20 (train/test) afin de limiter le surapprentissage.
J'ai choisi d'organiser et de centraliser mes fichiers en dossiers en fonction de leur utilité : classes utilitaires et classes principales (Voting, T-SNE et Borda), documentation personnelle, données téléchargées, images des résultats, et enfin le dossier de tests. Cela m'a permis d'organiser mon travail et d'avancer plus rapidement.
La classe **Trouve_params** m'a servi à limiter les réglages manuels. J'ai choisi la validation croisée pour trouver les paramètres, car c'est la méthode la plus efficace pour les vérifier ; elle permettait d'explorer automatiquement des grilles de paramètres, ce qui rendait les comparaisons plus justes et moins dépendantes d'un choix arbitraire.
Pour l'application 3, j'ai dû adapter la logique au fonctionnement de PyOD : j'ai créé un **myVotingPyOD** dédié et une recherche de paramètres spécifique pour gérer des sorties et des contraintes différentes de scikit-learn. Cela m'a permis de garder une comparaison cohérente malgré le non supervisé.
Le classement Borda a été choisi car il permet de comparer les modèles sur plusieurs configurations (taux d'anomalies différents) et pas seulement sur un seul cas. Cette approche donne une vision globale, moins sensible à un cas particulier. Le choix de métriques multiples (Accuracy, Précision, Rappel, F1, ROC-AUC, PR-AUC) permettait d'avoir une lecture plus complète des performances. Enfin, les modèles utilisés ont été choisis à partir de la liste proposée par mon tuteur, ce qui m'a permis de travailler sur un ensemble cohérent et réaliste.
Le choix des métriques a été important car il dépend du type de problème. En classification, Accuracy peut suffire, mais en détection d'anomalies la classe rare est essentielle, donc Précision, Rappel, F1 et PR-AUC deviennent déterminantes. Cette distinction m'a permis d'adapter mes analyses selon le contexte.

Enfin, la décision d'utiliser plusieurs configurations et un classement global allait dans le sens d'une évaluation plus robuste. Cela évitait de tirer des conclusions sur un seul cas particulier et permettait de vérifier la stabilité des modèles d'un scénario à l'autre.

### 3. Travail effectue

Le travail a été realise en plusieurs etapes, en lien direct avec les trois applications.

Pour l'application 1, j'ai integre les modeles de classification, prepare les jeux de données, mis en place les métriques, et developpe la logique de vote hard/soft/S&F. J'ai ensuite cree **MyVotingClassifier** pour regrouper ces regles et **MyTsne** pour la visualisation, puis integre ces classes dans une interface Streamlit.
J'ai aussi dû apprendre matplotlib et numpy pour produire les visualisations et manipuler les données de manière fiable.
J'ai aussi mis en place un affichage clair des résultats afin de comparer rapidement les modeles. Cette presentation m'a oblige a structurer les sorties, a uniformiser les formats et a produire des tableaux lisibles pour une lecture directe.
J'ai également consacré du temps a verifier la reproductibilité : relancer les tests, comparer les scores, et confirmer que les classements restaient proches. Cela m'a donne un cadre plus rigoureux pour les etapes suivantes.

Pour l'application 2, j'ai construit les configurations MNIST avec anomalies, calcule les métriques en train et en test, mis en place le classement Borda, puis ajoute l'affichage t-SNE pour interpréter une configuration a la fois. J'ai aussi gere les echantillons lorsque les données étaient trop volumineuses.
Une partie importante a été de verifier que les configurations étaient comparables entre elles, afin que le classement Borda ait un vrai sens. J'ai donc veille a garder la meme logique d'injection d'anomalies et a controler la proportion de données normales.

Ce travail m'a oblige a controler la qualite des données en entree, a verifier les équilibres, et a m'assurer que les métriques étaient calculees de maniere cohérente entre train et test.

Pour l'application 3, j'ai integre les jeux ADBench, calcule les métriques en train/test, compare les votes hard/soft/S&F, et extrait des tableaux de résultats. Cette partie a été plus longue car elle mobilisait plusieurs modeles PyOD et des données plus lourdes.

J'ai aussi du adapter certaines parties du code pour gerer les contraintes du non supervisé. La comparaison n'est pas toujours directe, ce qui m'a pousse a homogeniser la facon de produire les scores et a stabiliser les résultats.

Enfin, j'ai consolide les résultats sous forme de tableaux pour faciliter l'analyse, ce qui m'a servi directement pour la redaction du rapport et la preparation des annexes.

### 4. Problèmes rencontres, solutions eventuellement apportees

**Comprendre et reproduire t-SNE.** Au debut, les projections n'étaient pas cohérentes. J'ai corrige en testant les paramètres, en verifiant les conditions d'entree et en m'appuyant sur les scores de confiance et de silhouette.

Ce point a pris du temps car t-SNE est sensible aux paramètres et peut produire des résultats trompeurs si les données ne sont pas correctement preparees. J'ai appris a interpréter ces graphes avec prudence, en les considerant comme un outil d'exploration et non comme une preuve definitive.

Cela m'a oblige a tester plusieurs configurations et a comprendre l'effet de chaque paramètre. J'ai aussi appris qu'une visualisation jolie n'est pas forcement une visualisation fiable.

**Hyperparamètres des modeles.** Certains modeles étaient simples a configurer, d'autres demandaient beaucoup de réglages. J'ai procede par tests successifs et ajustements progressifs, parfois en m'aidant de ma classe **Trouve_params**.

Ce travail m'a montre que les performances ne dependent pas seulement de l'algorithme, mais aussi du choix fin des paramètres. Un mauvais réglages peut rendre un modele inefficace, meme si l'algorithme est bon.

Au fil des essais, j'ai appris a reduire la recherche a des plages raisonnables, pour eviter de perdre du temps sur des combinaisons peu utiles.

**Erreurs et mauvaises configurations.** Plusieurs executions ont echoue au debut (paramètres invalides, formats de données). J'ai corrige au fur et a mesure en verifiant chaque etape et en stabilisant les entrees.

J'ai aussi appris a verifier rapidement les erreurs et a isoler leur origine, ce qui m'a permis d'avancer plus efficacement sur le reste du projet.

Cette phase a été importante pour la suite, car elle m'a oblige a rendre le code plus robuste et a ajouter des controles simples pour eviter des erreurs silencieuses.

**Temps de calcul trop long.** Le voting classifier était couteux. J'ai etudie le fonctionnement des modeles scikit-learn et utilise le parallelisme via joblib, ce qui a nettement ameliore le temps d'execution.

Cette optimisation a été importante car elle a rendu les tests iteratifs possibles, et donc la comparaison entre méthodes plus fluide.

Sans cette optimisation, certaines series de tests devenaient trop longues pour être exploitees efficacement dans un stage court.

**Résultats limites sur certains jeux.** Dans l'application 2, le faible nombre de données et de modeles a donne des scores proches entre S&F et hard voting. Dans l'application 3, certains jeux ADBench étaient trop complexes ou lourds, ce qui a degrade les performances.

Cela m'a appris que la méthode ne peut pas être jugee sur un seul jeu de données, et qu'il faut multiplier les cas pour avoir une conclusion solide.

J'ai aussi compris qu'un résultat moyen n'est pas forcement un echec : il peut pointer un manque de données, un problème de configuration, ou une limite de la méthode sur un type d'anomalie precis.

**PyOD et non supervisé.** Les algorithmes PyOD sont non supervisés, ce qui complique l'interpretation et la comparaison. J'ai du comprendre en detail leur fonctionnement et reecrire/ameliore une partie du voting outlier pour l'adapter a mes besoins.

Cette partie a été la plus technique, car la logique de vote doit être adaptee a des scores et non a des labels directs. Cela m'a oblige a reviser les concepts de base et a ajuster le code pour rester cohérent avec les objectifs du stage.

Ce travail m'a aussi montre que les comparaisons doivent être faites avec prudence : les scores de chaque détecteur n'ont pas toujours la meme echelle, et il faut donc harmoniser les sorties pour garder un vote cohérent.

