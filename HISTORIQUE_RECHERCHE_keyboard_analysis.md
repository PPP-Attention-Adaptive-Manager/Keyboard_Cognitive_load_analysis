# Historique scientifique et technique du dépôt

> Dépôt : `Keyboard_Cognitive_load_analysis`  
> Période couverte par Git : du 1er avril 2026 au 1er mai 2026  
> Audit réalisé le 19 septembre 2026 après `git fetch --all --prune`  
> Branche de référence : `main`, commit `945821e`

## 1. Objet de ce document

Ce document reconstruit l'évolution du projet à partir des preuves versionnées : commits, diffs, code, documentation et sorties enregistrées dans le notebook d'expérimentation. Il est destiné à faciliter la rédaction ultérieure d'un article de recherche.

Trois niveaux de preuve sont distingués :

- **fait Git** : directement observable dans l'historique ou les fichiers ;
- **décision explicite** : annoncée par un message de commit, le README ou un commentaire ;
- **interprétation** : intention vraisemblable déduite du code, à confirmer par l'auteur avant publication.

Le dépôt contient 12 commits, tous attribués à `mohannedbt <mohannedbentaleb8@gmail.com>`, aucun tag et aucune release Git. Le premier état contient 8 fichiers et 717 lignes ajoutées. À l'état actuel, 49 fichiers sont suivis. Par rapport au premier commit, Git mesure 6 458 insertions et 42 suppressions, avec détection des renommages.

## 2. Résumé de l'évolution

Le projet a suivi quatre étapes principales :

1. **Prototype historique** : fusion de journaux de frappe, extraction de variables au niveau session, cible synthétique de charge cognitive, Random Forest et interface temps réel.
2. **Architecture V2 modulaire** : séparation preprocessing/features/proxies/model/inference/UI, personnalisation utilisateur et premières briques RL.
3. **Étiquetage teacher–student** : encodeur LSTM étudiant, clustering en trois groupes, score de risque et baselines avec séparation par participant.
4. **Exploration V3 et notebook multimodal** : parquet, représentations contrastives, évaluation d'embeddings et expériences clavier/souris/biosignaux. Les résultats enregistrés sont majoritairement négatifs ou non concluants.

La conclusion scientifique actuelle est donc une **preuve de faisabilité logicielle partielle**, et non une validation robuste de l'inférence de charge cognitive à partir du clavier.

## 3. Chronologie exhaustive des commits

### 1 — `6167560` — 1er avril 2026

**Message :** `first version of the model next work is memory`  
**Volume :** 8 fichiers, 717 insertions.

Faits et changements :

- création du pipeline initial : `Data-merging.py`, `Data_processing.py`, `Model_training.py`, `testing_model.py` et `data_understanding.py` ;
- ajout d'un README, d'un `.gitignore` et des dépendances ;
- agrégation des événements de frappe en métriques de session ;
- modèle Random Forest entraîné sur une cible proxy calculée à partir de variables comportementales ;
- première interface temps réel.

Décision explicite : le prochain travail devait porter sur la « mémoire ».  
Interprétation : la personnalisation temporelle et la conservation d'un profil utilisateur étaient déjà considérées comme nécessaires.

### 2 — `fe10b44` — 2 avril 2026

**Message :** `Detailed Readme`  
**Volume :** README, 402 insertions et 42 suppressions.

Faits et changements : documentation détaillée du pipeline, des variables, de la cible proxy et de l'utilisation des scripts. Aucun changement de code.

Décision : formaliser l'approche avant une nouvelle évolution architecturale.

### 3 — `d3e412e` — 3 avril 2026

**Message :** `Revise cognitive load formula in Readme`  
**Volume :** 2 insertions et 2 suppressions dans le README.

Fait important : le contenu mathématique visible de la formule ne change pas ; le diff remplace surtout un délimiteur LaTeX `$$...$$` par `$...$`. Le message de commit surestime donc la modification scientifique réelle.

Forme documentée du proxy à ce stade : combinaison non linéaire du temps de maintien, des pauses, du maximum de maintien, des pauses par minute et d'autres mesures, annoncée sur une échelle 0–10. La formule complète est implémentée dans le prototype historique.

### 4 — `96379f4` — 14 avril 2026, 18:46

**Message :** `initialize cognitive system architecture`  
**Volume :** 18 fichiers, 898 insertions et 628 suppressions.

Faits et changements :

- déplacement du prototype dans `legacy/` ;
- ajout de `legacy/dqn_agent.py` et `legacy/user_profile.py` ;
- introduction d'un agent DQN avec mémoire de rejeu ;
- profil utilisateur persistant avec moyennes/variances et normalisation ;
- modification de l'interface historique vers un système combinant modèle, RL et profil utilisateur.

Décision explicite : initialiser une architecture de « système cognitif ».  
Interprétation : passage d'un prédicteur statique à un système adaptatif mêlant prédiction, mémoire utilisateur et décision.

### 5 — `f4b3f9f` — 14 avril 2026, 22:04 — branche `cognitive-upgrade`

**Message :** `Added V2 and legacy V2 contains the new version but RL not tested`  
**Volume :** 23 fichiers, 1 094 insertions.

Faits et changements :

- création de la structure V2 : `features`, `preprocessing`, `proxies`, `training`, `models`, `inference`, `personalization`, `rl` et `ui` ;
- pipeline mis en cache pour données temporelles, features de session et proxies ;
- modèle comportemental, inférence cognitive et tableau de bord ;
- environnement et politique RL minimaux ;
- déplacement du prototype précédent dans `legacy/`.

Décision explicite : V2 devient la nouvelle version et l'ancien système est conservé comme référence.  
Limite explicitement reconnue : la partie RL n'est pas testée.

### 6 — `8aef3f3` — 14 avril 2026, 22:07 — branche principale de l'époque

**Message :** `added legacy folder in main`  
**Volume :** 7 renommages sans modification de contenu.

Fait : les sept fichiers du prototype racine sont déplacés dans `legacy/`. Cette préparation parallèle facilite la fusion de V2.

### 7 — `9868047` — 14 avril 2026, 22:11

**Message :** `Merge branch 'cognitive-upgrade'`  
**Parents :** `8aef3f3` et `f4b3f9f`.

Fait : intégration de l'architecture V2 dans la branche principale. La branche historique distante `cognitive-upgrade` pointe encore sur son dernier commit pré-fusion.

### 8 — `c73f0d6` — 14 avril 2026, 22:14

**Message :** `done the cache`  
**Volume :** suppression de 10 fichiers binaires de cache Python/Numba.

Décision : nettoyer les artefacts `__pycache__` et les caches compilés.  
Résultat actuel : cette règle n'a pas été maintenue ; plusieurs `.pyc` sont de nouveau suivis dans des commits ultérieurs.

### 9 — `024cab2` — 15 avril 2026, 19:53

**Message :** `added the interface and everything is ok`  
**Volume :** 13 fichiers, 319 insertions et 227 suppressions.

Faits et changements :

- ajout/amélioration du tableau de bord Tkinter ;
- ajout du constructeur de variables temps réel ;
- remplacement de `behavior_model.py` par `cognitive_model.py` ;
- ajout d'un modèle avec fenêtre glissante, normalisation personnalisée et sorties bornées dans `[0,1]` ;
- adaptation du profil utilisateur ;
- ajout de caches binaires au suivi Git.

Décision interprétée : privilégier un chemin d'inférence temps réel intégré plutôt qu'un simple modèle isolé.  
Attention : « everything is ok » n'est accompagné d'aucun test ou résultat versionné permettant de valider cette affirmation.

### 10 — `ea29b1e` — 15 avril 2026, 19:58

**Message :** `updated a README`  
**Volume :** 245 insertions et 305 suppressions.

Fait : mise à jour substantielle de la documentation pour refléter V2, les chemins d'exécution et les composants.

### 11 — `2e7f84c` — 26 avril 2026

**Message :** `added student labeling from teacher code in collab`  
**Volume :** 27 fichiers, 1 638 insertions et 408 suppressions.

Faits et changements :

- ajout d'un encodeur étudiant LSTM bidirectionnel : 7 variables d'entrée, deux couches, état caché 128 et projection 64 dimensions ;
- étiquetage par fenêtres de 30 événements ;
- `MiniBatchKMeans` à 3 clusters ;
- création de `cognitive_label` (identifiant de cluster) et `cognitive_risk` (distance normalisée au centroïde) ;
- traitement par chunks de 100 000 lignes et limite de 500 participants pour contenir la mémoire ;
- baselines de classification : dummy, régression logistique et HistGradientBoosting ;
- baselines de régression : dummy, Ridge et HistGradientBoosting ;
- séparation train/test par participant avec `GroupShuffleSplit`, ce qui réduit le risque de fuite inter-utilisateur ;
- ajout d'artefacts entraînés (`.pth`, `.pt`, `.joblib`) et d'un prototype d'analyse temps réel ;
- enrichissement des features : erreur de Levenshtein, KSPC, mots, digraphes, entropie, corrections et indice d'effort.

Décisions : employer une distillation teacher–student afin d'exploiter le clavier seul à l'inférence ; utiliser un clustering non supervisé pour produire des pseudo-étiquettes ; traiter les gros volumes en streaming ; évaluer par groupes d'utilisateurs.

Limite scientifique : les numéros de cluster ne constituent pas, à eux seuls, des classes sémantiques validées de type concentration/fatigue/surcharge.

### 12 — `945821e` — 1er mai 2026 — état actuel de `main`

**Message :** `removed modelsmore`  
**Volume :** 15 fichiers, 3 816 insertions et 330 suppressions.

Faits et changements :

- suppression des six artefacts de modèles entraînés suivis au commit précédent ;
- ajout du notebook `Experimentation/Cognitive_Load_Pipeline (1).ipynb` avec sorties expérimentales ;
- ajout de V3 : phase 1 de construction de séquences/parquet, phase 2 contrastive et évaluation des embeddings ;
- ajout d'un lanceur racine pour les baselines V2 ;
- correction de chemins afin de cibler explicitement `V2/data` et `V2/models` ;
- README réécrit pour présenter V2 comme système courant et V3 comme expérimental.

Décision interprétée : ne plus versionner les poids entraînés et ouvrir une voie expérimentale fondée sur les embeddings. Le message du commit ne rend toutefois pas compte de l'ajout majeur de V3 et du notebook.

## 4. Évolution des choix méthodologiques

### Données et unité d'analyse

- Le prototype historique agrège les frappes au niveau `(participant, section de test)`.
- V2 conserve une voie « session features » et ajoute une voie séquentielle de 30 événements.
- V3 phase 1 génère des séquences et des variables agrégées dans des lots parquet ; phase 2 apprend une représentation par objectif contrastif.
- Le notebook multimodal versionné utilise 9 648 lignes, 104 colonnes et seulement 5 utilisateurs. La séparation enregistrée contient 3 utilisateurs d'entraînement (6 560 lignes) et 2 utilisateurs de test (3 088 lignes).

### Variables

Les familles de variables introduites au cours du projet comprennent : IKI/IKT, temps de maintien, pauses, vitesse, burstiness, KSPC, erreurs de Levenshtein, backspaces, caractéristiques de mots et digraphes, entropie des touches et densité d'interaction. Le notebook ajoute des variables souris et des biosignaux.

### Cibles

Trois stratégies différentes coexistent et ne doivent pas être confondues dans un article :

1. **Proxy déterministe historique/V2** calculé à partir des variables de frappe ;
2. **Pseudo-étiquettes teacher–student** obtenues par embedding puis clustering ;
3. **Cible biosignal du notebook**, combinaison pondérée annoncée de `1/RMSSD` (30 %), HR (20 %), réactivité EDA (20 %), ratio EEG beta/alpha (15 %), énergie ACC (10 %) et variabilité respiratoire (5 %), normalisée par utilisateur.

Le premier cas mesure surtout la capacité d'un modèle à reproduire une formule construite à partir de ses entrées. Il ne valide pas directement une charge cognitive réelle. Le second produit des groupes latents qui nécessitent une validation externe avant de recevoir des noms psychologiques. Le troisième est plus proche d'une référence physiologique, mais reste une cible composite construite et testée sur cinq utilisateurs.

### Modèles

- Random Forest dans le prototype ;
- MLP PyTorch `BehaviorNet` dans V2 ;
- baselines sklearn linéaires, boosting et dummy ;
- LSTM bidirectionnel pour l'étudiant ;
- RNN, GRU, LSTM, BiGRU, TCN, attention et modèles contrastifs dans le notebook/V3 ;
- DQN et politique adaptative présents comme prototypes, sans validation versionnée.

### Évaluation

L'amélioration méthodologique la plus nette est l'introduction d'une séparation par participant dans V2 et d'un découpage par utilisateurs dans le notebook. Cela vise une généralisation inter-sujets. En revanche, aucun protocole complet répété, intervalle de confiance, test statistique, étude d'ablation ou graine multiple n'est enregistré.

## 5. Résultats expérimentaux effectivement versionnés

Les seuls résultats chiffrés persistants trouvés sont les sorties du notebook ajouté au dernier commit. Ils doivent être rapportés comme **résultats exploratoires d'une exécution**, pas comme résultats définitifs.

### Régression de la cible biosignal

| Modalité / modèle | R² test enregistré | Lecture |
|---|---:|---|
| Clavier — Random Forest | 0,0175 | Très faible, à peine au-dessus de la moyenne |
| Souris — Random Forest | -0,0484 | Moins bon que le prédicteur moyen |
| Souris — TCN | -0,1370 | Négatif |
| Alignement multimodal latent + RF | -0,2339 | Négatif |
| Souris — LSTM | -0,4066 | Négatif |
| Souris — MLP | -0,4656 | Négatif |
| Souris — Attention | -0,5041 | Négatif |
| Souris — BiGRU | -0,6770 | Négatif |
| Souris — GRU | -0,7417 | Négatif |
| Modèles profonds clavier | -1,0000 de repli | Échec d'évaluation à cause de NaN, pas un score valide |

Les sept modèles profonds clavier (`MLP`, `RNN`, `GRU`, `LSTM`, `BiGRU`, `TCN`, `Attn`) ont rencontré `Input contains NaN`; le notebook force ensuite un score de repli de `-1`. Ces valeurs ne doivent pas être comparées comme de véritables performances.

### Associations univariées avec la cible

Les corrélations absolues maximales rapportées sont faibles : `inactivity_ratio` 0,0875, `typing_error_proxy` 0,0630, `burstiness` 0,0548, `speed_trend` 0,0528 et `cognitive_load_proxy` 0,0524. Cela suggère peu de signal linéaire individuel dans cette exécution.

### Représentation contrastive et clustering

- perte contrastive : 9,1196 à l'époque 0, 6,2805 à l'époque 45 ;
- score d'alignement latent cosinus : 0,1420 ;
- R² de régression sur l'espace latent : -0,2339 ;
- silhouette du clustering : 0,2866 ;
- répartition affichée : cluster 0 « Focus » 29,86 %, cluster 1 « Fatigue » 36,57 %, cluster 2 « Overload » 33,57 %.

La baisse de perte indique une optimisation effective de l'objectif d'entraînement, mais le faible alignement et le R² négatif ne démontrent pas une représentation prédictive de la cible. Les noms des clusters apparaissent assignés dans le notebook ; aucune validation externe de leur sémantique n'est versionnée.

### Teacher–student

- teacher : perte de 12,4840 à 0,2632 sur 20 époques ; embeddings de forme `(9608, 64)` ;
- student : perte cosinus de 14,0075 à 11,8008 sur 30 époques ; modèle sauvegardé pendant l'expérience.

La convergence du teacher est nette sur sa fonction de perte. La perte student baisse peu après la première époque et reste élevée ; aucune métrique de fidélité teacher–student sur un jeu tenu à part n'est enregistrée. Les poids ont ensuite été supprimés du dépôt.

### Résultats absents

Aucun résultat chiffré persistant n'a été trouvé pour :

- le Random Forest historique sur le grand jeu de frappes ;
- `BehaviorNet` V2 ;
- les baselines V2 de classification/régression sur les pseudo-étiquettes ;
- le tableau de bord temps réel ;
- le DQN ou la politique RL ;
- le pipeline V3 autonome.

Le code sait calculer certaines métriques, mais la présence du code ne prouve pas qu'une expérience a été exécutée ni validée.

## 6. Branches vérifiées

Après synchronisation avec `origin` le 19 septembre 2026 :

| Référence | Commit | Relation avec `main` | État |
|---|---|---|---|
| locale `main` | `945821e` | identique à `origin/main` | branche locale active |
| distante `origin/main` | `945821e` | référence | branche principale |
| distante `origin/RedoV3` | `945821e` | 0 commit d'écart | alias exact de `main` actuellement |
| distante `origin/cognitive-upgrade` | `f4b3f9f` | 7 commits derrière, 0 devant | branche V2 historique déjà fusionnée |

Il n'existe pas de branche locale autre que `main`. `origin/HEAD` est un pointeur symbolique vers `origin/main`, pas une branche supplémentaire. Aucun tag n'est présent.

Conséquence pratique : il n'y a aujourd'hui aucun travail unique à récupérer depuis `RedoV3` ou `cognitive-upgrade`. Leur suppression éventuelle serait une décision de maintenance distincte ; aucune branche n'a été supprimée pendant cet audit.

## 7. Limites et risques pour un article scientifique

### Validité de construit

- Les proxies de charge sont en partie dérivés des mêmes variables que celles données au modèle : risque de circularité.
- Les clusters sont des groupes géométriques, pas automatiquement des états cognitifs.
- La cible biosignal est une construction pondérée ; ses poids doivent être justifiés par la littérature ou appris/validés.
- Les concepts `CogLoad`, `Stress` et `Unfocus` doivent être définis opérationnellement et distingués.

### Validité interne

- Seulement cinq utilisateurs dans le notebook, dont deux au test.
- Une seule séparation et apparemment une seule graine ; forte variance probable.
- Des NaN invalident toutes les expériences profondes clavier enregistrées.
- Absence d'intervalles de confiance, de tests statistiques et d'ablations.
- Les sorties d'entraînement ne suffisent pas à mesurer la généralisation du teacher et du student.

### Reproductibilité

- données brutes, datasets transformés et poids de modèles absents de l'état actuel ;
- chemins absolus Windows dans V3 et chemins relatifs hétérogènes dans d'autres scripts ;
- pas de fichier d'environnement complet à la racine ni de verrouillage des versions ;
- aucun test automatisé ni intégration continue ;
- caches Python/Numba encore suivis malgré un commit antérieur de nettoyage ;
- notebook contenant des sorties mais pas de manifeste complet de l'environnement, du matériel et des graines ;
- quelques incohérences de nommage (`Phase_1` contre `phase_2`, V2/V3, variables IKI/IKT).

### Traçabilité

- certains messages de commit sont trop vagues ou ne décrivent pas l'ampleur réelle des changements ;
- aucune issue, décision architecturale formelle, release ou tag d'expérience n'est conservé dans Git ;
- les résultats de V2 n'ont pas été exportés sous une forme tabulaire versionnée.

## 8. Décisions à confirmer avec les auteurs avant rédaction

Les points suivants sont des questions ouvertes que Git ne permet pas de trancher :

1. Quel jeu de données exact a alimenté le prototype, V2 et le notebook, et sous quelle licence ?
2. Les cinq utilisateurs du notebook constituent-ils un sous-échantillon de débogage ou l'échantillon final ?
3. Comment les poids de la cible biosignal ont-ils été choisis ?
4. Quelle vérité terrain externe permet d'appeler les clusters « Focus », « Fatigue » et « Overload » ?
5. Quel teacher a produit `student_model.pt`, sur quelles modalités et avec quel protocole anti-fuite ?
6. Pourquoi la V3 a-t-elle été créée : performance, mémoire, généralisation, ou séparation multimodale ?
7. Quelles expériences V2 ont réellement été exécutées et où se trouvent leurs journaux ?
8. La composante RL est-elle abandonnée, différée ou encore incluse dans l'hypothèse scientifique ?

## 9. Proposition de récit pour le futur papier

Un récit fidèle aux preuves actuelles pourrait être :

> Nous avons étudié la faisabilité d'estimer des indicateurs de charge cognitive à partir de dynamiques d'interaction. Un premier pipeline fondé sur des variables agrégées et une cible heuristique a servi de preuve de concept. Nous l'avons ensuite modularisé pour permettre l'inférence temps réel et la personnalisation, puis avons exploré une distillation multimodale teacher–student et des représentations contrastives. Sur un petit échantillon inter-sujets de cinq participants, les baselines clavier et souris prédisent très faiblement la cible physiologique composite, tandis que plusieurs modèles clavier n'ont pas pu être évalués à cause de valeurs manquantes. Ces résultats négatifs motivent un protocole plus large, une définition indépendante de la vérité terrain et une validation répétée par participant.

Ce récit met en valeur l'apport d'ingénierie et l'apprentissage tiré des résultats négatifs sans revendiquer une performance non démontrée.

## 10. Expériences prioritaires avant publication

1. Figer une version nettoyée des données et documenter participants, tâches, capteurs, exclusions et synchronisation.
2. Choisir une cible principale indépendante des variables clavier ; conserver les proxies comme analyses auxiliaires.
3. Corriger les NaN puis exécuter une validation `GroupKFold`/leave-one-subject-out avec plusieurs graines.
4. Comparer systématiquement moyenne, modèles linéaires, Random Forest/boosting et modèles séquentiels.
5. Rapporter MAE, RMSE, R², corrélation, intervalles de confiance et résultats par participant.
6. Ajouter des ablations par famille de variables et par modalité.
7. Valider les clusters contre des questionnaires, conditions expérimentales ou mesures physiologiques indépendantes avant de les nommer.
8. Évaluer la distillation avec similarité teacher–student, performance sur cible tenue à part et généralisation à des utilisateurs inconnus.
9. Mesurer latence, mémoire et stabilité pour soutenir les affirmations temps réel.
10. Versionner configuration, graines, métriques CSV/JSON et figures sous un identifiant d'expérience ; créer ensuite un tag Git pour chaque résultat publié.

## 11. Éléments réutilisables dans les sections d'un papier

- **Introduction** : intérêt d'une estimation non intrusive à partir de la frappe et difficulté de la généralisation inter-utilisateur.
- **Méthodes** : segmentation en séquences, familles de variables temporelles/erreurs, séparation par utilisateur, baselines, distillation et apprentissage contrastif.
- **Résultats** : tableau des R² ci-dessus, corrélations faibles, résultats contrastifs et échecs dus aux NaN.
- **Discussion** : faible signal comportemental sur petit effectif, limites des proxies, besoin de vérité terrain indépendante et valeur informative des résultats négatifs.
- **Reproductibilité** : environnement à reconstruire, données et modèles à archiver, chemins à paramétrer et expériences à identifier.
- **Travaux futurs** : cohorte plus large, validation croisée inter-sujets, calibration/personnalisation, multimodalité et seulement ensuite adaptation RL.

## 12. Commandes d'audit utilisées

Les observations Git peuvent être reproduites avec :

```bash
git fetch --all --prune
git log --all --reverse --date=iso-strict --stat
git log --all --graph --decorate --oneline
git branch --all --verbose --no-abbrev
git rev-list --left-right --count origin/main...origin/RedoV3
git rev-list --left-right --count origin/main...origin/cognitive-upgrade
git diff --stat 6167560..HEAD
git tag --list
```

Les valeurs expérimentales ont été relevées dans les sorties enregistrées du notebook `Experimentation/Cognitive_Load_Pipeline (1).ipynb`. Toute réexécution peut produire d'autres valeurs ; les futurs résultats devraient donc être associés à une configuration, une graine et un commit précis.
