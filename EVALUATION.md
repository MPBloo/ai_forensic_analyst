# Évaluation de la pipeline de catégorisation

Ce document décrit la méthodologie d'évaluation de `classify_image_by_category`
(app.py), les limites reconnues de cette évaluation, et les résultats obtenus.

## 1. Objectif

Mesurer, de façon reproductible, la qualité de la catégorisation d'images de
l'application — en particulier sur `weapons`, la catégorie à plus fort coût
de faux positif dans un contexte judiciaire — et isoler l'effet de **deux
axes de design indépendants**, mesurés dans le même script plutôt que
mélangés :

- **questions** : `closed` (l'approche pré-refactor, des questions fermées
  type yes/no posées par catégorie, jusqu'à 2 par catégorie sur 10
  catégories) vs `open` (l'approche actuelle, 3 questions ouvertes communes
  posées une seule fois par image).
- **matching** : `lexical` (comptage de mots-clés pondéré, le mécanisme
  utilisé par `app.py` avant son passage au cosinus) vs `semantic`
  (similarité cosinus MiniLM entre le texte de travail et une phrase
  descriptive par catégorie, le mécanisme actuel de
  `app.categories_from_text`).

`scripts/evaluate.py` exécute les 4 combinaisons (`closed_lexical`,
`closed_semantic`, `open_lexical`, `open_semantic`), toutes sur le **même
modèle BLIP-2** (4-bit) : `closed_lexical` est le système originel,
`open_semantic` est le système actuel de `app.py`, et les deux autres
cellules isolent chaque variable séparément (effet des questions seul à
matching fixé, effet du matching seul à questions fixées).

## 2. Jeu de données

Un sous-ensemble de 200 images de **COCO val2017** (`scripts/build_eval_set.py`),
choisi pour sa disponibilité, ses annotations multi-label déjà faites, et sa
grande diversité de scènes.

### 2.1 Mapping COCO → CATEGORIES_POLICE

| Catégorie COCO | → CATEGORIES_POLICE |
|---|---|
| person | people |
| bicycle, car, motorcycle, airplane, bus, train, truck, boat | vehicles |
| knife, scissors | weapons |
| book | documents |
| bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe | animals |
| backpack, handbag, suitcase, bottle | objects (sous-ensemble restreint, voir 3.3) |
| *(tout le reste : nourriture, meubles, électronique, sport...)* | non mappé, ignoré |

`buildings`, `advertising`, `indoor`, `outdoor` n'ont **aucune classe COCO
correspondante** et sont exclues de l'évaluation (voir 3.2).

### 2.2 Échantillonnage stratifié

knife/scissors sont rares dans COCO (classes "kitchen"/"indoor" peu
fréquentes) : un tirage purement aléatoire de 200 images donnerait ~0
exemple positif pour `weapons`, rendant precision/recall non significatifs
sur la catégorie prioritaire de ce projet. `build_eval_set.py` garantit donc
un minimum de 18 images contenant knife/scissors (`--n-weapons-min`), le
reste tiré aléatoirement.

**Conséquence assumée** : l'échantillon n'est pas représentatif de la
distribution naturelle de COCO. C'est un choix délibéré pour avoir un
signal exploitable sur `weapons` plutôt qu'un chiffre non significatif basé
sur 0-2 exemples.

## 3. Limites reconnues

### 3.1 COCO n'est pas un corpus d'enquête

Ce sont des photos grand public (Flickr), pas des scènes de crime ni du
matériel de renseignement. Les résultats mesurent la capacité du modèle à
reconnaître des objets et scènes courantes, pas sa pertinence opérationnelle
réelle pour une enquête. C'est un proxy, pas une validation métier.

### 3.2 Couverture partielle des catégories

- **`weapons` — la limite la plus importante** : COCO n'a **aucune classe
  arme à feu** (pas de "gun", "pistol", "rifle"). Le mapping ne couvre que
  des objets tranchants (knife, scissors). Cette évaluation ne dit **rien**
  sur la capacité de l'app à détecter une arme à feu.
- `documents` : seul `book` est mappé. Les vrais documents d'enquête
  (papiers, panneaux, notes manuscrites) n'existent quasiment pas dans
  COCO — signal très partiel.
- `buildings`, `advertising`, `indoor`, `outdoor` : aucune classe COCO
  correspondante, exclues de l'évaluation. Pour `indoor`/`outdoor` en
  particulier, COCO a bien des supercatégories nommées ainsi, mais elles
  désignent des *types d'objets* (indoor = livre, ciseaux, vase, ours en
  peluche... ; outdoor = feu tricolore, panneau stop, borne incendie, banc),
  pas le cadre de la photo — les mapper aurait été un faux raccourci
  sémantique, donc elles ne sont pas mappées du tout.

### 3.3 `objects` restreint volontairement

COCO a ~50 classes qui pourraient techniquement devenir "objects" (nourriture,
meubles, électronique...), ce qui rendrait la vérité terrain quasi-universelle
et le score peu informatif. Seuls backpack/handbag/suitcase/bottle sont
mappés (objets suspects plausibles en contexte d'enquête), au prix d'un
rappel structurellement bas sur cette catégorie.

### 3.4 Taille de l'échantillon

200 images (dont un sous-ensemble stratifié pour `weapons`) donnent des
intervalles de confiance larges, en particulier sur les catégories peu
représentées. Les pourcentages doivent être lus comme des ordres de
grandeur, pas des mesures de précision.

### 3.5 Métrique multi-label

Une matrice de confusion classique (une case par paire classe prédite ×
classe réelle) n'a pas de sens direct en multi-label. `evaluate.py` rapporte
donc TP/FP/FN/TN **par catégorie**, l'équivalent standard et interprétable
en multi-label.

### 3.6 Les configs `closed_*` et `*_lexical` sont des reconstructions figées

`CLOSED_CATEGORY_QUESTIONS` et `LEXICAL_CATEGORY_KEYWORDS`/`LEXICAL_THRESHOLDS`
(dans `evaluate.py`) reconstruisent fidèlement l'esprit des mécanismes
pré-refactor, mais avec un nombre de questions réduit (2 au lieu de jusqu'à
4 par catégorie) pour rester lisibles, et sont **figés indépendamment** de
l'évolution de `app.py` : si `app.py` change encore ses seuils ou son
mécanisme, ces constantes ne suivent pas automatiquement. C'est voulu — le
but est de comparer des points de référence stables, pas de traquer
`app.py` en continu.

### 3.7 La cellule `closed_semantic` n'a pas d'équivalent naturel

Appliquer un cosinus sémantique à des réponses fermées type "yes"/"no" est
un cas de figure qui n'a jamais existé dans une version réelle du projet :
une réponse "yes" ou "no" isolée porte peu de signal sémantique par
embedding (contrairement à un mot-clé exact comme "yes" qui, lui, matche
littéralement). Cette cellule est incluse pour compléter le carré 2×2 et
objectiver ce constat plutôt que l'affirmer sans preuve, mais un score
dégradé sur `closed_semantic` n'est pas une surprise à traiter comme un bug.

## 4. Méthodologie d'exécution

```bash
# 1. Construire le jeu d'évaluation (télécharge annotations + images COCO)
python scripts/build_eval_set.py --n-images 200 --n-weapons-min 18

# 2. Lancer l'évaluation comparative (les 4 configurations)
python scripts/evaluate.py --config all --output-markdown eval_results.md

# 3. Calibrer les seuils sémantiques (F1 max, recall sous contrainte pour weapons)
python scripts/evaluate.py --config open_semantic --calibrate-thresholds
```

`evaluate.py` traite les images une par une (pas de batching) : l'objectif
ici est de comparer le **nombre d'appels VQA** par approche (3 pour les
configs `open_*`, ~15-18 pour les configs `closed_*` sur 10 catégories),
pas de redémontrer le gain de débit du batching (voir `scripts/benchmark.py`
pour ça).

## 5. Calibration des seuils sémantiques

`CATEGORY_COSINE_THRESHOLDS` (dans `app.py`) contrôle, par catégorie, le
cosinus minimal pour assigner une catégorie. Les valeurs actuelles sont des
**estimations non calibrées** (placeholders dans la plage réaliste des
cosinus MiniLM, ~0.2-0.6 — voir `app.py`, ces valeurs sont commentées comme
telles). `evaluate.py --calibrate-thresholds` :

1. Exécute la config `*_semantic` demandée une seule fois (un passage
   BLIP-2 + MiniLM par image), en gardant le cosinus **brut** par catégorie
   pour chaque image (avant tout seuillage).
2. Balaie ensuite, gratuitement (pur calcul sur les scores déjà obtenus,
   pas de nouvel appel modèle), les seuils candidats de 0.20 à 0.60 par pas
   de 0.01.
3. Retient, par catégorie, le seuil qui **maximise le F1** — sauf
   `weapons`, où c'est le seuil qui **maximise le recall sous une
   contrainte de precision minimale** (`--weapons-precision-floor`, défaut
   0.50) : un faux négatif (arme non détectée) coûte plus cher qu'un faux
   positif (fausse alerte, quelques secondes de vérification) dans un
   contexte judiciaire — l'asymétrie doit se lire dans le critère
   d'optimisation, pas seulement dans le discours.

Si aucun seuil ne respecte la contrainte de precision sur l'échantillon
fourni, le script le signale explicitement (`N/A`) plutôt que de renvoyer
une valeur qui ne respecte pas la contrainte demandée.

**Cette calibration est spécifique à l'échantillon COCO utilisé** (dont le
sur-échantillonnage de `weapons`, voir 2.2, le rend déjà non représentatif).
Les seuils obtenus sont un point de départ raisonnable, pas une valeur à
transférer telle quelle sur un corpus d'enquête réel — ils devraient être
recalibrés dès que des données réelles (même en petit nombre) sont
disponibles.

## 6. Résultats

**Non exécuté** : ce travail a été fait dans un environnement sans GPU et
sans accès réseau à `cocodataset.org`/`huggingface.co` (bloqués par la
politique réseau du sandbox), donc ni le téléchargement de COCO ni
l'inférence BLIP-2/MiniLM n'ont pu être testés en conditions réelles ici.

Toute la logique (mapping, échantillonnage stratifié, calcul precision/
recall/F1, gestion des cas N/A pour éviter les divisions par zéro, focus
weapons, génération du tableau comparatif, balayage de seuils et gestion du
cas où la contrainte de precision est inatteignable) a été vérifiée avec
des scripts qui mockent BLIP-2/MiniLM et un faux jeu de données COCO — voir
les tests utilisés pendant le développement (non inclus dans le repo,
exécutés en session). Elle n'a **pas** été validée avec le vrai modèle ni
sur GPU.

**À faire avant de présenter ce travail** : exécuter les trois commandes
ci-dessus sur une machine avec GPU (~4 Go VRAM) et accès internet, coller
ici le tableau comparatif 4 configurations (`eval_results.md`), les cas de
faux positifs/négatifs sur `weapons` affichés par `--config all`, et les
seuils calibrés — puis reporter ces seuils dans `CATEGORY_COSINE_THRESHOLDS`
(`app.py`) si les résultats sont jugés fiables.

<!-- Coller ici le tableau comparatif markdown généré par evaluate.py -->
