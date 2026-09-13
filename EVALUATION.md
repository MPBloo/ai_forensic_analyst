# Évaluation de la pipeline de catégorisation

Ce document décrit la méthodologie d'évaluation de `classify_image_by_category`
(app.py), les limites reconnues de cette évaluation, et les résultats obtenus.

## 1. Objectif

Mesurer, de façon reproductible, la qualité de la catégorisation d'images de
l'application — en particulier sur `weapons`, la catégorie à plus fort coût
de faux positif dans un contexte judiciaire — et comparer deux approches de
prompting VQA :

- **vqa_closed** : l'ancienne approche (avant refactor), des questions
  fermées type yes/no posées par catégorie (jusqu'à 4 questions ×
  10 catégories).
- **vqa_open** : l'approche actuelle, 3 questions ouvertes communes posées
  une seule fois par image, catégories déduites par mots-clés dans les
  réponses.

Les deux configurations tournent sur le **même modèle BLIP-2** (4-bit) :
la comparaison isole l'effet du style de questions, pas celui du modèle.

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

### 3.6 `vqa_closed` est une reconstruction, pas l'implémentation d'origine

`classify_vqa_closed` (dans `evaluate.py`) reconstruit fidèlement l'esprit de
l'ancienne approche (questions fermées par catégorie, scoring yes/no) mais
avec un nombre de questions réduit (2 au lieu de jusqu'à 4 par catégorie)
pour rester lisible. Le point de comparaison est le **style** de question
(fermé vs ouvert), pas une reproduction bit-à-bit de l'ancien code.

## 4. Méthodologie d'exécution

```bash
# 1. Construire le jeu d'évaluation (télécharge annotations + images COCO)
python scripts/build_eval_set.py --n-images 200 --n-weapons-min 18

# 2. Lancer l'évaluation comparative
python scripts/evaluate.py --config both --output-markdown eval_results.md
```

`evaluate.py` traite les images une par une (pas de batching) : l'objectif
ici est de comparer le **nombre d'appels VQA** par approche (3 pour
vqa_open, ~10-12 pour vqa_closed sur 6 catégories), pas de redémontrer le
gain de débit du batching (voir `scripts/benchmark.py` pour ça).

## 5. Résultats

**Non exécuté** : ce travail a été fait dans un environnement sans GPU et
sans accès réseau à `cocodataset.org`/`huggingface.co` (bloqués par la
politique réseau du sandbox), donc ni le téléchargement de COCO ni
l'inférence BLIP-2 n'ont pu être testés en conditions réelles ici.

Toute la logique (mapping, échantillonnage stratifié, calcul precision/
recall/F1, gestion des cas N/A pour éviter les divisions par zéro, focus
weapons, génération du tableau comparatif) a été vérifiée avec des scripts
qui mockent BLIP-2 et un faux jeu de données COCO — voir les tests
utilisés pendant le développement (non inclus dans le repo, exécutés en
session). Elle n'a **pas** été validée avec le vrai modèle ni sur GPU.

**À faire avant de présenter ce travail** : exécuter les deux commandes
ci-dessus sur une machine avec GPU (~4 Go VRAM) et accès internet, puis
coller ici le tableau généré (`eval_results.md`) et les cas de faux
positifs/négatifs sur `weapons` affichés par `--config both`.

<!-- Coller ici le tableau comparatif markdown généré par evaluate.py -->
