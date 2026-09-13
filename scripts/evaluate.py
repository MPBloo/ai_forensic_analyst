"""
Évalue la pipeline de catégorisation sur le set COCO construit par
build_eval_set.py.

Deux axes de comparaison, isolés indépendamment dans le même script :
- questions : "closed" (anciennes questions fermées type yes/no, une à deux
  par catégorie) vs "open" (3 questions ouvertes communes, l'approche
  actuelle de app.py).
- matching  : "lexical" (comptage de mots-clés pondéré, l'ancien mécanisme
  de app.py avant son passage au cosinus) vs "semantic" (similarité cosinus
  MiniLM contre une phrase par catégorie, le mécanisme actuel de
  app.categories_from_text).

4 configurations, chacune isolant une combinaison précise :
    closed_lexical  : le système originel (avant tout refactor)
    open_lexical    : effet des questions ouvertes seules (matching figé)
    closed_semantic : effet du cosinus seul sur des réponses fermées
    open_semantic   : le système actuel de app.py

Calibration des seuils sémantiques : --calibrate-thresholds balaie les
seuils de 0.20 à 0.60 (pas 0.01) sur les scores cosinus déjà calculés (un
seul passage BLIP-2 par image, le balayage ensuite est gratuit — CPU,
millisecondes). Retient le F1 maximal par catégorie, sauf "weapons" où
c'est le recall maximal sous une contrainte de precision minimale
(--weapons-precision-floor, défaut 0.50) : un faux négatif coûte plus cher
qu'un faux positif dans ce contexte.

Usage:
    python scripts/evaluate.py --config all
    python scripts/evaluate.py --config open_semantic --calibrate-thresholds
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

import app

# ============================================================================
# Axe "questions" : closed (reconstruction figée de l'approche pré-refactor,
# une à deux questions fermées par catégorie) vs open (app.GENERAL_VQA_QUESTIONS)
# ============================================================================

CLOSED_CATEGORY_QUESTIONS = {
    "people": [
        "Are there any people, persons, or human beings visible in this image?",
        "Can you see a man, woman, or child in this picture?",
    ],
    "vehicles": [
        "Can you see any vehicles, cars, or means of transportation?",
        "Is there a car, truck, motorcycle, or bicycle in this image?",
    ],
    "weapons": [
        "Is there a knife, blade, or sharp cutting tool visible in this image?",
        "Can you see a gun, firearm, rifle, or pistol?",
    ],
    "documents": [
        "Is there any text, document, or written content visible?",
        "Can you see any words, letters, or writing in this image?",
    ],
    "buildings": [
        "Can you see any buildings, houses, or architectural structures?",
        "Is there a wall, door, window, or building structure visible?",
    ],
    "outdoor": ["Is this an outdoor scene or taken outside?"],
    "indoor": ["Is this an indoor scene or taken inside a building?"],
    "objects": [
        "Are there any specific objects, items, or things in this image?",
        "What objects or items can you see in this picture?",
    ],
    "animals": [
        "Is there a dog, cat, or any animal in this image?",
        "Can you see a pet or animal?",
    ],
    "advertising": [
        "Is this an advertisement, commercial poster, or marketing material?",
        "Can you see any brand logos, company names, or advertising content?",
    ],
}

# ============================================================================
# Axe "matching" : lexical (comptage de mots-clés pondéré, mécanisme utilisé
# par app.py avant son passage au cosinus MiniLM ; figé ici pour comparaison,
# indépendant de l'évolution future de app.py) vs semantic
# (app.category_cosine_scores, le mécanisme actuel).
# ============================================================================

LEXICAL_CATEGORY_KEYWORDS = {
    "people": {"keywords": ["person", "man", "woman", "people", "child", "boy", "girl", "human", "face", "crowd", "group"], "weight": 1.0},
    "vehicles": {"keywords": ["car", "vehicle", "truck", "motorcycle", "bike", "bus", "train", "automobile", "taxi", "van"], "weight": 1.0},
    "weapons": {"keywords": ["weapon", "gun", "knife", "rifle", "pistol", "blade", "sharp", "firearm", "cutting"], "weight": 1.2},
    "documents": {"keywords": ["document", "paper", "text", "sign", "writing", "letter", "book", "page", "note", "card", "words"], "weight": 1.0},
    "buildings": {"keywords": ["building", "house", "structure", "architecture", "wall", "door", "window", "roof", "facade"], "weight": 1.0},
    "outdoor": {"keywords": ["outdoor", "outside", "street", "road", "park", "sky", "nature", "exterior", "sidewalk"], "weight": 0.8},
    "indoor": {"keywords": ["indoor", "inside", "room", "interior", "ceiling", "floor", "furniture", "wall"], "weight": 0.8},
    "objects": {"keywords": ["object", "item", "thing", "tool", "equipment", "device", "bag", "box", "bottle", "holding"], "weight": 0.7},
    "animals": {"keywords": ["dog", "cat", "animal", "pet", "bird", "horse"], "weight": 0.6},
    "advertising": {"keywords": ["advertisement", "ad", "brand", "logo", "commercial", "marketing", "poster", "billboard", "sign", "promotion"], "weight": 0.5},
}

LEXICAL_THRESHOLDS = {
    "weapons": 30, "people": 20, "vehicles": 20, "documents": 20,
    "buildings": 20, "outdoor": 15, "indoor": 15, "objects": 25,
    "animals": 15, "advertising": 20,
}
LEXICAL_DEFAULT_THRESHOLD = 20
SEMANTIC_DEFAULT_THRESHOLD = 0.35


def lexical_category_scores(working_text: str) -> dict:
    scores = {}
    for category, config in LEXICAL_CATEGORY_KEYWORDS.items():
        matches = sum(1 for k in config["keywords"] if k in working_text)
        scores[category] = (matches * 25) * config["weight"]
    return scores


def assign_categories(scores: dict, thresholds: dict, default_threshold: float, max_categories: int = 5) -> list:
    """Post-traitement commun aux deux mécanismes (identique à
    app.categories_from_text) : seuil par catégorie, plafond, conflit
    indoor/outdoor (garde le meilleur score), fallback unclassified."""
    sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    assigned = []
    for category, score in sorted_categories:
        threshold = thresholds.get(category, default_threshold)
        if score >= threshold and len(assigned) < max_categories:
            assigned.append(category)

    if "indoor" in assigned and "outdoor" in assigned:
        if scores["indoor"] > scores["outdoor"]:
            assigned.remove("outdoor")
        else:
            assigned.remove("indoor")

    if not assigned:
        assigned.append("unclassified")

    return assigned


# ============================================================================
# Les 4 configurations : {closed, open} x {lexical, semantic}
# ============================================================================

CONFIGS = ["closed_lexical", "closed_semantic", "open_lexical", "open_semantic"]


def get_working_text(image: Image.Image, description: str, questions: str) -> str:
    if questions == "open":
        answers = [app.ask_vqa_question(image, q) for q in app.GENERAL_VQA_QUESTIONS]
    else:
        answers = [
            app.ask_vqa_question(image, q)
            for question_list in CLOSED_CATEGORY_QUESTIONS.values()
            for q in question_list
        ]
    return app.build_working_text(description, answers)


def classify(working_text: str, matching: str) -> tuple:
    if matching == "semantic":
        scores = app.category_cosine_scores(working_text)
        return assign_categories(scores, app.CATEGORY_COSINE_THRESHOLDS, SEMANTIC_DEFAULT_THRESHOLD), scores
    scores = lexical_category_scores(working_text)
    return assign_categories(scores, LEXICAL_THRESHOLDS, LEXICAL_DEFAULT_THRESHOLD), scores


def run_config(config_name: str, records: list, images_dir: Path) -> tuple:
    """Retourne (predictions, raw_scores, débit img/min, pic VRAM Go ou None, temps s)"""
    questions, matching = config_name.split("_", 1)
    predictions = {}
    raw_scores = {}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    for i, record in enumerate(records, 1):
        path = images_dir / record["file_name"]
        try:
            with Image.open(path) as raw:
                image = raw.convert("RGB")
        except Exception as e:
            print(f"  Erreur ouverture {path}: {e}")
            predictions[record["image_id"]] = []
            raw_scores[record["image_id"]] = {}
            continue

        description = app.generate_caption(image)
        working_text = get_working_text(image, description, questions)
        categories, scores = classify(working_text, matching)

        predictions[record["image_id"]] = categories
        raw_scores[record["image_id"]] = scores

        if i % 20 == 0:
            print(f"  [{config_name}] {i}/{len(records)}")

    elapsed = time.time() - start
    throughput = len(records) / (elapsed / 60) if elapsed > 0 else 0.0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None

    return predictions, raw_scores, throughput, peak_vram_gb, elapsed


# ============================================================================
# Métriques multi-label (TP/FP/FN/TN par catégorie, pas de matrice jointe :
# en multi-label une matrice de confusion classique n'a pas de sens direct,
# le TP/FP/FN/TN par catégorie est l'équivalent standard et interprétable)
# ============================================================================

# Seules ces catégories ont une classe COCO correspondante (voir
# build_eval_set.COCO_TO_POLICE) : buildings/advertising/indoor/outdoor
# n'ont aucun signal COCO et sont exclues plutôt que de générer un faux 0%.
EVALUABLE_CATEGORIES = ["people", "vehicles", "weapons", "documents", "animals", "objects"]


def compute_metrics(records: list, predictions: dict) -> dict:
    metrics = {}
    for category in EVALUABLE_CATEGORIES:
        tp = fp = fn = tn = 0
        for record in records:
            gt = category in record["police_categories"]
            pred = category in predictions.get(record["image_id"], [])
            if gt and pred:
                tp += 1
            elif pred and not gt:
                fp += 1
            elif gt and not pred:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall) > 0) else None

        metrics[category] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}

    return metrics


def format_metric(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def print_metrics_table(config_name: str, metrics: dict) -> None:
    print(f"\n--- {config_name} ---")
    print(f"{'Catégorie':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    for category in EVALUABLE_CATEGORIES:
        m = metrics[category]
        print(
            f"{category:<12} {format_metric(m['precision']):>10} {format_metric(m['recall']):>10} "
            f"{format_metric(m['f1']):>10} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {m['tn']:>5}"
        )


def print_weapons_focus(records: list, predictions: dict, config_name: str) -> None:
    print(f"\n=== FOCUS WEAPONS ({config_name}) ===")
    print("Rappel : COCO n'a aucune classe arme à feu, seulement knife/scissors.")
    false_positives = [
        r for r in records
        if "weapons" in predictions.get(r["image_id"], []) and "weapons" not in r["police_categories"]
    ]
    false_negatives = [
        r for r in records
        if "weapons" not in predictions.get(r["image_id"], []) and "weapons" in r["police_categories"]
    ]
    print(f"Faux positifs ({len(false_positives)}) :")
    for r in false_positives[:10]:
        print(f"  - {r['file_name']} (catégories COCO réelles: {r['coco_categories']})")
    print(f"Faux négatifs ({len(false_negatives)}) :")
    for r in false_negatives[:10]:
        print(f"  - {r['file_name']} (catégories COCO réelles: {r['coco_categories']})")


def build_comparison_markdown(all_metrics: dict, all_throughput: dict) -> str:
    lines = ["| Catégorie | Config | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for category in EVALUABLE_CATEGORIES:
        for config_name, metrics in all_metrics.items():
            m = metrics[category]
            lines.append(
                f"| {category} | {config_name} | {format_metric(m['precision'])} | "
                f"{format_metric(m['recall'])} | {format_metric(m['f1'])} |"
            )

    lines += ["", "| Config | Débit (img/min) | Pic VRAM (Go) | Temps total (s) |", "|---|---|---|---|"]
    for config_name, (throughput, peak_vram, elapsed) in all_throughput.items():
        vram_str = f"{peak_vram:.2f}" if peak_vram is not None else "N/A"
        lines.append(f"| {config_name} | {throughput:.1f} | {vram_str} | {elapsed:.1f} |")

    return "\n".join(lines)


# ============================================================================
# Calibration des seuils sémantiques par balayage (F1 max, ou recall max
# sous contrainte de precision pour "weapons")
# ============================================================================

def sweep_category_threshold(records: list, raw_scores: dict, category: str, candidates: list, precision_floor=None):
    """Calcule precision/recall/F1 pour chaque seuil candidat, puis retient
    le meilleur selon le critère demandé. Retourne (meilleur, tous_les_résultats)."""
    results = []
    for t in candidates:
        tp = fp = fn = 0
        for r in records:
            gt = category in r["police_categories"]
            pred = raw_scores.get(r["image_id"], {}).get(category, float("-inf")) >= t
            if gt and pred:
                tp += 1
            elif pred and not gt:
                fp += 1
            elif gt and not pred:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall) > 0) else None
        results.append({"threshold": t, "precision": precision, "recall": recall, "f1": f1})

    if precision_floor is not None:
        eligible = [r for r in results if r["precision"] is not None and r["precision"] >= precision_floor]
        best = max(eligible, key=lambda r: r["recall"] or 0.0) if eligible else None
    else:
        eligible = [r for r in results if r["f1"] is not None]
        best = max(eligible, key=lambda r: r["f1"]) if eligible else None

    return best, results


def calibrate_thresholds(records: list, raw_scores: dict, weapons_precision_floor: float) -> dict:
    candidates = [round(0.20 + 0.01 * i, 2) for i in range(41)]  # 0.20 à 0.60
    calibrated = {}

    print(f"\n{'Catégorie':<12} {'Seuil':>7} {'Precision':>10} {'Recall':>10} {'F1':>8}  Critère")
    for category in EVALUABLE_CATEGORIES:
        if category == "weapons":
            best, _ = sweep_category_threshold(records, raw_scores, category, candidates, precision_floor=weapons_precision_floor)
            critere = f"recall max, precision >= {weapons_precision_floor:.2f}"
        else:
            best, _ = sweep_category_threshold(records, raw_scores, category, candidates)
            critere = "F1 max"

        if best is None:
            print(f"{category:<12} {'N/A':>7} {'--':>10} {'--':>10} {'--':>8}  contrainte non atteignable sur cet échantillon")
            continue

        calibrated[category] = best["threshold"]
        print(
            f"{category:<12} {best['threshold']:>7.2f} {format_metric(best['precision']):>10} "
            f"{format_metric(best['recall']):>10} {format_metric(best['f1']):>8}  {critere}"
        )

    print("\nÀ coller dans CATEGORY_COSINE_THRESHOLDS (app.py) :")
    print("CATEGORY_COSINE_THRESHOLDS = {")
    for category, threshold in calibrated.items():
        print(f'    "{category}": {threshold},')
    print("}")
    print(
        "\n⚠️  Calibré sur le seul échantillon COCO fourni (dont l'échantillonnage est "
        "stratifié pour 'weapons', donc déjà non représentatif) — indicatif, "
        "pas transférable tel quel à un corpus d'enquête réel."
    )

    return calibrated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", default="data/coco_eval/ground_truth.json")
    parser.add_argument("--images-dir", default="data/coco_eval/images")
    parser.add_argument("--config", choices=CONFIGS + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre d'images (test rapide)")
    parser.add_argument("--output-markdown", default=None, help="Fichier où écrire le tableau markdown comparatif")
    parser.add_argument(
        "--calibrate-thresholds", action="store_true",
        help="Balaie les seuils sémantiques (0.20-0.60) sur les configs *_semantic exécutées et affiche les valeurs calibrées",
    )
    parser.add_argument("--weapons-precision-floor", type=float, default=0.50)
    args = parser.parse_args()

    with open(args.ground_truth) as f:
        data = json.load(f)
    records = data["images"]
    if args.limit:
        records = records[: args.limit]

    images_dir = Path(args.images_dir)
    print(f"{len(records)} image(s) à évaluer depuis {args.ground_truth}")

    configs = CONFIGS if args.config == "all" else [args.config]

    all_metrics = {}
    all_throughput = {}
    all_raw_scores = {}

    for config_name in configs:
        print(f"\n=== Exécution : {config_name} ===")
        predictions, raw_scores, throughput, peak_vram, elapsed = run_config(config_name, records, images_dir)
        metrics = compute_metrics(records, predictions)
        all_metrics[config_name] = metrics
        all_throughput[config_name] = (throughput, peak_vram, elapsed)
        all_raw_scores[config_name] = raw_scores
        print_metrics_table(config_name, metrics)
        print_weapons_focus(records, predictions, config_name)

    if len(all_metrics) > 1:
        markdown = build_comparison_markdown(all_metrics, all_throughput)
        print("\n" + "=" * 60)
        print("TABLEAU COMPARATIF (markdown)")
        print("=" * 60)
        print(markdown)
        if args.output_markdown:
            Path(args.output_markdown).write_text(markdown)
            print(f"\nTableau sauvegardé dans {args.output_markdown}")

    if args.calibrate_thresholds:
        semantic_configs = [c for c in configs if c.endswith("_semantic")]
        if not semantic_configs:
            print("\n--calibrate-thresholds nécessite une config *_semantic (ex: open_semantic). Ignoré.")
        else:
            for config_name in semantic_configs:
                print(f"\n=== Calibration des seuils sémantiques ({config_name}) ===")
                calibrate_thresholds(records, all_raw_scores[config_name], args.weapons_precision_floor)


if __name__ == "__main__":
    main()
