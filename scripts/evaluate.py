"""
Évalue la pipeline de catégorisation sur le set COCO construit par
build_eval_set.py : precision / recall / F1 multi-label par catégorie
(TP/FP/FN/TN), avec un focus sur "weapons" (fort coût de faux positif en
contexte judiciaire), plus débit (images/minute) et pic de VRAM.

Compare deux configurations :
- vqa_open   : l'approche actuelle d'app.py (3 questions ouvertes communes
  par image, catégories déduites par mots-clés dans les réponses).
- vqa_closed : reconstruction figée de l'ancienne approche (questions
  fermées type yes/no, une à deux par catégorie), tournant sur le même
  modèle BLIP-2 qu'app.py — pour isoler l'effet du style de questions,
  pas celui du modèle.

Usage:
    python scripts/evaluate.py --config both
    python scripts/evaluate.py --config vqa_open --limit 20
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
# Config "vqa_closed" : reconstruction figée de l'approche pré-refactor
# (questions fermées par catégorie). Indépendante de la logique actuelle
# d'app.py (categories_from_text) : c'est un instantané délibérément gelé
# de l'ancienne méthode, à des fins de comparaison.
# ============================================================================

CLOSED_CATEGORY_ANALYSIS = {
    "people": {
        "keywords": ["person", "man", "woman", "people", "child", "boy", "girl", "human", "face", "crowd", "group"],
        "vqa_questions": [
            "Are there any people, persons, or human beings visible in this image?",
            "Can you see a man, woman, or child in this picture?",
        ],
        "weight": 1.0,
    },
    "vehicles": {
        "keywords": ["car", "vehicle", "truck", "motorcycle", "bike", "bus", "train", "automobile", "taxi", "van"],
        "vqa_questions": [
            "Can you see any vehicles, cars, or means of transportation?",
            "Is there a car, truck, motorcycle, or bicycle in this image?",
        ],
        "weight": 1.0,
    },
    "weapons": {
        "keywords": ["weapon", "gun", "knife", "rifle", "pistol", "blade", "sharp", "firearm", "cutting"],
        "vqa_questions": [
            "Is there a knife, blade, or sharp cutting tool visible in this image?",
            "Can you see a gun, firearm, rifle, or pistol?",
        ],
        "weight": 1.2,
    },
    "documents": {
        "keywords": ["document", "paper", "text", "sign", "writing", "letter", "book", "page", "note", "card", "words"],
        "vqa_questions": [
            "Is there any text, document, or written content visible?",
            "Can you see any words, letters, or writing in this image?",
        ],
        "weight": 1.0,
    },
    "animals": {
        "keywords": ["dog", "cat", "animal", "pet", "bird", "horse"],
        "vqa_questions": [
            "Is there a dog, cat, or any animal in this image?",
            "Can you see a pet or animal?",
        ],
        "weight": 0.6,
    },
    "objects": {
        "keywords": ["object", "item", "thing", "tool", "equipment", "device", "bag", "box", "bottle", "holding"],
        "vqa_questions": [
            "Are there any specific objects, items, or things in this image?",
            "What objects or items can you see in this picture?",
        ],
        "weight": 0.7,
    },
}

CLOSED_THRESHOLDS = {"weapons": 30, "people": 20, "vehicles": 20, "documents": 20, "animals": 15, "objects": 25}
POSITIVE_WORDS = ["yes", "true", "there is", "there are", "visible", "can see", "holding"]
NEGATIVE_WORDS = ["no", "not", "none", "cannot", "can't", "nothing"]


def classify_vqa_closed(image: Image.Image, description: str) -> list:
    """Reconstruction de l'ancienne approche à questions fermées, par catégorie."""
    description_lower = description.lower()
    scores = {}

    for category, config in CLOSED_CATEGORY_ANALYSIS.items():
        score = 0.0
        keyword_matches = sum(1 for k in config["keywords"] if k in description_lower)
        if keyword_matches:
            score += keyword_matches * 20 * config["weight"]

        positive_answers = 0
        for question in config["vqa_questions"]:
            answer = app.ask_vqa_question(image, question)
            if not answer:
                continue
            if any(w in answer for w in POSITIVE_WORDS):
                positive_answers += 1
                score += 25 * config["weight"]
            elif any(w in answer for w in NEGATIVE_WORDS):
                score -= 5
            elif any(k in answer for k in config["keywords"][:8]):
                positive_answers += 0.5
                score += 20 * config["weight"]

        if positive_answers >= 2:
            score += 20 * config["weight"]
        scores[category] = score

    assigned = [cat for cat, sc in scores.items() if sc >= CLOSED_THRESHOLDS.get(cat, 20)]
    return assigned or ["unclassified"]


# ============================================================================
# Exécution de la pipeline sur le set d'évaluation
# ============================================================================

def run_config(config_name: str, records: list, images_dir: Path) -> tuple:
    """Retourne (predictions par image_id, débit img/min, pic VRAM en Go ou None, temps total en s)"""
    predictions = {}

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
            continue

        description = app.generate_caption(image)

        if config_name == "vqa_open":
            vqa_answers = [app.ask_vqa_question(image, q) for q in app.GENERAL_VQA_QUESTIONS]
            categories = app.categories_from_text(description, vqa_answers)
        elif config_name == "vqa_closed":
            categories = classify_vqa_closed(image, description)
        else:
            raise ValueError(f"Config inconnue: {config_name}")

        predictions[record["image_id"]] = categories

        if i % 20 == 0:
            print(f"  [{config_name}] {i}/{len(records)}")

    elapsed = time.time() - start
    throughput = len(records) / (elapsed / 60) if elapsed > 0 else 0.0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None

    return predictions, throughput, peak_vram_gb, elapsed


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", default="data/coco_eval/ground_truth.json")
    parser.add_argument("--images-dir", default="data/coco_eval/images")
    parser.add_argument("--config", choices=["vqa_open", "vqa_closed", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre d'images (test rapide)")
    parser.add_argument("--output-markdown", default=None, help="Fichier où écrire le tableau markdown comparatif")
    args = parser.parse_args()

    with open(args.ground_truth) as f:
        data = json.load(f)
    records = data["images"]
    if args.limit:
        records = records[: args.limit]

    images_dir = Path(args.images_dir)
    print(f"{len(records)} image(s) à évaluer depuis {args.ground_truth}")

    configs = ["vqa_open", "vqa_closed"] if args.config == "both" else [args.config]

    all_metrics = {}
    all_throughput = {}

    for config_name in configs:
        print(f"\n=== Exécution : {config_name} ===")
        predictions, throughput, peak_vram, elapsed = run_config(config_name, records, images_dir)
        metrics = compute_metrics(records, predictions)
        all_metrics[config_name] = metrics
        all_throughput[config_name] = (throughput, peak_vram, elapsed)
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


if __name__ == "__main__":
    main()
