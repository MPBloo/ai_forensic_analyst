"""
Construit un set d'évaluation à partir d'un sous-ensemble de COCO val2017.

Télécharge (si absent) les annotations COCO val2017 officielles et un
sous-ensemble d'images, mappe les catégories COCO vers CATEGORIES_POLICE
(mapping validé, voir EVALUATION.md), et sauvegarde la vérité terrain en
JSON pour scripts/evaluate.py.

Usage:
    python scripts/build_eval_set.py --n-images 200 --output data/coco_eval
    python scripts/build_eval_set.py --n-images 200 --skip-images  # JSON seul, pour test rapide
"""
import argparse
import json
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_BASE_URL = "http://images.cocodataset.org/val2017/"
ANNOTATIONS_FILENAME = "instances_val2017.json"

# Mapping validé : catégorie COCO -> catégorie CATEGORIES_POLICE.
# Catégories COCO absentes de ce dict = non mappées (ignorées, ni positif ni
# négatif dans la vérité terrain).
# Catégories CATEGORIES_POLICE sans entrée ici (buildings, advertising,
# indoor, outdoor) = pas de classe COCO correspondante, volontairement
# exclues de l'évaluation (voir EVALUATION.md, section limites).
COCO_TO_POLICE = {
    "person": "people",
    "bicycle": "vehicles", "car": "vehicles", "motorcycle": "vehicles",
    "airplane": "vehicles", "bus": "vehicles", "train": "vehicles",
    "truck": "vehicles", "boat": "vehicles",
    "knife": "weapons", "scissors": "weapons",
    "book": "documents",
    "bird": "animals", "cat": "animals", "dog": "animals", "horse": "animals",
    "sheep": "animals", "cow": "animals", "elephant": "animals", "bear": "animals",
    "zebra": "animals", "giraffe": "animals",
    "backpack": "objects", "handbag": "objects", "suitcase": "objects", "bottle": "objects",
}

# COCO n'a aucune classe arme à feu : "weapons" ne peut être testé que sur
# des objets tranchants. À rappeler dans toute lecture des résultats.
WEAPONS_COCO_CATEGORIES = {"knife", "scissors"}


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  déjà présent : {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  téléchargement : {url}")
    urllib.request.urlretrieve(url, dest)


def ensure_annotations(cache_dir: Path) -> Path:
    ann_path = cache_dir / "annotations" / ANNOTATIONS_FILENAME
    if ann_path.exists():
        return ann_path
    zip_path = cache_dir / "annotations_trainval2017.zip"
    download_file(COCO_ANNOTATIONS_URL, zip_path)
    print("  extraction des annotations...")
    with zipfile.ZipFile(zip_path) as z:
        z.extract(f"annotations/{ANNOTATIONS_FILENAME}", cache_dir)
    return ann_path


def load_coco_index(coco: dict):
    """
    À partir du JSON d'annotations COCO (dict avec "images", "annotations",
    "categories"), retourne (images_by_id, categories_by_image_id).
    """
    category_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    images_by_id = {img["id"]: img for img in coco["images"]}

    categories_by_image = defaultdict(set)
    for ann in coco["annotations"]:
        cat_name = category_id_to_name[ann["category_id"]]
        categories_by_image[ann["image_id"]].add(cat_name)

    return images_by_id, categories_by_image


def select_images(images_by_id: dict, categories_by_image: dict, n_images: int, n_weapons_min: int, seed: int) -> list:
    """
    Échantillonnage stratifié : garantit au moins n_weapons_min images
    contenant knife/scissors (rares dans COCO, sinon "weapons" aurait ~0
    exemple positif sur un tirage aléatoire de 200 images), le reste tiré
    au hasard. Rend l'échantillon non représentatif de la distribution
    naturelle de COCO — choix délibéré pour un signal exploitable sur la
    catégorie prioritaire (voir EVALUATION.md).
    """
    rng = random.Random(seed)
    all_ids = list(images_by_id.keys())

    weapons_ids = [
        img_id for img_id in all_ids
        if categories_by_image[img_id] & WEAPONS_COCO_CATEGORIES
    ]
    rng.shuffle(weapons_ids)
    selected_weapons = weapons_ids[:n_weapons_min]
    selected_weapons_set = set(selected_weapons)

    remaining_pool = [img_id for img_id in all_ids if img_id not in selected_weapons_set]
    rng.shuffle(remaining_pool)
    n_remaining = max(0, n_images - len(selected_weapons))
    selected_rest = remaining_pool[:n_remaining]

    selected = selected_weapons + selected_rest
    rng.shuffle(selected)
    return selected[:n_images]


def build_ground_truth(images_by_id: dict, categories_by_image: dict, selected_ids: list) -> list:
    records = []
    for img_id in selected_ids:
        img = images_by_id[img_id]
        coco_cats = sorted(categories_by_image[img_id])
        police_cats = sorted({COCO_TO_POLICE[c] for c in coco_cats if c in COCO_TO_POLICE})
        records.append({
            "image_id": img_id,
            "file_name": img["file_name"],
            "coco_categories": coco_cats,
            "police_categories": police_cats,
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-images", type=int, default=200)
    parser.add_argument(
        "--n-weapons-min", type=int, default=18,
        help="Nombre minimum d'images contenant knife/scissors (échantillonnage stratifié, défaut : 18)",
    )
    parser.add_argument("--output", default="data/coco_eval", help="Dossier de sortie")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-images", action="store_true",
        help="Ne télécharge pas les fichiers image, seulement la vérité terrain JSON (utile pour un test rapide)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    cache_dir = output_dir / "_coco_cache"
    images_dir = output_dir / "images"

    print("1/4 - Annotations COCO...")
    ann_path = ensure_annotations(cache_dir)
    with open(ann_path) as f:
        coco = json.load(f)

    print("2/4 - Indexation...")
    images_by_id, categories_by_image = load_coco_index(coco)
    print(f"  {len(images_by_id)} images dans val2017")

    print("3/4 - Échantillonnage stratifié...")
    selected_ids = select_images(images_by_id, categories_by_image, args.n_images, args.n_weapons_min, args.seed)
    n_weapons = sum(1 for img_id in selected_ids if categories_by_image[img_id] & WEAPONS_COCO_CATEGORIES)
    print(f"  {len(selected_ids)} images sélectionnées, dont {n_weapons} avec knife/scissors")

    ground_truth = build_ground_truth(images_by_id, categories_by_image, selected_ids)

    if not args.skip_images:
        print("4/4 - Téléchargement des images...")
        for i, record in enumerate(ground_truth, 1):
            dest = images_dir / record["file_name"]
            download_file(COCO_IMAGE_BASE_URL + record["file_name"], dest)
            if i % 20 == 0:
                print(f"  {i}/{len(ground_truth)}")
    else:
        print("4/4 - Téléchargement des images ignoré (--skip-images)")

    output_dir.mkdir(parents=True, exist_ok=True)
    gt_path = output_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump({"mapping": COCO_TO_POLICE, "images": ground_truth}, f, indent=2, ensure_ascii=False)

    print(f"\nVérité terrain sauvegardée : {gt_path}")
    if not args.skip_images:
        print(f"Images : {images_dir}")


if __name__ == "__main__":
    main()
