"""
Benchmark du pipeline d'analyse BLIP-2 sur un dossier d'images.

Mesure le débit (images/minute), le pic de VRAM (torch.cuda.max_memory_allocated)
et le temps total, pour valider la contrainte "~2000 images sur ~4 Go de VRAM"
avec une taille de batch donnée.

Usage:
    python scripts/benchmark.py --folder /chemin/vers/images --batch-size 4
    python scripts/benchmark.py --folder /chemin/vers/images --batch-size 8 --reuse-db
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import app


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list:
    return sorted(
        str(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_benchmark(folder: str, batch_size: int, reuse_db: bool) -> None:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"Erreur : {folder} n'est pas un dossier valide.")
        sys.exit(1)

    image_paths = list_images(folder_path)
    if not image_paths:
        print(f"Erreur : aucune image trouvée dans {folder} ({sorted(IMAGE_EXTENSIONS)}).")
        sys.exit(1)

    print(f"{len(image_paths)} image(s) trouvée(s) dans {folder}")
    print(f"Taille de batch : {batch_size}")
    print(f"Device : {app.device}")

    # Par défaut, base SQLite jetable pour mesurer le coût réel de l'analyse
    # (pas la reprise). --reuse-db pointe sur la base de l'app pour mesurer
    # la reprise après interruption.
    if reuse_db:
        db_path = app.DB_PATH
        print(f"Base SQLite réutilisée : {db_path}")
    else:
        db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_file.close()
        db_path = db_file.name
        app.DB_PATH = db_path
        print(f"Base SQLite jetable : {db_path}")

    state = app.EnqueteData()
    state.images = [
        {
            "path": path,
            "filename": Path(path).name,
            "upload_date": "",
        }
        for path in image_paths
    ]
    state.enquete_info["nombre_images"] = len(state.images)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def progress_log(fraction, desc=""):
        print(f"  [{fraction * 100:5.1f}%] {desc}")

    start = time.time()
    app.analyze_all_images(state, batch_size=batch_size, progress=progress_log)
    total_time = time.time() - start

    analyzed = sum(1 for a in state.analyses.values() if a.get("analyzed"))
    throughput = analyzed / (total_time / 60) if total_time > 0 else 0.0

    print("\n" + "=" * 60)
    print("RÉSULTATS DU BENCHMARK")
    print("=" * 60)
    print(f"Images analysées      : {analyzed}/{len(image_paths)}")
    print(f"Temps total           : {total_time:.1f} s")
    print(f"Débit                 : {throughput:.1f} images/minute")
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Pic de VRAM           : {peak_vram_gb:.2f} Go")
    else:
        print("Pic de VRAM           : N/A (pas de GPU CUDA disponible)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, help="Dossier contenant les images à analyser")
    parser.add_argument("--batch-size", type=int, default=4, help="Taille de batch pour l'inférence BLIP-2 (défaut : 4)")
    parser.add_argument(
        "--reuse-db", action="store_true",
        help="Utiliser la base SQLite de l'application (mesure la reprise) au lieu d'une base jetable",
    )
    args = parser.parse_args()
    run_benchmark(args.folder, args.batch_size, args.reuse_db)


if __name__ == "__main__":
    main()
