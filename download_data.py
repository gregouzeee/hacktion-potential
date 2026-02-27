"""
Télécharge toutes les données depuis Google Cloud Storage.

Usage:
    python download_data.py            # Télécharge tous les fichiers (parquet, json, npy)
    python download_data.py --format t # Télécharge les tfrec au lieu des parquet
"""
import os
import argparse
import requests
import pandas as pd

BASE_URL = "https://storage.googleapis.com/projet2_hacktion_potential/"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download_file(filename, dest_dir, file_num=0, total_files=0):
    """Télécharge un fichier depuis GCS dans dest_dir avec barre de progression."""
    prefix = f"  [{file_num}/{total_files}]" if total_files else " "
    path = os.path.join(dest_dir, filename)
    if os.path.exists(path):
        print(f"{prefix} Déjà présent : {filename}")
        return
    r = requests.get(BASE_URL + filename, stream=True)
    r.raise_for_status()
    total_size = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                bar_len = 30
                filled = int(bar_len * downloaded / total_size)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r{prefix} {filename}  {bar} {pct:5.1f}%  ({downloaded/1e6:.1f}/{total_size/1e6:.1f} MB)", end="", flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(description="Télécharge les données du hackathon")
    parser.add_argument("--format", choices=["p", "t"], default="p",
                        help="Format: p=parquet (défaut), t=tfrec")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # Récupérer la liste des fichiers
    file_list = pd.read_csv(BASE_URL + "file_list.csv")
    files = file_list[args.format].tolist()

    print(f"Téléchargement de {len(files)} fichiers dans {DATA_DIR}/")
    for i, f in enumerate(files, 1):
        download_file(f, DATA_DIR, file_num=i, total_files=len(files))

    print("Terminé !")


if __name__ == "__main__":
    main()
