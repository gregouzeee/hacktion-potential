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


def download_file(filename, dest_dir):
    """Télécharge un fichier depuis GCS dans dest_dir."""
    path = os.path.join(dest_dir, filename)
    if os.path.exists(path):
        print(f"  Déjà présent : {filename}")
        return
    print(f"  Téléchargement : {filename} ...", end=" ", flush=True)
    r = requests.get(BASE_URL + filename, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(path) / 1e6
    print(f"{size_mb:.1f} MB")


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
    for f in files:
        download_file(f, DATA_DIR)

    print("Terminé !")


if __name__ == "__main__":
    main()
