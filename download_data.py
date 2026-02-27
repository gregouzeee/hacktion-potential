"""
Télécharge les données depuis Google Cloud Storage.

Usage:
    python download_data.py                    # Télécharge M1199_PAG stride4 win108 (par défaut)
    python download_data.py --mouse M1162_MFB  # Autre souris/région
    python download_data.py --all              # Tous les fichiers parquet
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
    parser.add_argument("--mouse", default="M1199_PAG",
                        help="Identifiant souris_région (ex: M1199_PAG, M1162_MFB)")
    parser.add_argument("--stride", default=4, type=int, help="Stride (1 ou 4)")
    parser.add_argument("--win", default=108, type=int, help="Taille fenêtre (36, 108, 252)")
    parser.add_argument("--all", action="store_true", help="Télécharger tous les fichiers parquet")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # Récupérer la liste des fichiers
    file_list = pd.read_csv(BASE_URL + "file_list.csv")

    if args.all:
        files = [f for f in file_list["p"].tolist() if f.endswith(".parquet") or f.endswith(".json")]
    else:
        prefix = f"{args.mouse}_stride{args.stride}_win{args.win}"
        # Trouver le JSON config (ex: M1199_PAG.json)
        json_name = f"{args.mouse}.json"
        files = [json_name, f"{prefix}_test.parquet"]
        # Ajouter le speedMask si disponible
        speedmask_name = f"{prefix}_speedMask_hab&pre.npy"
        if speedmask_name in file_list["p"].values:
            files.append(speedmask_name)

    print(f"Téléchargement dans {DATA_DIR}/")
    for f in files:
        download_file(f, DATA_DIR)

    print("Terminé !")


if __name__ == "__main__":
    main()
