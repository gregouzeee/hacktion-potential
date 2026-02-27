# Hacktion Potential

Projet de **decodage de la position spatiale** d'une souris a partir de l'activite neuronale enregistree dans l'hippocampe (spikes extracellulaires).

## Donnees

- **Souris** : M1199, enregistrement PAG (periaquaductal gray / hippocampe)
- **Electrodes** : 4 shanks (groupes), 20 canaux au total (6+4+6+4)
- **Fenetres** : 108ms, stride 4, ~62k fenetres dont ~23k avec la souris en mouvement
- **Target** : position (x, y) normalisee [0, 1]

Les fichiers de donnees (`.parquet`, `.json`) ne sont pas inclus dans le repo. Utilisez le script de telechargement :
```bash
python download_data.py                    # M1199_PAG stride4 win108 (defaut)
python download_data.py --mouse M1162_MFB  # Autre souris
python download_data.py --all              # Tous les fichiers
```
Les fichiers sont telecharges dans `data/`.

Sur **SSPCloud/Onyxia**, les notebooks detectent automatiquement le stockage S3 (MinIO) si les credentials sont disponibles.

## Structure du projet

```
data/               # Donnees telechargees (.parquet, .json, .npy) [gitignored]
notebooks/          # Notebooks d'experimentation (a executer dans l'ordre)
scripts/            # Script d'analyse exploratoire
outputs/            # Modeles sauvegardes (.pt), predictions (.npy) [gitignored]
figures/            # Figures generees par le script EDA [gitignored]
download_data.py    # Script de telechargement des donnees
```

## Notebooks

| # | Notebook | Approche | Entree |
|---|----------|----------|--------|
| 01 | Feature Engineering + XGBoost | Features manuelles (~54) + ML classique (XGBoost, RF, GB) + Kalman | Statistiques agregees par fenetre |
| 02 | Transformer | CNN 1D par shank + self-attention + positional encoding | Sequence chronologique de waveforms bruts |
| 03 | CNN Temporal Bins | Image 2D (canaux x bins de 5ms) + CNN 2D | Matrice spike-count / amplitude |
| 04 | Comparaison | Charge les predictions des notebooks 1-3 et 5, compare les metriques | Fichiers `.npy` |
| 05 | Ensemble et ameliorations | Kalman a vitesse constante, ensemble pondere, GRU multi-fenetre | Predictions des modeles precedents |

## Resultats

Metriques sur le test set (20% temporel) :

| Modele | R² X | R² Y | Erreur eucl. moyenne |
|--------|------|------|---------------------|
| XGBoost | 0.20 | 0.39 | 0.327 |
| XGBoost + Kalman | 0.32 | 0.49 | 0.310 |
| Transformer | ~ | ~ | ~ |
| CNN | 0.18 | 0.33 | 0.346 |

> Les resultats du Transformer et des modeles du notebook 05 dependent de l'execution.

## Reproduction

```bash
pip install -r requirements.txt
python download_data.py
# Puis executer les notebooks dans l'ordre (01 → 02 → 03 → 05 → 04)
```

## Stack

- Python 3.13
- PyTorch (MPS / CUDA / CPU)
- XGBoost, scikit-learn
- pandas, numpy, matplotlib
