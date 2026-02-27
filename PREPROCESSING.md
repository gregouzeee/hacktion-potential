# Preprocessing des donnees — Explication detaillee

Ce document explique la structure des donnees brutes et les differentes etapes de preprocessing utilisees dans chaque notebook.

---

## 1. Les donnees brutes

### Source

Enregistrement extracellulaire dans l'hippocampe d'une souris (M1199) se deplacant dans un labyrinthe en U.
L'electrode est une sonde de type silicon probe avec **4 shanks** (groupes de canaux inseres a differents endroits dans le tissu neural) :

| Shank | Nb canaux | Canaux physiques |
|-------|-----------|------------------|
| 0     | 6         | 47, 55, 56, 42, 43, 44 |
| 1     | 4         | 50, 61, 60, 34 |
| 2     | 6         | 2, 28, 29, 14, 13, 0 |
| 3     | 4         | 25, 22, 23, 11 |

**Total : 20 canaux**. Chaque canal est un microelectrode qui enregistre le potentiel extracellulaire a **20 kHz**.

### Detection des spikes

Quand un neurone emet un potentiel d'action a proximite d'un canal, on observe une deflexion rapide du signal (~1ms). Le systeme d'acquisition detecte ces evenements quand le signal depasse un seuil (defini par canal dans le JSON, ex: canal 47 → seuil = 92 uV). A chaque detection, on extrait une **waveform** : un segment de 32 echantillons (32/20000 = 1.6ms) autour du pic, sur **tous les canaux du meme shank**.

Un spike detecte sur le shank 0 produit donc une waveform de forme **(6, 32)** : 6 canaux x 32 timesteps.

### Fenetrage

L'enregistrement continu (~41 minutes, ~49.5 millions d'echantillons) est decoupe en **fenetres temporelles glissantes** :
- **Duree** : 108ms (~2160 echantillons a 20 kHz)
- **Stride** : ~39ms (pas de 4 a la frequence d'echantillonnage d'origine, d'ou le nom `stride4`)
- **Chevauchement** : les fenetres se recouvrent largement (~64%)

Resultat : **62 257 fenetres**, chacune contenant entre 2 et 190 spikes (moyenne ~55).

---

## 2. Structure du fichier Parquet

Chaque ligne du fichier `M1199_PAG_stride4_win108_test.parquet` represente **une fenetre temporelle** et contient :

### Colonnes principales

| Colonne | Shape | Description |
|---------|-------|-------------|
| `group0` | (n0 * 6 * 32,) | Waveforms aplaties de tous les spikes du shank 0. Reshape en (n0, 6, 32) |
| `group1` | (n1 * 4 * 32,) | Idem shank 1. Reshape en (n1, 4, 32) |
| `group2` | (n2 * 6 * 32,) | Idem shank 2. Reshape en (n2, 6, 32) |
| `group3` | (n3 * 4 * 32,) | Idem shank 3. Reshape en (n3, 4, 32) |
| `length` | (1,) | Nombre total de spikes dans la fenetre |
| `groups` | (length,) | Pour chaque spike (en ordre chronologique), le numero du shank (0-3) |
| `indexInDat` | (length,) | Position absolue du spike dans le fichier d'enregistrement (en echantillons a 20 kHz) |
| `indices0` | (length,) | Index (1-based) dans `group0` pour chaque spike. 0 si le spike n'est pas du shank 0 |
| `indices1` | (length,) | Idem pour shank 1 |
| `indices2` | (length,) | Idem pour shank 2 |
| `indices3` | (length,) | Idem pour shank 3 |
| `pos` | (4,) | Position : [x, y, angle, vitesse]. x et y normalises dans [0, 1] |
| `speedMask` | (1,) | `True` si la souris est en mouvement (vitesse suffisante) |
| `time` | (1,) | Timestamp de la fenetre (en secondes) |
| `time_behavior` | (1,) | Timestamp comportemental (camera de tracking) |
| `indexInDat_raw` | (length,) | Positions brutes des spikes (avant correction) |
| `pos_index` | (1,) | Index dans le fichier de positions |
| `zeroForGather` | (64,) | Vecteur de zeros (utilite interne) |

### Comment reconstruire la sequence chronologique des spikes

Le tableau `groups` donne l'ordre chronologique. Pour le i-eme spike :

1. `groups[i]` donne le shank d'origine (0, 1, 2 ou 3)
2. `indicesX[i]` (ou X = `groups[i]`) donne l'index **1-based** dans le tableau de waveforms du shank correspondant
3. La waveform se trouve dans `groupX.reshape(-1, nCh, 32)[indicesX[i] - 1]`

**Exemple concret** (fenetre avec `length=73`, 15+21+18+19 spikes repartis sur les 4 shanks) :

```
Spike 0 : groups[0]=1 → shank 1, indices1[0]=1 → group1.reshape(-1,4,32)[0] → waveform (4, 32)
Spike 1 : groups[1]=0 → shank 0, indices0[1]=1 → group0.reshape(-1,6,32)[0] → waveform (6, 32)
Spike 2 : groups[2]=2 → shank 2, indices2[2]=1 → group2.reshape(-1,6,32)[0] → waveform (6, 32)
Spike 3 : groups[3]=2 → shank 2, indices2[3]=2 → group2.reshape(-1,6,32)[1] → waveform (6, 32)
...
```

### Filtrage `speedMask`

Seules les fenetres ou la souris est en mouvement (`speedMask=True`) sont utilisees pour l'entrainement et l'evaluation. Raison : quand la souris est immobile, la position ne change pas et les patterns neuronaux sont tres differents (replay, ondes lentes...). On passe de 62 257 a **22 974 fenetres** utilisables.

### Target

La cible a predire est `pos[0]` (x) et `pos[1]` (y), normalises entre 0 et 1. Les deux autres composantes sont l'angle (`pos[2]`, en radians, range [-pi, pi]) et la vitesse (`pos[3]`, range [0, 0.18]).

---

## 3. Preprocessing par notebook

### Notebook 01 — Feature Engineering + XGBoost

**Principe** : extraire des statistiques manuelles a partir des waveforms pour creer un vecteur de features de taille fixe.

**Etapes :**

1. **Reshape des waveforms** : pour chaque shank, `groupX.reshape(-1, nCh, 32)` → matrice (n_spikes, n_canaux, 32)

2. **Calcul des features par shank** (x4 shanks) :
   - `n_spikes` : nombre de spikes detectes sur ce shank
   - `amp_mean`, `amp_max`, `amp_std` : statistiques de l'amplitude peak-to-trough (difference entre le max et le min de chaque waveform, sur le canal dominant)
   - `energy_mean` : energie moyenne des waveforms (somme des carres)
   - `dominant_channel` : canal avec l'amplitude la plus forte

3. **Features globales** :
   - `length` : nombre total de spikes
   - `isi_mean`, `isi_std`, `isi_median` : statistiques des Inter-Spike Intervals (differences entre `indexInDat` consecutifs, converties en ms)
   - `temporal_spread` : ecart entre le premier et le dernier spike (en ms)
   - Ratios inter-shanks : proportion de spikes par shank

4. **Resultat** : un vecteur d'environ ~54 features par fenetre → matrice (22974, 54) prete pour XGBoost/RandomForest.

### Notebook 02 — Transformer

**Principe** : traiter la sequence chronologique de spikes comme une sequence de tokens pour un Transformer.

**Etapes :**

1. **Reconstruction de la sequence** : utiliser `groups` + `indicesX` pour reconstituer la liste ordonnee des waveforms (cf. section 2)

2. **Encoding par shank** : chaque waveform passe dans un petit CNN 1D specifique a son shank :
   - Conv1D (nCh canaux → D filtres) sur les 32 timesteps
   - Produit un embedding de dimension D=64 par spike

3. **Shank embedding** : un embedding appris (dimension D) est ajoute selon le shank d'origine (0-3)

4. **Positional encoding** : encoding sinusoidal base sur la position temporelle normalisee du spike dans la fenetre (en utilisant `indexInDat`)

5. **Padding et masque d'attention** : les fenetres ont un nombre variable de spikes (2-190). On padde toutes les sequences a la meme longueur dans un batch, avec un masque d'attention pour ignorer les positions paddees

6. **Transformer encoder** : 2 couches, 4 heads, dim=64, dropout=0.2

7. **Attention pooling** : plutot qu'un simple mean pooling, un vecteur de requete appris pondère les tokens par attention → vecteur de sortie unique

8. **Head de prediction** : couche Dense → (x, y)

### Notebook 03 — CNN Temporal Bins

**Principe** : discretiser la fenetre temporelle en bins et construire une "image" 2D representant l'activite neuronale.

**Etapes :**

1. **Binning temporel** : diviser les 108ms en bins de 5ms → **22 bins** (108/5 = 21.6, arrondi a 22)

2. **Pour chaque spike** :
   - Calculer dans quel bin temporel il tombe : `bin = floor((indexInDat - indexInDat_min) / (span / n_bins))`
   - Identifier son canal global (shank 0 canaux 0-5, shank 1 canaux 6-9, shank 2 canaux 10-15, shank 3 canaux 16-19)

3. **Construction de l'image** : matrice de shape **(2, 20, 22)** = 2 "canaux d'image" :
   - **Canal 0 : spike count** — nombre de spikes par (canal, bin)
   - **Canal 1 : max amplitude** — amplitude peak-to-trough maximale observee par (canal, bin)

4. **Resultat** : un tenseur (2, 20, 22) par fenetre, directement utilisable par un CNN 2D (similaire a une petite image 20x22 a 2 canaux)

5. **Architecture CNN** : 2-3 blocs Conv2D + BatchNorm + ReLU + MaxPool → Flatten → Dense → (x, y)

### Notebook 05 — Ensemble et ameliorations

**Principe** : combiner les predictions des modeles precedents et ajouter du contexte temporel.

**Preprocessing specifique :**

1. **Kalman a vitesse constante** : post-traitement des predictions brutes. Etat = [x, y, vx, vy], le modele suppose que la vitesse est localement constante entre deux observations (c'est un a priori faible, pas une hypothese forte — le filtre corrige a chaque observation).

2. **Ensemble** : combine les predictions des 3 modeles (XGBoost, Transformer, CNN) par moyenne ponderee (poids inversement proportionnels a l'erreur de chaque modele) ou par stacking (Ridge regression).

3. **GRU multi-fenetre** : prend en entree les features XGBoost de **10 fenetres consecutives** (~1 seconde de contexte) et les passe dans un GRU bidirectionnel pour exploiter la continuite temporelle. Les features sont les memes que le notebook 01, pas de recalcul necessaire.

---

## 4. Split train/test

Le split est **temporel** (pas aleatoire) : les premiers 80% des fenetres en mouvement constituent le train set, les 20% restants le test set. Cela garantit qu'on evalue la capacite du modele a generaliser a un moment futur de l'enregistrement, ce qui est plus realiste qu'un split aleatoire (qui permettrait au modele de "tricher" en interpolant entre des fenetres voisines).

- **Train** : 18 379 fenetres (~29 premieres minutes)
- **Test** : 4 595 fenetres (~12 dernieres minutes)

---

## 5. Metriques

Toutes les metriques sont calculees sur le test set :

| Metrique | Description |
|----------|-------------|
| MSE (x), MSE (y) | Mean Squared Error par coordonnee |
| MAE (x), MAE (y) | Mean Absolute Error par coordonnee |
| R² (x), R² (y) | Coefficient de determination (1 = parfait, 0 = baseline moyenne) |
| Erreur euclidienne | Distance entre position predite et reelle : sqrt((x_pred-x_true)² + (y_pred-y_true)²) |

L'erreur euclidienne est la metrique la plus interpretable physiquement : elle represente la distance en coordonnees normalisees entre la position predite et la position reelle. Une erreur de 0.1 correspond a 10% de la longueur du labyrinthe.

---

## 6. Resume visuel du pipeline

```
Enregistrement brut (20 canaux, 20 kHz, 41 min)
        |
        v
Detection de spikes (seuil par canal)
        |
        v
Fenetrage glissant (108ms, stride ~39ms) → 62 257 fenetres
        |
        v
Filtrage speedMask → 22 974 fenetres (souris en mouvement)
        |
        v
Split temporel 80/20 → 18 379 train / 4 595 test
        |
        v
   ┌────────────────────┬─────────────────────┬──────────────────────┐
   │                    │                     │                      │
   v                    v                     v                      v
NB 01: Features       NB 02: Sequence       NB 03: Image          NB 05: Ensemble
manuelles (~54)       de waveforms +        2D (20x22) +          + Kalman vel.
+ XGBoost/RF          Transformer           CNN 2D                + GRU multi-fenetre
   │                    │                     │                      │
   └────────────────────┴─────────────────────┴──────────────────────┘
                                    |
                                    v
                         NB 04: Comparaison
```
