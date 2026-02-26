"""
Analyse exploratoire des données HacktionPotential - M1199_PAG
Hackathon Neurosciences - Décodage spatial à partir de spikes hippocampiques

Usage:
    python analyse_exploratoire.py

Génère 10 figures dans le dossier courant (fig1 à fig10).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
import os

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MOUSE = "M1199_PAG"
STRIDE = 4
WINDOW_SIZE = 108
PARQUET_FILE = os.path.join(DATA_DIR, f"{MOUSE}_stride{STRIDE}_win{WINDOW_SIZE}_test.parquet")
JSON_FILE = os.path.join(DATA_DIR, f"{MOUSE}.json")
OUTPUT_DIR = DATA_DIR

# =============================================================================
# Chargement des données
# =============================================================================
print("Chargement du fichier Parquet...")
df = pd.read_parquet(PARQUET_FILE)
print(f"  Shape: {df.shape}")
print(f"  Colonnes: {df.columns.tolist()}")

with open(JSON_FILE, "r") as f:
    params = json.load(f)

nGroups = params["nGroups"]
nChannelsPerGroup = [params[f"group{g}"]["nChannels"] for g in range(nGroups)]
print(f"  nGroups: {nGroups}, nChannelsPerGroup: {nChannelsPerGroup}")

# =============================================================================
# Extraction des variables scalaires
# =============================================================================
print("\nExtraction des variables...")
pos_x = np.array([x[0] for x in df['pos']])
pos_y = np.array([x[1] for x in df['pos']])
pos_2 = np.array([x[2] for x in df['pos']])
pos_3 = np.array([x[3] for x in df['pos']])
times = np.array([x[0] for x in df['time']])
time_beh = np.array([x[0] for x in df['time_behavior']])
speed_masks = np.array([x[0] for x in df['speedMask']])
lengths = np.array([x[0] if hasattr(x, '__len__') else x for x in df['length']])
pos_indices = np.array([x[0] if hasattr(x, '__len__') else x for x in df['pos_index']])

# =============================================================================
# Statistiques descriptives
# =============================================================================
print("\n" + "=" * 60)
print("STATISTIQUES DESCRIPTIVES")
print("=" * 60)

print(f"\nNombre total d'exemples: {len(df)}")

print(f"\n--- Length (nb spikes par fenêtre de {WINDOW_SIZE}ms) ---")
print(f"  min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}, "
      f"median={np.median(lengths):.0f}, std={lengths.std():.1f}")

print(f"\n--- Temps ---")
print(f"  min={times.min():.1f}s, max={times.max():.1f}s, "
      f"durée={times.max()-times.min():.1f}s ({(times.max()-times.min())/60:.1f} min)")

print(f"\n--- SpeedMask ---")
print(f"  En mouvement: {speed_masks.sum()} ({speed_masks.mean()*100:.1f}%)")
print(f"  Immobile: {(~speed_masks).sum()} ({(~speed_masks).mean()*100:.1f}%)")

print(f"\n--- Position X ---")
print(f"  min={pos_x.min():.4f}, max={pos_x.max():.4f}, "
      f"mean={pos_x.mean():.4f}, std={pos_x.std():.4f}")
print(f"--- Position Y ---")
print(f"  min={pos_y.min():.4f}, max={pos_y.max():.4f}, "
      f"mean={pos_y.mean():.4f}, std={pos_y.std():.4f}")
print(f"--- Dim 2 (direction tête probable) ---")
print(f"  min={pos_2.min():.4f}, max={pos_2.max():.4f}, "
      f"mean={pos_2.mean():.4f}, std={pos_2.std():.4f}")
print(f"--- Dim 3 (vitesse probable) ---")
print(f"  min={pos_3.min():.4f}, max={pos_3.max():.4f}, "
      f"mean={pos_3.mean():.4f}, std={pos_3.std():.4f}")

print(f"\n--- Distribution des groupes ---")
all_groups = np.concatenate([x for x in df['groups']])
unique, counts = np.unique(all_groups, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Group {u}: {c} spikes ({c/len(all_groups)*100:.1f}%)")
print(f"  Total spikes: {len(all_groups)}")

print(f"\n--- Spikes par groupe (premiers 10 exemples) ---")
for g in range(nGroups):
    shapes = [len(df[f'group{g}'].iloc[i]) for i in range(min(10, len(df)))]
    nch = nChannelsPerGroup[g]
    n_spikes = [s // (nch * 32) for s in shapes]
    print(f"  group{g} (nCh={nch}): n_spikes = {n_spikes}")

print(f"\n--- Pos index ---")
print(f"  min={pos_indices.min()}, max={pos_indices.max()}, n_unique={len(np.unique(pos_indices))}")
diffs = np.diff(np.sort(pos_indices))
print(f"  Stride most common: {np.bincount(diffs.astype(int)).argmax()}")


# =============================================================================
# FIGURE 1 : Trajectoire de la souris
# =============================================================================
print("\nGénération Figure 1 : Trajectoire...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sc = axes[0].scatter(pos_x, pos_y, c=times - times.min(), cmap='viridis', s=0.5, alpha=0.3)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title('Trajectoire (couleur = temps)')
axes[0].set_aspect('equal')
plt.colorbar(sc, ax=axes[0], label='Temps (s)')

axes[1].scatter(pos_x[~speed_masks], pos_y[~speed_masks], c='gray', s=0.5, alpha=0.2, label='Immobile')
axes[1].scatter(pos_x[speed_masks], pos_y[speed_masks], c='red', s=0.5, alpha=0.3, label='En mouvement')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_title('Trajectoire (Speed Mask)')
axes[1].set_aspect('equal')
axes[1].legend(markerscale=10)

h = axes[2].hist2d(pos_x, pos_y, bins=50, cmap='hot', norm=LogNorm())
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
axes[2].set_title('Heatmap des positions visitées')
axes[2].set_aspect('equal')
plt.colorbar(h[3], ax=axes[2], label='Nombre de visites (log)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_trajectoire.png'), dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 2 : Heatmap toutes positions vs en mouvement
# =============================================================================
print("Génération Figure 2 : Heatmap comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

h1 = axes[0].hist2d(pos_x, pos_y, bins=50, cmap='hot', norm=LogNorm())
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title('Toutes les positions')
axes[0].set_aspect('equal')
plt.colorbar(h1[3], ax=axes[0], label='Count (log)')

h2 = axes[1].hist2d(pos_x[speed_masks], pos_y[speed_masks], bins=50, cmap='hot', norm=LogNorm())
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_title('Positions en mouvement uniquement (speedMask=True)')
axes[1].set_aspect('equal')
plt.colorbar(h2[3], ax=axes[1], label='Count (log)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_heatmap_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 3 : Distribution des spikes
# =============================================================================
print("Génération Figure 3 : Distribution des spikes...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(lengths, bins=80, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean={np.mean(lengths):.1f}')
axes[0, 0].axvline(np.median(lengths), color='orange', linestyle='--', label=f'Median={np.median(lengths):.0f}')
axes[0, 0].set_xlabel('Nombre de spikes par fenêtre')
axes[0, 0].set_ylabel('Fréquence')
axes[0, 0].set_title(f'Distribution du nombre de spikes (window={WINDOW_SIZE}ms)')
axes[0, 0].legend()

group_spike_counts = {g: [] for g in range(nGroups)}
for idx in range(len(df)):
    groups = df['groups'].iloc[idx]
    for g in range(nGroups):
        group_spike_counts[g].append(np.sum(groups == g))

group_means = [np.mean(group_spike_counts[g]) for g in range(nGroups)]
group_stds = [np.std(group_spike_counts[g]) for g in range(nGroups)]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = axes[0, 1].bar(range(nGroups), group_means, yerr=group_stds, color=colors,
                       edgecolor='black', capsize=5, alpha=0.8)
for i, bar in enumerate(bars):
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + group_stds[i] + 0.3,
                    f'{nChannelsPerGroup[i]} ch', ha='center', va='bottom', fontsize=10)
axes[0, 1].set_xlabel('Groupe (shank)')
axes[0, 1].set_ylabel('Nombre moyen de spikes')
axes[0, 1].set_title('Spikes moyens par groupe et par fenêtre')
axes[0, 1].set_xticks(range(nGroups))

window = 500
sorted_idx = np.argsort(times)
sorted_times = times[sorted_idx]
sorted_lengths = lengths[sorted_idx]
ma_lengths = np.convolve(sorted_lengths, np.ones(window) / window, mode='valid')
ma_times = sorted_times[window // 2:window // 2 + len(ma_lengths)]
axes[1, 0].plot(ma_times - times.min(), ma_lengths, color='steelblue', linewidth=0.5)
axes[1, 0].set_xlabel('Temps (s)')
axes[1, 0].set_ylabel('Nombre de spikes (moy. glissante)')
axes[1, 0].set_title(f'Évolution temporelle du nombre de spikes (MA={window})')
axes[1, 0].fill_between(ma_times - times.min(), ma_lengths, alpha=0.2, color='steelblue')

axes[1, 1].hist(lengths[speed_masks], bins=60, color='red', alpha=0.5,
                label=f'En mouvement (n={speed_masks.sum()})', density=True)
axes[1, 1].hist(lengths[~speed_masks], bins=60, color='gray', alpha=0.5,
                label=f'Immobile (n={(~speed_masks).sum()})', density=True)
axes[1, 1].set_xlabel('Nombre de spikes par fenêtre')
axes[1, 1].set_ylabel('Densité')
axes[1, 1].set_title('Distribution des spikes : mouvement vs immobilité')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_spike_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 4 : Distribution des 4 dimensions de position
# =============================================================================
print("Génération Figure 4 : Distributions des positions...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(pos_x, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Position X')
axes[0, 0].set_ylabel('Fréquence')
axes[0, 0].set_title('Distribution de X')

axes[0, 1].hist(pos_y, bins=100, color='darkorange', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Position Y')
axes[0, 1].set_ylabel('Fréquence')
axes[0, 1].set_title('Distribution de Y')

axes[1, 0].hist(pos_2, bins=100, color='green', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Pos dim 2 (angle/head direction?)')
axes[1, 0].set_ylabel('Fréquence')
axes[1, 0].set_title('Dim 2: [-π, π] → probablement direction tête')

axes[1, 1].hist(pos_3, bins=100, color='purple', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Pos dim 3 (vitesse ?)')
axes[1, 1].set_ylabel('Fréquence')
axes[1, 1].set_title('Dim 3: [0, 0.18] → probablement vitesse')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_position_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 5 : Évolution temporelle des positions
# =============================================================================
print("Génération Figure 5 : Positions temporelles...")
fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

t_rel = times - times.min()
axes[0].scatter(t_rel, pos_x, c=np.where(speed_masks, 'red', 'gray'), s=0.2, alpha=0.3)
axes[0].set_ylabel('Position X')
axes[0].set_title('Position X au cours du temps')

axes[1].scatter(t_rel, pos_y, c=np.where(speed_masks, 'red', 'gray'), s=0.2, alpha=0.3)
axes[1].set_ylabel('Position Y')
axes[1].set_title('Position Y au cours du temps')

axes[2].scatter(t_rel, pos_3, c=np.where(speed_masks, 'red', 'gray'), s=0.2, alpha=0.3)
axes[2].set_ylabel('Dim 3 (vitesse?)')
axes[2].set_xlabel('Temps (s)')
axes[2].set_title('Dim 3 (vitesse probable) au cours du temps')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_temporal_positions.png'), dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 6 : Exemples de spike waveforms
# =============================================================================
print("Génération Figure 6 : Spike waveforms...")
fig, axes = plt.subplots(nGroups, 5, figsize=(20, 14))

moving_idx = np.where(speed_masks)[0]
sample_idx = moving_idx[1000]
sample = df.iloc[sample_idx]

for g in range(nGroups):
    nCh = nChannelsPerGroup[g]
    raw = sample[f'group{g}']
    spikes = raw.reshape(-1, nCh, 32)
    n_spikes = spikes.shape[0]

    for s in range(min(5, n_spikes)):
        ax = axes[g, s]
        for ch in range(nCh):
            ax.plot(np.arange(32) / 20.0, spikes[s, ch, :],
                    label=f'Ch {ch}' if s == 0 else None)
        ax.set_title(f'Group {g}, Spike {s + 1}', fontsize=9)
        if s == 0:
            ax.set_ylabel(f'Group {g}\n({nCh} ch)\nVoltage')
        if g == nGroups - 1:
            ax.set_xlabel('Time (ms)')

axes[0, 0].legend(fontsize=7, loc='upper right')
fig.suptitle(f'Exemples de waveforms de spikes (sample #{sample_idx}, en mouvement)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_spike_waveforms.png'), dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 7 : Distribution des amplitudes de spikes
# =============================================================================
print("Génération Figure 7 : Amplitudes des spikes...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

np.random.seed(42)
sample_indices = np.random.choice(len(df), size=min(500, len(df)), replace=False)

for g in range(nGroups):
    nCh = nChannelsPerGroup[g]
    amplitudes = []
    for idx in sample_indices:
        raw = df[f'group{g}'].iloc[idx]
        spikes = raw.reshape(-1, nCh, 32)
        for s in range(spikes.shape[0]):
            amp = np.max(spikes[s]) - np.min(spikes[s])
            amplitudes.append(amp)

    ax = axes[g // 2, g % 2]
    ax.hist(amplitudes, bins=100, color=colors[g], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Amplitude peak-to-trough')
    ax.set_ylabel('Count')
    ax.set_title(f'Group {g} ({nCh} channels) - Distribution des amplitudes')
    ax.axvline(np.mean(amplitudes), color='red', linestyle='--',
               label=f'Mean={np.mean(amplitudes):.1f}')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_spike_amplitudes.png'), dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 8 : Densité de spikes par position spatiale
# =============================================================================
print("Génération Figure 8 : Densité spatiale des spikes...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

nbins = 30
x_edges = np.linspace(0, 1, nbins + 1)
y_edges = np.linspace(0, 1, nbins + 1)

spike_map = np.full((nbins, nbins), np.nan)
count_map = np.zeros((nbins, nbins))

for i in range(len(df)):
    xi = np.clip(np.searchsorted(x_edges, pos_x[i]) - 1, 0, nbins - 1)
    yi = np.clip(np.searchsorted(y_edges, pos_y[i]) - 1, 0, nbins - 1)
    if np.isnan(spike_map[yi, xi]):
        spike_map[yi, xi] = 0
    spike_map[yi, xi] += lengths[i]
    count_map[yi, xi] += 1

mean_spike_map = np.where(count_map > 0, spike_map / count_map, np.nan)

im1 = axes[0].imshow(mean_spike_map, origin='lower', aspect='equal', cmap='viridis',
                      extent=[0, 1, 0, 1])
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title('Nombre moyen de spikes par position')
plt.colorbar(im1, ax=axes[0], label='Spikes moyens')

im2 = axes[1].imshow(count_map, origin='lower', aspect='equal', cmap='hot',
                      extent=[0, 1, 0, 1])
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_title('Occupation spatiale (nombre de visites)')
plt.colorbar(im2, ax=axes[1], label='Visites')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_spatial_spike_density.png'), dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 9 : Analyses de corrélation
# =============================================================================
print("Génération Figure 9 : Corrélations...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 9a) Vitesse vs spike count
axes[0, 0].scatter(pos_3, lengths, s=0.5, alpha=0.2, c='steelblue')
axes[0, 0].set_xlabel('Dim 3 (vitesse probable)')
axes[0, 0].set_ylabel('Nombre de spikes')
axes[0, 0].set_title(f'Vitesse vs Nb spikes (r={np.corrcoef(pos_3, lengths)[0, 1]:.3f})')

# 9b) Dim 3 par état de mouvement
axes[0, 1].hist(pos_3[speed_masks], bins=80, alpha=0.6, color='red', density=True, label='Moving')
axes[0, 1].hist(pos_3[~speed_masks], bins=80, alpha=0.6, color='gray', density=True, label='Still')
axes[0, 1].set_xlabel('Dim 3')
axes[0, 1].set_ylabel('Densité')
axes[0, 1].set_title('Distribution dim 3 par état de mouvement')
axes[0, 1].legend()

# 9c) Direction de tête colorée sur trajectoire
axes[0, 2].scatter(pos_x[speed_masks], pos_y[speed_masks], c=pos_2[speed_masks],
                    cmap='hsv', s=0.5, alpha=0.3)
axes[0, 2].set_xlabel('X')
axes[0, 2].set_ylabel('Y')
axes[0, 2].set_title('Position colorée par dim 2 (direction tête?)')
axes[0, 2].set_aspect('equal')

# 9d) Inter-spike intervals
np.random.seed(42)
isis = []
for idx in np.random.choice(len(df), size=2000, replace=False):
    iid = df['indexInDat'].iloc[idx]
    if len(iid) > 1:
        diffs = np.diff(iid) / 20.0  # ms (20kHz)
        isis.extend(diffs[diffs > 0])

axes[1, 0].hist(isis, bins=200, color='steelblue', edgecolor='black', alpha=0.7, range=(0, 50))
axes[1, 0].set_xlabel('Inter-spike interval (ms)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title(f'Distribution des ISI (median={np.median(isis):.2f}ms)')
axes[1, 0].axvline(np.median(isis), color='red', linestyle='--')

# 9e) Proportion de chaque groupe par fenêtre
group_props = {g: [] for g in range(nGroups)}
for idx in range(min(5000, len(df))):
    groups = df['groups'].iloc[idx]
    total = len(groups)
    for g in range(nGroups):
        group_props[g].append(np.sum(groups == g) / total)

bp_data = [group_props[g] for g in range(nGroups)]
bp = axes[1, 1].boxplot(bp_data, patch_artist=True,
                         tick_labels=[f'Group {g}' for g in range(nGroups)])
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 1].set_ylabel('Proportion des spikes')
axes[1, 1].set_title('Proportion de chaque groupe par fenêtre')

# 9f) Autocorrélation des positions
moving_x = pos_x[speed_masks]
moving_y = pos_y[speed_masks]
max_lag = 100
autocorr_x = np.correlate(moving_x[:5000] - moving_x[:5000].mean(),
                           moving_x[:5000] - moving_x[:5000].mean(), mode='full')
autocorr_x = autocorr_x[len(autocorr_x) // 2:len(autocorr_x) // 2 + max_lag]
autocorr_x /= autocorr_x[0]

autocorr_y = np.correlate(moving_y[:5000] - moving_y[:5000].mean(),
                           moving_y[:5000] - moving_y[:5000].mean(), mode='full')
autocorr_y = autocorr_y[len(autocorr_y) // 2:len(autocorr_y) // 2 + max_lag]
autocorr_y /= autocorr_y[0]

lags = np.arange(max_lag) / 15.0  # ~15Hz sampling
axes[1, 2].plot(lags, autocorr_x, 'b-', label='X', linewidth=2)
axes[1, 2].plot(lags, autocorr_y, 'r-', label='Y', linewidth=2)
axes[1, 2].set_xlabel('Lag (s)')
axes[1, 2].set_ylabel('Autocorrélation')
axes[1, 2].set_title('Autocorrélation des positions (mouvement)')
axes[1, 2].legend()
axes[1, 2].axhline(0, color='gray', linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig9_correlations.png'), dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 10 : Waveforms moyens par groupe
# =============================================================================
print("Génération Figure 10 : Waveforms moyens...")
fig, axes = plt.subplots(1, nGroups, figsize=(20, 5))

np.random.seed(42)
for g in range(nGroups):
    nCh = nChannelsPerGroup[g]
    all_spikes = []
    sample_indices = np.random.choice(len(df), size=300, replace=False)
    for idx in sample_indices:
        raw = df[f'group{g}'].iloc[idx]
        spikes = raw.reshape(-1, nCh, 32)
        all_spikes.append(spikes)

    all_spikes = np.concatenate(all_spikes, axis=0)
    mean_spike = np.mean(all_spikes, axis=0)
    std_spike = np.std(all_spikes, axis=0)

    time_ms = np.arange(32) / 20.0  # ms
    for ch in range(nCh):
        axes[g].plot(time_ms, mean_spike[ch], linewidth=2, label=f'Ch {ch}')
        axes[g].fill_between(time_ms, mean_spike[ch] - std_spike[ch],
                              mean_spike[ch] + std_spike[ch], alpha=0.15)
    axes[g].set_xlabel('Time (ms)')
    axes[g].set_ylabel('Voltage (a.u.)')
    axes[g].set_title(f'Group {g} ({nCh} ch) - Waveform moyen')
    axes[g].legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_average_waveforms.png'), dpi=150, bbox_inches='tight')
plt.close()


print("\n" + "=" * 60)
print("Toutes les figures ont été générées avec succès !")
print("=" * 60)
