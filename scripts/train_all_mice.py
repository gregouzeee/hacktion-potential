"""
Train le modèle 02i (SpikeTransformerHierarchical) sur tous les fichiers win108
disponibles dans le dossier data/, sauf M1199 (déjà fait).

Usage (depuis la racine du projet) :
    python scripts/train_all_mice.py

Les modèles et prédictions sont sauvegardés dans outputs_mice/<MOUSE_ID>/
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import glob
import math
import traceback
from datetime import datetime

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_BASE = os.path.join(PROJECT_DIR, 'outputs_mice')

SKIP_MICE = {'M1199'}  # déjà fait, on déplacera les outputs à la main

# Hyperparamètres (identiques au notebook 02i)
EMBED_DIM = 64
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.2
SPIKE_DROPOUT = 0.15
NOISE_STD = 0.5
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 7
LAMBDA_D = 1.0
LAMBDA_FEAS = 10.0
N_FOLDS = 5
BATCH_SIZE = 64
MAX_SEQ_LEN = 128

# Device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

# ============================================================
# Squelette du U-maze et fonctions curvilignes
# ============================================================
SKELETON_SEGMENTS = np.array([
    [0.15, 0.0, 0.15, 0.90],
    [0.15, 0.90, 0.85, 0.90],
    [0.85, 0.90, 0.85, 0.0],
])
CORRIDOR_HALF_WIDTH = 0.15
SEGMENT_LENGTHS = np.array([
    np.sqrt((s[2]-s[0])**2 + (s[3]-s[1])**2) for s in SKELETON_SEGMENTS
])
TOTAL_LENGTH = SEGMENT_LENGTHS.sum()
CUMULATIVE_LENGTHS = np.concatenate([[0], np.cumsum(SEGMENT_LENGTHS)])
D_LEFT_END = CUMULATIVE_LENGTHS[1] / TOTAL_LENGTH
D_RIGHT_START = CUMULATIVE_LENGTHS[2] / TOTAL_LENGTH
N_ZONES = 3
ZONE_NAMES = ['Gauche', 'Haut', 'Droite']


def project_point_on_segment(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx**2 + dy**2
    if seg_len_sq < 1e-12:
        return 0.0, np.sqrt((px - x1)**2 + (py - y1)**2), x1, y1
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    return t, dist, proj_x, proj_y


def compute_curvilinear_distance(x, y):
    best_dist = np.inf
    best_d = 0.0
    for i, (x1, y1, x2, y2) in enumerate(SKELETON_SEGMENTS):
        t, dist, _, _ = project_point_on_segment(x, y, x1, y1, x2, y2)
        if dist < best_dist:
            best_dist = dist
            best_d = (CUMULATIVE_LENGTHS[i] + t * SEGMENT_LENGTHS[i]) / TOTAL_LENGTH
    return best_d


def compute_distance_to_skeleton(x, y):
    best_dist = np.inf
    for x1, y1, x2, y2 in SKELETON_SEGMENTS:
        _, dist, _, _ = project_point_on_segment(x, y, x1, y1, x2, y2)
        best_dist = min(best_dist, dist)
    return best_dist


def d_to_zone(d):
    if d < D_LEFT_END:
        return 0
    elif d < D_RIGHT_START:
        return 1
    else:
        return 2


# ============================================================
# Preprocessing
# ============================================================
def reconstruct_sequence(row, nGroups, nChannelsPerGroup, max_seq_len=MAX_SEQ_LEN):
    groups = row['groups']
    length = min(len(groups), max_seq_len)
    waveforms = {}
    for g in range(nGroups):
        nCh = nChannelsPerGroup[g]
        raw = row[f'group{g}']
        waveforms[g] = raw.reshape(-1, nCh, 32)
    seq_waveforms = []
    seq_shank_ids = []
    for t in range(length):
        g = int(groups[t])
        idx = int(row[f'indices{g}'][t])
        if idx > 0 and idx <= waveforms[g].shape[0]:
            seq_waveforms.append((waveforms[g][idx - 1], g))
            seq_shank_ids.append(g)
    return seq_waveforms, seq_shank_ids


# ============================================================
# Dataset
# ============================================================
class SpikeSequenceDataset(Dataset):
    def __init__(self, dataframe, nGroups, nChannelsPerGroup, curvilinear_d, zone_labels,
                 max_channels, max_seq_len=MAX_SEQ_LEN):
        self.df = dataframe
        self.nGroups = nGroups
        self.nChannelsPerGroup = nChannelsPerGroup
        self.max_seq_len = max_seq_len
        self.max_channels = max_channels
        self.targets = np.array([[x[0], x[1]] for x in dataframe['pos']], dtype=np.float32)
        self.curvilinear_d = curvilinear_d.astype(np.float32)
        self.zone_labels = zone_labels.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq, shank_ids = reconstruct_sequence(row, self.nGroups, self.nChannelsPerGroup, self.max_seq_len)
        seq_len = len(seq)
        if seq_len == 0:
            seq_len = 1
            waveforms = np.zeros((1, self.max_channels, 32), dtype=np.float32)
            shank_ids_arr = np.array([0], dtype=np.int64)
        else:
            waveforms = np.zeros((seq_len, self.max_channels, 32), dtype=np.float32)
            shank_ids_arr = np.array(shank_ids, dtype=np.int64)
            for t, (wf, g) in enumerate(seq):
                nCh = wf.shape[0]
                waveforms[t, :nCh, :] = wf
        return {
            'waveforms': torch.from_numpy(waveforms),
            'shank_ids': torch.from_numpy(shank_ids_arr),
            'seq_len': seq_len,
            'target': torch.from_numpy(self.targets[idx]),
            'd': torch.tensor(self.curvilinear_d[idx], dtype=torch.float32),
            'zone': torch.tensor(self.zone_labels[idx], dtype=torch.long)
        }


def make_collate_fn(max_channels):
    def collate_fn(batch):
        max_len = max(item['seq_len'] for item in batch)
        batch_size = len(batch)
        waveforms = torch.zeros(batch_size, max_len, max_channels, 32)
        shank_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        mask = torch.ones(batch_size, max_len, dtype=torch.bool)
        targets = torch.stack([item['target'] for item in batch])
        d_targets = torch.stack([item['d'] for item in batch])
        zone_targets = torch.stack([item['zone'] for item in batch])
        for i, item in enumerate(batch):
            sl = item['seq_len']
            waveforms[i, :sl] = item['waveforms']
            shank_ids[i, :sl] = item['shank_ids']
            mask[i, :sl] = False
        return {
            'waveforms': waveforms, 'shank_ids': shank_ids, 'mask': mask,
            'targets': targets, 'd_targets': d_targets, 'zone_targets': zone_targets
        }
    return collate_fn


# ============================================================
# Modèle
# ============================================================
class SpikeEncoder(nn.Module):
    def __init__(self, n_channels, embed_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, embed_dim, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x):
        return self.conv(x).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FeasibilityLoss(nn.Module):
    def __init__(self, skeleton_segments, corridor_half_width):
        super().__init__()
        self.register_buffer('segments', torch.tensor(skeleton_segments, dtype=torch.float32))
        self.corridor_half_width = corridor_half_width

    def forward(self, xy_pred):
        px, py = xy_pred[:, 0], xy_pred[:, 1]
        distances = []
        for i in range(self.segments.shape[0]):
            x1, y1, x2, y2 = self.segments[i]
            dx, dy = x2 - x1, y2 - y1
            seg_len_sq = dx**2 + dy**2
            t = ((px - x1) * dx + (py - y1) * dy) / (seg_len_sq + 1e-8)
            t = t.clamp(0.0, 1.0)
            proj_x, proj_y = x1 + t * dx, y1 + t * dy
            dist = torch.sqrt((px - proj_x)**2 + (py - proj_y)**2 + 1e-8)
            distances.append(dist)
        distances = torch.stack(distances, dim=1)
        min_dist = distances.min(dim=1).values
        return torch.relu(min_dist - self.corridor_half_width).pow(2).mean()


class SpikeTransformerHierarchical(nn.Module):
    def __init__(self, nGroups, nChannelsPerGroup, n_zones=3, embed_dim=64, nhead=4,
                 num_layers=2, dropout=0.2, spike_dropout=0.15, noise_std=0.5,
                 max_channels=6):
        super().__init__()
        self.nGroups = nGroups
        self.embed_dim = embed_dim
        self.n_zones = n_zones
        self.max_channels = max_channels
        self.spike_dropout = spike_dropout
        self.noise_std = noise_std

        self.spike_encoders = nn.ModuleList([
            SpikeEncoder(max_channels, embed_dim) for _ in range(nGroups)
        ])
        self.shank_embedding = nn.Embedding(nGroups, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, n_zones)
        )
        self.mu_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(embed_dim, 2)
            ) for _ in range(n_zones)
        ])
        self.log_sigma_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(embed_dim, 2)
            ) for _ in range(n_zones)
        ])
        self.d_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, 1), nn.Sigmoid()
        )

    def _apply_spike_dropout(self, mask):
        if not self.training or self.spike_dropout <= 0:
            return mask
        drop_mask = torch.rand_like(mask.float()) < self.spike_dropout
        active = ~mask
        new_drops = drop_mask & active
        remaining = active & ~new_drops
        all_dropped = remaining.sum(dim=1) == 0
        if all_dropped.any():
            new_drops[all_dropped] = False
        return mask | new_drops

    def _apply_waveform_noise(self, waveforms):
        if not self.training or self.noise_std <= 0:
            return waveforms
        return waveforms + torch.randn_like(waveforms) * self.noise_std

    def _encode(self, waveforms, shank_ids, mask):
        batch_size, seq_len = waveforms.shape[:2]
        mask = self._apply_spike_dropout(mask)
        waveforms = self._apply_waveform_noise(waveforms)
        embeddings = torch.zeros(batch_size, seq_len, self.embed_dim, device=waveforms.device)
        for g in range(self.nGroups):
            group_mask = (shank_ids == g) & (~mask)
            if group_mask.any():
                embeddings[group_mask] = self.spike_encoders[g](waveforms[group_mask])
        embeddings = embeddings + self.shank_embedding(shank_ids)
        embeddings = self.pos_encoding(embeddings)
        encoded = self.transformer(embeddings, src_key_padding_mask=mask)
        active_mask = (~mask).unsqueeze(-1).float()
        pooled = (encoded * active_mask).sum(dim=1) / (active_mask.sum(dim=1) + 1e-8)
        return pooled

    def forward(self, waveforms, shank_ids, mask):
        pooled = self._encode(waveforms, shank_ids, mask)
        cls_logits = self.cls_head(pooled)
        mus = [head(pooled) for head in self.mu_heads]
        sigmas = [torch.exp(head(pooled)) for head in self.log_sigma_heads]
        d_pred = self.d_head(pooled)
        return cls_logits, mus, sigmas, d_pred

    def predict(self, waveforms, shank_ids, mask):
        cls_logits, mus, sigmas, d_pred = self.forward(waveforms, shank_ids, mask)
        probs = torch.softmax(cls_logits, dim=1)
        mu_stack = torch.stack(mus, dim=1)
        sigma_stack = torch.stack(sigmas, dim=1)
        p = probs.unsqueeze(-1)
        mu = (p * mu_stack).sum(dim=1)
        var_combined = (p * (sigma_stack ** 2 + mu_stack ** 2)).sum(dim=1) - mu ** 2
        sigma = torch.sqrt(var_combined.clamp(min=1e-8))
        return mu, sigma, probs, d_pred


# ============================================================
# Training / Eval
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, criterion_ce, criterion_nll,
                criterion_d, feas_loss, device):
    model.train()
    totals = {'loss': 0, 'cls': 0, 'pos': 0, 'd': 0, 'feas': 0, 'correct': 0, 'n': 0, 'batches': 0}
    for batch in loader:
        wf = batch['waveforms'].to(device)
        sid = batch['shank_ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)
        d_targets = batch['d_targets'].to(device)
        zone_targets = batch['zone_targets'].to(device)

        optimizer.zero_grad()
        cls_logits, mus, sigmas, d_pred = model(wf, sid, mask)

        loss_cls = criterion_ce(cls_logits, zone_targets)
        loss_pos = torch.tensor(0.0, device=device)
        for z in range(N_ZONES):
            zmask = (zone_targets == z)
            if zmask.any():
                loss_pos = loss_pos + criterion_nll(
                    mus[z][zmask], targets[zmask], sigmas[z][zmask] ** 2
                )
        loss_d = criterion_d(d_pred.squeeze(-1), d_targets)
        probs = torch.softmax(cls_logits, dim=1).unsqueeze(-1)
        mu_stack = torch.stack(mus, dim=1)
        mu_combined = (probs * mu_stack).sum(dim=1)
        loss_feas = feas_loss(mu_combined)

        loss = loss_cls + loss_pos + LAMBDA_D * loss_d + LAMBDA_FEAS * loss_feas
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        totals['loss'] += loss.item()
        totals['cls'] += loss_cls.item()
        totals['pos'] += loss_pos.item()
        totals['d'] += loss_d.item()
        totals['feas'] += loss_feas.item()
        with torch.no_grad():
            totals['correct'] += (cls_logits.argmax(dim=1) == zone_targets).sum().item()
            totals['n'] += len(zone_targets)
        totals['batches'] += 1

    nb = totals['batches']
    return {k: totals[k] / nb for k in ['loss', 'cls', 'pos', 'd', 'feas']}, totals['correct'] / totals['n']


@torch.no_grad()
def eval_epoch(model, loader, criterion_ce, criterion_nll, criterion_d, feas_loss, device):
    model.eval()
    totals = {'loss': 0, 'cls': 0, 'pos': 0, 'd': 0, 'feas': 0, 'correct': 0, 'n': 0, 'batches': 0}
    all_mu, all_sigma, all_probs, all_d = [], [], [], []
    all_targets, all_d_targets, all_zone_targets = [], [], []

    for batch in loader:
        wf = batch['waveforms'].to(device)
        sid = batch['shank_ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)
        d_targets = batch['d_targets'].to(device)
        zone_targets = batch['zone_targets'].to(device)

        mu, sigma, probs, d_pred = model.predict(wf, sid, mask)
        cls_logits, mus, sigmas_z, _ = model(wf, sid, mask)

        loss_cls = criterion_ce(cls_logits, zone_targets)
        loss_pos = torch.tensor(0.0, device=device)
        for z in range(N_ZONES):
            zmask = (zone_targets == z)
            if zmask.any():
                loss_pos = loss_pos + criterion_nll(mus[z][zmask], targets[zmask], sigmas_z[z][zmask] ** 2)
        loss_d = criterion_d(d_pred.squeeze(-1), d_targets)
        loss_feas = feas_loss(mu)
        loss = loss_cls + loss_pos + LAMBDA_D * loss_d + LAMBDA_FEAS * loss_feas

        totals['loss'] += loss.item(); totals['cls'] += loss_cls.item()
        totals['pos'] += loss_pos.item(); totals['d'] += loss_d.item()
        totals['feas'] += loss_feas.item()
        totals['correct'] += (cls_logits.argmax(dim=1) == zone_targets).sum().item()
        totals['n'] += len(zone_targets); totals['batches'] += 1

        all_mu.append(mu.cpu().numpy()); all_sigma.append(sigma.cpu().numpy())
        all_probs.append(probs.cpu().numpy()); all_d.append(d_pred.cpu().numpy())
        all_targets.append(targets.cpu().numpy()); all_d_targets.append(d_targets.cpu().numpy())
        all_zone_targets.append(zone_targets.cpu().numpy())

    nb = totals['batches']
    losses = {k: totals[k] / nb for k in ['loss', 'cls', 'pos', 'd', 'feas']}
    acc = totals['correct'] / totals['n']
    arrays = [np.concatenate(a) for a in [all_mu, all_sigma, all_probs, all_d,
                                           all_targets, all_d_targets, all_zone_targets]]
    return losses, acc, arrays


def reproject_to_skeleton(xy, skeleton_segments, corridor_half_width):
    xy_fixed = xy.copy()
    n_reprojected = 0
    for i in range(len(xy)):
        px, py = xy[i, 0], xy[i, 1]
        best_dist = np.inf
        best_proj = (px, py)
        for x1, y1, x2, y2 in skeleton_segments:
            _, dist, proj_x, proj_y = project_point_on_segment(px, py, x1, y1, x2, y2)
            if dist < best_dist:
                best_dist = dist
                best_proj = (proj_x, proj_y)
        if best_dist > corridor_half_width:
            proj_x, proj_y = best_proj
            dx, dy = px - proj_x, py - proj_y
            norm = np.sqrt(dx**2 + dy**2) + 1e-8
            xy_fixed[i, 0] = proj_x + dx / norm * corridor_half_width
            xy_fixed[i, 1] = proj_y + dy / norm * corridor_half_width
            n_reprojected += 1
    return xy_fixed, n_reprojected


# ============================================================
# Découverte des fichiers
# ============================================================
def discover_datasets(data_dir):
    """Trouve les paires (parquet win108, json) dans data_dir.
    Retourne une liste de dicts {mouse_id, parquet, json}."""
    parquets = sorted(glob.glob(os.path.join(data_dir, '*_win108_*.parquet')))
    jsons = sorted(glob.glob(os.path.join(data_dir, '*.json')))

    datasets = []
    for pq in parquets:
        pq_name = os.path.basename(pq)
        # Ex: M1199_PAG_stride4_win108_test.parquet -> mouse_id = M1199
        mouse_id = pq_name.split('_')[0]

        # Trouver le JSON correspondant : celui qui commence par le même mouse_id
        matching_jsons = [j for j in jsons if os.path.basename(j).startswith(mouse_id + '_')]
        if not matching_jsons:
            print(f'  [WARN] Pas de JSON trouvé pour {pq_name}, skip')
            continue

        # S'il y en a plusieurs, prendre celui qui contient la même region (PAG, MFB, Known...)
        # Region = 2eme partie du nom du parquet
        region = pq_name.split('_')[1]  # PAG, MFB, Known
        best_json = matching_jsons[0]
        for j in matching_jsons:
            if region in os.path.basename(j):
                best_json = j
                break

        datasets.append({
            'mouse_id': mouse_id,
            'region': region,
            'parquet': pq,
            'json': best_json,
        })

    return datasets


# ============================================================
# Pipeline pour une souris
# ============================================================
def train_mouse(dataset_info):
    mouse_id = dataset_info['mouse_id']
    region = dataset_info['region']
    tag = f'{mouse_id}_{region}'
    parquet_file = dataset_info['parquet']
    json_file = dataset_info['json']

    output_dir = os.path.join(OUTPUT_BASE, mouse_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f'\n{"#"*60}')
    print(f'# {tag}')
    print(f'# Parquet : {os.path.basename(parquet_file)}')
    print(f'# JSON    : {os.path.basename(json_file)}')
    print(f'# Output  : {output_dir}')
    print(f'{"#"*60}')

    # --- Chargement ---
    df = pd.read_parquet(parquet_file)
    with open(json_file, 'r') as f:
        params = json.load(f)

    nGroups = params['nGroups']
    nChannelsPerGroup = [params[f'group{g}']['nChannels'] for g in range(nGroups)]
    max_channels = max(nChannelsPerGroup)
    print(f'  Shape: {df.shape}, nGroups={nGroups}, nChannelsPerGroup={nChannelsPerGroup}')

    # --- Filtrage speedMask ---
    speed_masks = np.array([x[0] for x in df['speedMask']])
    df_moving = df[speed_masks].reset_index(drop=True)
    print(f'  Exemples en mouvement : {len(df_moving)}')

    if len(df_moving) < 100:
        print(f'  [SKIP] Trop peu d\'exemples en mouvement ({len(df_moving)})')
        return

    # --- Curvilinear distance + zones ---
    positions = np.array([[x[0], x[1]] for x in df_moving['pos']], dtype=np.float32)
    curvilinear_d = np.array([compute_curvilinear_distance(x, y) for x, y in positions], dtype=np.float32)
    zone_labels = np.array([d_to_zone(d) for d in curvilinear_d], dtype=np.int64)

    print(f'  d curviligne : min={curvilinear_d.min():.4f}, max={curvilinear_d.max():.4f}')
    for z in range(N_ZONES):
        count = (zone_labels == z).sum()
        print(f'    {ZONE_NAMES[z]:8s} : {count} ({count / len(zone_labels):.1%})')

    # --- Split 90/10 ---
    split_idx = int(len(df_moving) * 0.9)
    df_train_full = df_moving.iloc[:split_idx].reset_index(drop=True)
    df_test = df_moving.iloc[split_idx:].reset_index(drop=True)
    d_train_full = curvilinear_d[:split_idx]
    d_test = curvilinear_d[split_idx:]
    zone_train_full = zone_labels[:split_idx]
    zone_test = zone_labels[split_idx:]

    print(f'  Train: {len(df_train_full)} | Test: {len(df_test)}')

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=41)
    collate = make_collate_fn(max_channels)

    test_dataset = SpikeSequenceDataset(df_test, nGroups, nChannelsPerGroup, d_test, zone_test, max_channels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)

    # --- Entraînement K-Fold ---
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train_full)):
        print(f'\n  --- Fold {fold+1}/{N_FOLDS} ---')
        df_ft = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_fv = df_train_full.iloc[val_idx].reset_index(drop=True)

        ds_t = SpikeSequenceDataset(df_ft, nGroups, nChannelsPerGroup, d_train_full[train_idx], zone_train_full[train_idx], max_channels)
        ds_v = SpikeSequenceDataset(df_fv, nGroups, nChannelsPerGroup, d_train_full[val_idx], zone_train_full[val_idx], max_channels)
        dl_t = DataLoader(ds_t, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=0)
        dl_v = DataLoader(ds_v, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)

        print(f'    Train: {len(ds_t)}, Val: {len(ds_v)}')

        model = SpikeTransformerHierarchical(
            nGroups, nChannelsPerGroup, n_zones=N_ZONES,
            embed_dim=EMBED_DIM, nhead=NHEAD, num_layers=NUM_LAYERS,
            dropout=DROPOUT, spike_dropout=SPIKE_DROPOUT, noise_std=NOISE_STD,
            max_channels=max_channels
        ).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(dl_t))
        criterion_ce = nn.CrossEntropyLoss()
        criterion_nll = nn.GaussianNLLLoss()
        criterion_d = nn.MSELoss()
        feas_loss_fn = FeasibilityLoss(SKELETON_SEGMENTS, CORRIDOR_HALF_WIDTH).to(DEVICE)

        best_val_loss = float('inf')
        patience_counter = 0
        model_path = os.path.join(output_dir, f'best_transformer_02i_fold{fold+1}.pt')

        for epoch in range(EPOCHS):
            t_losses, t_acc = train_epoch(model, dl_t, optimizer, scheduler, criterion_ce, criterion_nll, criterion_d, feas_loss_fn, DEVICE)
            v_losses, v_acc, _ = eval_epoch(model, dl_v, criterion_ce, criterion_nll, criterion_d, feas_loss_fn, DEVICE)

            if epoch % 5 == 0 or epoch == EPOCHS - 1:
                print(f'    Epoch {epoch+1:02d}/{EPOCHS} | Train: {t_losses["loss"]:.4f} (acc={t_acc:.1%}) | Val: {v_losses["loss"]:.4f} (acc={v_acc:.1%})')

            if v_losses['loss'] < best_val_loss:
                best_val_loss = v_losses['loss']
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f'    Early stopping a epoch {epoch+1}')
                    break

        # Eval fold
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        _, val_acc, (val_mu, val_sigma, val_probs, val_d_pred, val_targets, val_d_targets, val_zone_targets) = eval_epoch(
            model, dl_v, criterion_ce, criterion_nll, criterion_d, feas_loss_fn, DEVICE
        )
        val_eucl = np.sqrt(((val_targets - val_mu) ** 2).sum(axis=1))

        fold_results.append({
            'fold': fold + 1, 'best_val_loss': best_val_loss,
            'val_eucl_mean': val_eucl.mean(),
            'val_r2_x': r2_score(val_targets[:, 0], val_mu[:, 0]),
            'val_r2_y': r2_score(val_targets[:, 1], val_mu[:, 1]),
            'val_cls_acc': val_acc,
        })
        print(f'    => Eucl={val_eucl.mean():.4f} | R2: X={fold_results[-1]["val_r2_x"]:.4f} Y={fold_results[-1]["val_r2_y"]:.4f} | cls={val_acc:.1%}')

    # --- Évaluation ensemble sur test ---
    print(f'\n  === Évaluation ensemble sur test ===')
    all_fold_mu, all_fold_sigma, all_fold_probs, all_fold_d = [], [], [], []

    for fold in range(N_FOLDS):
        model = SpikeTransformerHierarchical(
            nGroups, nChannelsPerGroup, n_zones=N_ZONES,
            embed_dim=EMBED_DIM, nhead=NHEAD, num_layers=NUM_LAYERS,
            dropout=DROPOUT, spike_dropout=SPIKE_DROPOUT, noise_std=NOISE_STD,
            max_channels=max_channels
        ).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(output_dir, f'best_transformer_02i_fold{fold+1}.pt'),
                                         map_location=DEVICE, weights_only=True))
        _, fold_acc, (fold_mu, fold_sigma, fold_probs, fold_d, y_test, d_test_targets, zone_test_targets) = eval_epoch(
            model, test_loader, criterion_ce, criterion_nll, criterion_d, feas_loss_fn, DEVICE
        )
        all_fold_mu.append(fold_mu); all_fold_sigma.append(fold_sigma)
        all_fold_probs.append(fold_probs); all_fold_d.append(fold_d)

    all_fold_mu = np.stack(all_fold_mu)
    all_fold_sigma = np.stack(all_fold_sigma)
    all_fold_probs = np.stack(all_fold_probs)
    all_fold_d = np.stack(all_fold_d)

    y_pred_raw = all_fold_mu.mean(axis=0)
    d_pred_ensemble = all_fold_d.mean(axis=0).squeeze()
    probs_ensemble = all_fold_probs.mean(axis=0)
    zone_pred = probs_ensemble.argmax(axis=1)

    mean_var = (all_fold_sigma ** 2).mean(axis=0)
    var_mu = all_fold_mu.var(axis=0)
    y_sigma = np.sqrt(mean_var + var_mu)

    y_pred, n_reproj = reproject_to_skeleton(y_pred_raw, SKELETON_SEGMENTS, CORRIDOR_HALF_WIDTH)
    print(f'  Reprojection : {n_reproj} points ({n_reproj / len(y_pred):.1%})')

    eucl_errors = np.sqrt(((y_test - y_pred) ** 2).sum(axis=1))
    r2_x = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_y = r2_score(y_test[:, 1], y_pred[:, 1])
    d_mae = np.abs(d_test_targets - d_pred_ensemble).mean()
    cls_accuracy = (zone_pred == zone_test_targets).mean()

    print(f'  Eucl : mean={eucl_errors.mean():.4f}, median={np.median(eucl_errors):.4f}')
    print(f'  R²   : X={r2_x:.4f}, Y={r2_y:.4f}')
    print(f'  d MAE: {d_mae:.4f} | cls acc: {cls_accuracy:.1%}')

    # --- Sauvegarde ---
    np.save(os.path.join(output_dir, 'preds_transformer_02i.npy'), y_pred)
    np.save(os.path.join(output_dir, 'sigma_transformer_02i.npy'), y_sigma)
    np.save(os.path.join(output_dir, 'd_pred_transformer_02i.npy'), d_pred_ensemble)
    np.save(os.path.join(output_dir, 'y_test_transformer_02i.npy'), y_test)
    np.save(os.path.join(output_dir, 'd_test_transformer_02i.npy'), d_test_targets)
    np.save(os.path.join(output_dir, 'zone_pred_transformer_02i.npy'), zone_pred)
    np.save(os.path.join(output_dir, 'zone_test_transformer_02i.npy'), zone_test_targets)
    np.save(os.path.join(output_dir, 'probs_transformer_02i.npy'), probs_ensemble)

    # Résumé JSON
    summary = {
        'mouse_id': mouse_id,
        'region': region,
        'parquet': os.path.basename(parquet_file),
        'json': os.path.basename(json_file),
        'n_moving': len(df_moving),
        'n_train': len(df_train_full),
        'n_test': len(df_test),
        'nGroups': nGroups,
        'nChannelsPerGroup': nChannelsPerGroup,
        'eucl_mean': float(eucl_errors.mean()),
        'eucl_median': float(np.median(eucl_errors)),
        'r2_x': float(r2_x),
        'r2_y': float(r2_y),
        'd_mae': float(d_mae),
        'cls_accuracy': float(cls_accuracy),
        'n_reprojected': n_reproj,
        'fold_results': fold_results,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n  Sauvegarde terminée dans {output_dir}/')
    return summary


# ============================================================
# Main
# ============================================================
def main():
    print(f'=== Train 02i sur toutes les souris ===')
    print(f'Device: {DEVICE}')
    print(f'Data dir: {DATA_DIR}')
    print(f'Output dir: {OUTPUT_BASE}')
    print(f'Skip: {SKIP_MICE}')
    print()

    datasets = discover_datasets(DATA_DIR)
    if not datasets:
        print('Aucun dataset trouvé ! Vérifiez que les fichiers parquet win108 sont dans data/')
        sys.exit(1)

    print(f'Datasets trouvés : {len(datasets)}')
    for ds in datasets:
        skip = ' [SKIP]' if ds['mouse_id'] in SKIP_MICE else ''
        print(f'  {ds["mouse_id"]}_{ds["region"]} : {os.path.basename(ds["parquet"])}{skip}')
    print()

    results = []
    for ds in datasets:
        if ds['mouse_id'] in SKIP_MICE:
            print(f'\n[SKIP] {ds["mouse_id"]} (dans SKIP_MICE)')
            continue
        try:
            summary = train_mouse(ds)
            results.append(summary)
        except Exception as e:
            print(f'\n[ERREUR] {ds["mouse_id"]}_{ds["region"]} : {e}')
            traceback.print_exc()
            print('Passage à la souris suivante...\n')
            continue

    # Résumé final
    print(f'\n{"="*60}')
    print(f'RÉSUMÉ FINAL')
    print(f'{"="*60}')
    for r in results:
        print(f'  {r["mouse_id"]}_{r["region"]} : Eucl={r["eucl_mean"]:.4f} | R2: X={r["r2_x"]:.4f} Y={r["r2_y"]:.4f} | cls={r["cls_accuracy"]:.1%}')

    if not results:
        print('  Aucune souris entraînée avec succès.')

    # Sauvegarder le résumé global
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    with open(os.path.join(OUTPUT_BASE, 'summary_all.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nRésumé global sauvegardé dans {OUTPUT_BASE}/summary_all.json')


if __name__ == '__main__':
    main()