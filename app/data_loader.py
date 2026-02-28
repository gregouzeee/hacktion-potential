"""Load prediction outputs and precompute all cumulative metrics."""

import os
import numpy as np
import streamlit as st


@st.cache_data
def load_data(output_dir: str) -> dict:
    """Load .npy files and precompute cumulative arrays for real-time display.

    Precomputed arrays (length N) allow O(1) lookup at any timestep
    instead of recomputing aggregates each frame.
    """
    def _load(name):
        return np.load(os.path.join(output_dir, name))

    data = {
        'y_pred':    _load('preds_transformer_02i.npy'),
        'y_sigma':   _load('sigma_transformer_02i.npy'),
        'y_test':    _load('y_test_transformer_02i.npy'),
        'd_pred':    _load('d_pred_transformer_02i.npy'),
        'd_test':    _load('d_test_transformer_02i.npy'),
        'zone_pred': _load('zone_pred_transformer_02i.npy'),
        'zone_test': _load('zone_test_transformer_02i.npy'),
        'probs':     _load('probs_transformer_02i.npy'),
    }
    N = len(data['y_pred'])
    data['n_points'] = N

    # --- Per-point errors ---
    diff = data['y_test'] - data['y_pred']
    data['se_x'] = diff[:, 0] ** 2                          # squared error X
    data['se_y'] = diff[:, 1] ** 2                          # squared error Y
    data['se_pos'] = data['se_x'] + data['se_y']            # squared error position
    data['eucl_errors'] = np.sqrt(data['se_pos'])            # euclidean error

    # --- Cumulative MSE (running mean of squared errors) ---
    counts = np.arange(1, N + 1, dtype=np.float64)
    data['cum_mse_x'] = np.cumsum(data['se_x']) / counts
    data['cum_mse_y'] = np.cumsum(data['se_y']) / counts
    data['cum_mse_pos'] = np.cumsum(data['se_pos']) / counts

    # --- Cumulative zone accuracy ---
    zone_correct = (data['zone_pred'] == data['zone_test']).astype(np.float64)
    data['cum_zone_acc'] = np.cumsum(zone_correct) / counts

    # --- Cumulative mean euclidean error ---
    data['cum_mean_eucl'] = np.cumsum(data['eucl_errors']) / counts

    return data
