"""Plotly figure builders for the U-maze simulation interface."""

import numpy as np
import plotly.graph_objects as go

from maze_geometry import (
    SKELETON_SEGMENTS, compute_corridor_polygon,
    ZONE_NAMES, ZONE_COLORS,
)

# Pre-compute static geometry
_CORRIDOR_X, _CORRIDOR_Y = compute_corridor_polygon()
_THETA = np.linspace(0, 2 * np.pi, 60)

# --- Hacktion Potential color palette ---
_BG_CREAM = '#ffebbf'         # fond principal
_BG_LIGHT = '#fff6e1'         # fond plus clair
_TEXT_DARK = '#733a30'         # texte / titres
_RUST = '#af4635'              # accent principal (predictions)
_TERRACOTTA = '#b55445'        # accent secondaire
_ROSE = '#d67869'              # accent doux
_CORAL = '#cc776a'             # accent muted
_TRUE_BLUE = '#2a6e9e'         # couleur "vrai" (contraste avec le warm)

_HACKTION_LAYOUT = dict(
    plot_bgcolor=_BG_CREAM,
    paper_bgcolor=_BG_LIGHT,
    font=dict(color=_TEXT_DARK),
)


def _make_base_maze() -> go.Figure:
    """Create a figure with the U-maze corridor drawn once."""
    fig = go.Figure()

    # Filled corridor polygon
    fig.add_trace(go.Scatter(
        x=_CORRIDOR_X, y=_CORRIDOR_Y,
        fill='toself', fillcolor='rgba(175,70,53,0.12)',
        line=dict(color=_TEXT_DARK, width=2.5),
        mode='lines',
        showlegend=False, hoverinfo='skip',
    ))

    # Skeleton center line (dashed)
    skel_x, skel_y = [], []
    for x1, y1, x2, y2 in SKELETON_SEGMENTS:
        skel_x.extend([x1, x2, None])
        skel_y.extend([y1, y2, None])
    fig.add_trace(go.Scatter(
        x=skel_x, y=skel_y, mode='lines',
        line=dict(color='rgba(115,58,48,0.2)', width=1, dash='dot'),
        showlegend=False, hoverinfo='skip',
    ))

    return fig


def create_maze_figure(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_sigma: np.ndarray,
    current_idx: int,
    trail_length: int = 50,
    show_1sigma: bool = True,
    show_2sigma: bool = True,
) -> go.Figure:
    """Main U-maze 2D view with trails, current points and uncertainty ellipses."""
    fig = _make_base_maze()
    trail_start = max(0, current_idx - trail_length)

    # --- Trail ---
    if current_idx > trail_start:
        n_trail = current_idx - trail_start
        alphas = np.linspace(0.05, 0.5, n_trail)

        # True positions trail (blue)
        fig.add_trace(go.Scattergl(
            x=y_test[trail_start:current_idx, 0],
            y=y_test[trail_start:current_idx, 1],
            mode='markers',
            marker=dict(
                color=[f'rgba(42,110,158,{a:.2f})' for a in alphas],
                size=3,
            ),
            showlegend=False, hoverinfo='skip',
        ))

        # Predicted positions trail (rust)
        fig.add_trace(go.Scattergl(
            x=y_pred[trail_start:current_idx, 0],
            y=y_pred[trail_start:current_idx, 1],
            mode='markers',
            marker=dict(
                color=[f'rgba(175,70,53,{a:.2f})' for a in alphas],
                size=3, symbol='x',
            ),
            showlegend=False, hoverinfo='skip',
        ))

        # Error vectors
        err_x, err_y = [], []
        for i in range(trail_start, current_idx):
            err_x.extend([y_test[i, 0], y_pred[i, 0], None])
            err_y.extend([y_test[i, 1], y_pred[i, 1], None])
        fig.add_trace(go.Scattergl(
            x=err_x, y=err_y, mode='lines',
            line=dict(color='rgba(115,58,48,0.12)', width=0.7),
            showlegend=False, hoverinfo='skip',
        ))

    # --- Uncertainty ellipses ---
    cx, cy = float(y_pred[current_idx, 0]), float(y_pred[current_idx, 1])
    sx, sy = float(y_sigma[current_idx, 0]), float(y_sigma[current_idx, 1])

    if show_2sigma:
        ex = cx + 2 * sx * np.cos(_THETA)
        ey = cy + 2 * sy * np.sin(_THETA)
        fig.add_trace(go.Scatter(
            x=np.append(ex, ex[0]), y=np.append(ey, ey[0]),
            mode='lines', fill='toself',
            fillcolor='rgba(175,70,53,0.06)',
            line=dict(color='rgba(175,70,53,0.25)', width=1, dash='dot'),
            name='2-sigma',
        ))
    if show_1sigma:
        ex = cx + sx * np.cos(_THETA)
        ey = cy + sy * np.sin(_THETA)
        fig.add_trace(go.Scatter(
            x=np.append(ex, ex[0]), y=np.append(ey, ey[0]),
            mode='lines', fill='toself',
            fillcolor='rgba(175,70,53,0.12)',
            line=dict(color='rgba(175,70,53,0.5)', width=1.5, dash='dash'),
            name='1-sigma',
        ))

    # --- Error line ---
    fig.add_trace(go.Scatter(
        x=[y_test[current_idx, 0], y_pred[current_idx, 0]],
        y=[y_test[current_idx, 1], y_pred[current_idx, 1]],
        mode='lines', line=dict(color=_TEXT_DARK, width=1.5),
        showlegend=False, hoverinfo='skip',
    ))

    # --- Current points ---
    eucl = np.sqrt(((y_test[current_idx] - y_pred[current_idx]) ** 2).sum())
    fig.add_trace(go.Scatter(
        x=[y_test[current_idx, 0]], y=[y_test[current_idx, 1]],
        mode='markers',
        marker=dict(color=_TRUE_BLUE, size=12,
                    line=dict(color=_BG_CREAM, width=1.5)),
        name='Vrai',
    ))
    fig.add_trace(go.Scatter(
        x=[y_pred[current_idx, 0]], y=[y_pred[current_idx, 1]],
        mode='markers',
        marker=dict(color=_RUST, size=12, symbol='x',
                    line=dict(color=_BG_CREAM, width=1)),
        name='Prediction',
    ))

    fig.update_layout(
        **_HACKTION_LAYOUT,
        xaxis=dict(range=[-0.08, 1.08], scaleanchor='y', constrain='domain',
                   showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(range=[-0.08, 1.15], showgrid=False, zeroline=False,
                   fixedrange=True),
        height=520,
        margin=dict(l=30, r=10, t=30, b=50),
        title=dict(text=f't = {current_idx}  |  err = {eucl:.4f}',
                   font=dict(size=13, color=_TEXT_DARK)),
        legend=dict(orientation='h', x=0.0, y=-0.06, font=dict(size=10, color=_TEXT_DARK),
                    bgcolor='rgba(0,0,0,0)'),
    )
    return fig


# ================================================================
# Final summary charts (shown once at the end)
# ================================================================

def create_mse_components(
    se_x: np.ndarray, se_y: np.ndarray, se_pos: np.ndarray,
    cum_mse_x: np.ndarray, cum_mse_y: np.ndarray, cum_mse_pos: np.ndarray,
    current_idx: int,
) -> go.Figure:
    """3 cumulative MSE curves: MSE(x), MSE(y), MSE(position)."""
    idx = current_idx + 1
    x_ax = np.arange(idx)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x_ax, y=cum_mse_x[:idx],
        mode='lines', line=dict(color=_TRUE_BLUE, width=1.3),
        name='MSE x',
    ))
    fig.add_trace(go.Scattergl(
        x=x_ax, y=cum_mse_y[:idx],
        mode='lines', line=dict(color=_RUST, width=1.3),
        name='MSE y',
    ))
    fig.add_trace(go.Scattergl(
        x=x_ax, y=cum_mse_pos[:idx],
        mode='lines', line=dict(color=_TEXT_DARK, width=1.8),
        name='MSE pos',
    ))
    ymax = max(cum_mse_pos[0] * 1.1, cum_mse_pos[:idx].max() * 1.2) if idx > 0 else 1
    fig.update_layout(
        **_HACKTION_LAYOUT,
        xaxis=dict(range=[0, len(se_x)]),
        yaxis=dict(range=[0, ymax], title='MSE'),
        height=300, margin=dict(l=50, r=10, t=35, b=30),
        title=dict(text='MSE cumulative (x, y, pos)', font=dict(size=12, color=_TEXT_DARK)),
        legend=dict(orientation='h', x=0, y=1.15, font=dict(size=9, color=_TEXT_DARK),
                    bgcolor='rgba(0,0,0,0)'),
    )
    return fig


def create_mean_eucl_error(
    cum_mean_eucl: np.ndarray, n_total: int, current_idx: int,
) -> go.Figure:
    """Cumulative mean euclidean error."""
    idx = current_idx + 1
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=np.arange(idx), y=cum_mean_eucl[:idx],
        mode='lines', line=dict(color=_ROSE, width=1.5),
    ))
    val = cum_mean_eucl[current_idx]
    fig.add_annotation(
        x=current_idx, y=val,
        text=f'{val:.4f}', showarrow=False,
        font=dict(size=11, color=_TERRACOTTA),
        yshift=12,
    )
    fig.update_layout(
        **_HACKTION_LAYOUT,
        xaxis=dict(range=[0, n_total]),
        yaxis=dict(title='Erreur eucl. moy.'),
        height=300, margin=dict(l=50, r=10, t=35, b=30),
        title=dict(text='Erreur euclidienne moyenne', font=dict(size=12, color=_TEXT_DARK)),
        showlegend=False,
    )
    return fig


def create_zone_accuracy(
    cum_zone_acc: np.ndarray, n_total: int, current_idx: int,
) -> go.Figure:
    """Cumulative zone classification accuracy."""
    idx = current_idx + 1
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=np.arange(idx), y=cum_zone_acc[:idx],
        mode='lines', line=dict(color=_CORAL, width=1.5),
    ))
    val = cum_zone_acc[current_idx]
    fig.add_annotation(
        x=current_idx, y=val,
        text=f'{val:.1%}', showarrow=False,
        font=dict(size=11, color=_TERRACOTTA),
        yshift=12,
    )
    fig.update_layout(
        **_HACKTION_LAYOUT,
        xaxis=dict(range=[0, n_total]),
        yaxis=dict(range=[0, 1.05], title='Accuracy'),
        height=300, margin=dict(l=50, r=10, t=35, b=30),
        title=dict(text='Accuracy zone (cumulative)', font=dict(size=12, color=_TEXT_DARK)),
        showlegend=False,
    )
    return fig


def create_trajectory_aggregate(
    y_test: np.ndarray, y_pred: np.ndarray, current_idx: int,
) -> go.Figure:
    """Scatter of all true vs predicted positions."""
    fig = _make_base_maze()
    idx = current_idx + 1

    fig.add_trace(go.Scattergl(
        x=y_test[:idx, 0], y=y_test[:idx, 1],
        mode='markers',
        marker=dict(color=f'rgba(42,110,158,0.25)', size=3),
        name='Vrai',
    ))
    fig.add_trace(go.Scattergl(
        x=y_pred[:idx, 0], y=y_pred[:idx, 1],
        mode='markers',
        marker=dict(color=f'rgba(175,70,53,0.25)', size=3, symbol='x'),
        name='Prediction',
    ))

    fig.update_layout(
        **_HACKTION_LAYOUT,
        xaxis=dict(range=[-0.08, 1.08], scaleanchor='y', constrain='domain',
                   showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.08, 1.15], showgrid=False, zeroline=False),
        height=400,
        margin=dict(l=30, r=10, t=35, b=40),
        title=dict(text=f'Trajectoire agregee ({idx} pts)', font=dict(size=12, color=_TEXT_DARK)),
        legend=dict(orientation='h', x=0, y=-0.06, font=dict(size=9, color=_TEXT_DARK),
                    bgcolor='rgba(0,0,0,0)'),
    )
    return fig
