"""U-maze skeleton geometry and corridor polygon."""

import numpy as np

SKELETON_SEGMENTS = np.array([
    [0.15, 0.0, 0.15, 0.90],   # Left arm (bottom to top)
    [0.15, 0.90, 0.85, 0.90],  # Top corridor (left to right)
    [0.85, 0.90, 0.85, 0.0],   # Right arm (top to bottom)
])
CORRIDOR_HALF_WIDTH = 0.15

SEGMENT_LENGTHS = np.array([
    np.sqrt((s[2] - s[0])**2 + (s[3] - s[1])**2) for s in SKELETON_SEGMENTS
])
TOTAL_LENGTH = SEGMENT_LENGTHS.sum()
CUMULATIVE_LENGTHS = np.concatenate([[0], np.cumsum(SEGMENT_LENGTHS)])

D_LEFT_END = CUMULATIVE_LENGTHS[1] / TOTAL_LENGTH
D_RIGHT_START = CUMULATIVE_LENGTHS[2] / TOTAL_LENGTH

N_ZONES = 3
ZONE_NAMES = ['Gauche', 'Haut', 'Droite']
ZONE_COLORS = ['#1f77b4', '#2ca02c', '#d62728']


def compute_corridor_polygon():
    """Return (x_coords, y_coords) for the U-shaped corridor as a closed polygon.

    The U-corridor outer boundary goes:
      - Left arm outer:   x=0.00, from y=0.00 up to y=1.05
      - Top outer:        y=1.05, from x=0.00 to x=1.00
      - Right arm outer:  x=1.00, from y=1.05 down to y=0.00
      - Right arm bottom: y=0.00, from x=1.00 to x=0.70
      - Right arm inner:  x=0.70, from y=0.00 up to y=0.75
      - Top inner:        y=0.75, from x=0.70 to x=0.30
      - Left arm inner:   x=0.30, from y=0.75 down to y=0.00
      - Left arm bottom:  y=0.00, from x=0.30 to x=0.00  (close)
    """
    hw = CORRIDOR_HALF_WIDTH
    # Outer boundary (clockwise)
    x_outer_left = 0.15 - hw       # 0.00
    x_outer_right = 0.85 + hw      # 1.00
    y_top_outer = 0.90 + hw        # 1.05
    y_bottom = 0.00

    # Inner boundary
    x_inner_left = 0.15 + hw       # 0.30
    x_inner_right = 0.85 - hw      # 0.70
    y_top_inner = 0.90 - hw        # 0.75

    # Trace the U polygon (outer then inner cutout)
    x = [
        x_outer_left, x_outer_left,                    # left outer bottom -> top
        x_outer_right, x_outer_right,                  # top right -> right outer bottom
        x_inner_right, x_inner_right,                  # right inner bottom -> top inner
        x_inner_left, x_inner_left,                    # left inner top -> bottom
        x_outer_left,                                  # close
    ]
    y = [
        y_bottom, y_top_outer,                         # left outer
        y_top_outer, y_bottom,                         # right outer
        y_bottom, y_top_inner,                         # right inner
        y_top_inner, y_bottom,                         # left inner
        y_bottom,                                      # close
    ]
    return x, y
