"""Streamlit application: U-Maze Neural Decoder Simulation.

The maze animation runs entirely in the browser via Plotly.js + Plotly.react()
to avoid the flicker caused by Streamlit recreating DOM elements each frame.
All data is sent once, and JavaScript handles frame-by-frame updates.
"""

import json
import os

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from data_loader import load_data
from maze_geometry import ZONE_NAMES, SKELETON_SEGMENTS, compute_corridor_polygon
from figures import (
    create_mse_components,
    create_mean_eucl_error,
    create_zone_accuracy,
    create_trajectory_aggregate,
)

# --- Page config ---
st.set_page_config(
    page_title="Hacktion Potential - U-Maze",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Hacktion Potential theme ---
st.markdown("""
<style>
    .stApp { background-color: #fff6e1; }
    section[data-testid="stSidebar"] { background-color: #ffebbf; }
    .stApp h1 { color: #af4635; }
    .stApp h2, .stApp h3 { color: #733a30; }
    [data-testid="stMetricLabel"] { color: #733a30 !important; }
    [data-testid="stMetricValue"] { color: #af4635 !important; }
    .stButton > button {
        background-color: #af4635; color: #fff6e1;
        border: none; border-radius: 6px;
    }
    .stButton > button:hover {
        background-color: #733a30; color: #fff6e1;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #af4635;
    }
    .stProgress > div > div > div { background-color: #af4635; }
    hr { border-color: #d67869 !important; }
    .stCaption, .stMarkdown p { color: #733a30; }
    .stCheckbox label span { color: #733a30 !important; }
    .stApp, .stApp label, .stApp span, .stApp div { color: #733a30; }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div { color: #733a30; }
    .stSlider label, .stSlider div[data-testid="stTickBarMin"],
    .stSlider div[data-testid="stTickBarMax"] { color: #733a30 !important; }
</style>
""", unsafe_allow_html=True)

# --- Data loading ---
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
data = load_data(OUTPUT_DIR)
N = data['n_points']

# --- Sidebar ---
st.sidebar.title("Controles")
speed = st.sidebar.slider("Pas par frame", 1, 100, 1)
frame_delay = st.sidebar.slider("Delai (ms)", 20, 500, 80, step=10)
trail_length = st.sidebar.slider("Trail", 10, 300, 80, step=10)
show_1sigma = st.sidebar.checkbox("1-sigma", value=True)
show_2sigma = st.sidebar.checkbox("2-sigma", value=True)

# --- Title ---
st.title("Hacktion Potential - U-Maze Decoder")

# --- Prepare data for JS ---
corridor_x, corridor_y = compute_corridor_polygon()
skel_x, skel_y = [], []
for x1, y1, x2, y2 in SKELETON_SEGMENTS:
    skel_x.extend([x1, x2, None])
    skel_y.extend([y1, y2, None])

theta = np.linspace(0, 2 * np.pi, 60).tolist()

js_data = json.dumps({
    'y_test': data['y_test'].tolist(),
    'y_pred': data['y_pred'].tolist(),
    'y_sigma': data['y_sigma'].tolist(),
    'eucl_errors': data['eucl_errors'].tolist(),
    'cum_mse_pos': data['cum_mse_pos'].tolist(),
    'cum_mse_x': data['cum_mse_x'].tolist(),
    'cum_mse_y': data['cum_mse_y'].tolist(),
    'cum_mean_eucl': data['cum_mean_eucl'].tolist(),
    'cum_zone_acc': data['cum_zone_acc'].tolist(),
    'zone_pred': data['zone_pred'].tolist(),
    'zone_test': data['zone_test'].tolist(),
    'corridor_x': list(corridor_x),
    'corridor_y': list(corridor_y),
    'skel_x': skel_x,
    'skel_y': skel_y,
    'theta': theta,
    'n': N,
    'zone_names': ZONE_NAMES,
})

html_code = f"""
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<div id="maze" style="width:100%;"></div>
<div id="metrics" style="display:flex;gap:14px;flex-wrap:wrap;padding:12px 0;font-family:sans-serif;justify-content:center;"></div>
<div id="controls" style="padding:8px 0;font-family:sans-serif;">
  <button id="playBtn" onclick="togglePlay()"
    style="background:#af4635;color:#fff6e1;border:none;border-radius:6px;padding:6px 20px;cursor:pointer;font-size:14px;">
    Play
  </button>
  <button onclick="resetAnim()"
    style="background:#af4635;color:#fff6e1;border:none;border-radius:6px;padding:6px 20px;cursor:pointer;font-size:14px;margin-left:6px;">
    Reset
  </button>
  <span id="stepLabel" style="margin-left:12px;color:#733a30;font-size:13px;">t = 0 / 0</span>
</div>
<script>
const D = {js_data};
const speed = {speed};
const delay = {frame_delay};
const trailLen = {trail_length};
const show1s = {'true' if show_1sigma else 'false'};
const show2s = {'true' if show_2sigma else 'false'};

let idx = 0;
let playing = false;
let timer = null;

// -- Build initial traces (fixed structure, data updated via react) --
function buildTraces(i) {{
  const ts = Math.max(0, i - trailLen);
  const nTrail = i - ts;
  const traces = [];

  // 0: corridor
  traces.push({{
    x: D.corridor_x, y: D.corridor_y,
    fill: 'toself', fillcolor: 'rgba(175,70,53,0.12)',
    line: {{color:'#733a30', width:2.5}}, mode:'lines',
    showlegend:false, hoverinfo:'skip'
  }});
  // 1: skeleton
  traces.push({{
    x: D.skel_x, y: D.skel_y, mode:'lines',
    line: {{color:'rgba(115,58,48,0.2)', width:1, dash:'dot'}},
    showlegend:false, hoverinfo:'skip'
  }});

  // 2: true trail
  const trueX = [], trueY = [], trueCol = [];
  for (let j = ts; j < i; j++) {{
    const a = 0.05 + 0.45 * (j - ts) / Math.max(nTrail - 1, 1);
    trueX.push(D.y_test[j][0]); trueY.push(D.y_test[j][1]);
    trueCol.push('rgba(42,110,158,' + a.toFixed(2) + ')');
  }}
  traces.push({{
    x: trueX, y: trueY, mode:'markers',
    marker: {{color: trueCol, size:3}},
    showlegend:false, hoverinfo:'skip'
  }});

  // 3: pred trail
  const predX = [], predY = [], predCol = [];
  for (let j = ts; j < i; j++) {{
    const a = 0.05 + 0.45 * (j - ts) / Math.max(nTrail - 1, 1);
    predX.push(D.y_pred[j][0]); predY.push(D.y_pred[j][1]);
    predCol.push('rgba(175,70,53,' + a.toFixed(2) + ')');
  }}
  traces.push({{
    x: predX, y: predY, mode:'markers',
    marker: {{color: predCol, size:3, symbol:'x'}},
    showlegend:false, hoverinfo:'skip'
  }});

  // 4: error vectors
  const eX = [], eY = [];
  for (let j = ts; j < i; j++) {{
    eX.push(D.y_test[j][0], D.y_pred[j][0], null);
    eY.push(D.y_test[j][1], D.y_pred[j][1], null);
  }}
  traces.push({{
    x: eX, y: eY, mode:'lines',
    line: {{color:'rgba(115,58,48,0.12)', width:0.7}},
    showlegend:false, hoverinfo:'skip'
  }});

  // 5: 2-sigma ellipse
  const cx = D.y_pred[i][0], cy = D.y_pred[i][1];
  const sx = D.y_sigma[i][0], sy = D.y_sigma[i][1];
  if (show2s) {{
    const ex2 = [], ey2 = [];
    for (const t of D.theta) {{ ex2.push(cx + 2*sx*Math.cos(t)); ey2.push(cy + 2*sy*Math.sin(t)); }}
    ex2.push(ex2[0]); ey2.push(ey2[0]);
    traces.push({{
      x: ex2, y: ey2, mode:'lines', fill:'toself',
      fillcolor:'rgba(175,70,53,0.06)',
      line: {{color:'rgba(175,70,53,0.25)', width:1, dash:'dot'}},
      name:'2-sigma'
    }});
  }}

  // 6: 1-sigma ellipse
  if (show1s) {{
    const ex1 = [], ey1 = [];
    for (const t of D.theta) {{ ex1.push(cx + sx*Math.cos(t)); ey1.push(cy + sy*Math.sin(t)); }}
    ex1.push(ex1[0]); ey1.push(ey1[0]);
    traces.push({{
      x: ex1, y: ey1, mode:'lines', fill:'toself',
      fillcolor:'rgba(175,70,53,0.12)',
      line: {{color:'rgba(175,70,53,0.5)', width:1.5, dash:'dash'}},
      name:'1-sigma'
    }});
  }}

  // 7: error line
  traces.push({{
    x: [D.y_test[i][0], D.y_pred[i][0]],
    y: [D.y_test[i][1], D.y_pred[i][1]],
    mode:'lines', line: {{color:'#733a30', width:1.5}},
    showlegend:false, hoverinfo:'skip'
  }});

  // 8: true point
  traces.push({{
    x: [D.y_test[i][0]], y: [D.y_test[i][1]],
    mode:'markers',
    marker: {{color:'#2a6e9e', size:12, line:{{color:'#ffebbf', width:1.5}}}},
    name:'Vrai'
  }});

  // 9: pred point
  traces.push({{
    x: [D.y_pred[i][0]], y: [D.y_pred[i][1]],
    mode:'markers',
    marker: {{color:'#af4635', size:12, symbol:'x', line:{{color:'#ffebbf', width:1}}}},
    name:'Prediction'
  }});

  return traces;
}}

const layout = {{
  xaxis: {{range:[-0.08,1.08], scaleanchor:'y', constrain:'domain',
           showgrid:false, zeroline:false, fixedrange:true}},
  yaxis: {{range:[-0.08,1.15], showgrid:false, zeroline:false, fixedrange:true}},
  height: 550,
  margin: {{l:30, r:10, t:35, b:50}},
  title: {{text:'t = 0', font:{{size:13, color:'#733a30'}}}},
  legend: {{orientation:'h', x:0, y:-0.06, font:{{size:10, color:'#733a30'}},
            bgcolor:'rgba(0,0,0,0)'}},
  plot_bgcolor: '#ffebbf',
  paper_bgcolor: '#fff6e1',
  font: {{color: '#733a30'}},
}};

const config = {{displayModeBar: false, responsive: true}};

Plotly.newPlot('maze', buildTraces(0), layout, config);

function updateMetrics(i) {{
  const err = D.eucl_errors[i].toFixed(4);
  const mse = D.cum_mse_pos[i].toFixed(5);
  const mseX = D.cum_mse_x[i].toFixed(5);
  const mseY = D.cum_mse_y[i].toFixed(5);
  const eucl = D.cum_mean_eucl[i].toFixed(4);
  const acc = (D.cum_zone_acc[i] * 100).toFixed(1) + '%';
  const sx = D.y_sigma[i][0].toFixed(4);
  const sy = D.y_sigma[i][1].toFixed(4);
  const zp = D.zone_names[D.zone_pred[i]];
  const zt = D.zone_names[D.zone_test[i]];
  const zOk = D.zone_pred[i] === D.zone_test[i];

  const m = (label, val) =>
    '<div style="text-align:center;min-width:120px;padding:8px 12px;background:#fff6e1;border-radius:8px;border:1px solid #d6786940;">' +
    '<div style="font-size:13px;color:#733a30;margin-bottom:4px;">' + label + '</div>' +
    '<div style="font-size:22px;font-weight:bold;color:#af4635;">' + val + '</div></div>';

  document.getElementById('metrics').innerHTML =
    m('Erreur', err) + m('MSE pos', mse) + m('MSE x', mseX) + m('MSE y', mseY) +
    m('Err. moy.', eucl) + m('Acc. zone', acc) +
    m('Sigma x', sx) + m('Sigma y', sy) +
    m('Zone', zp + ' (' + (zOk ? 'OK' : 'FAUX') + ')');

  document.getElementById('stepLabel').textContent = 't = ' + i + ' / ' + (D.n - 1);
}}

function stepTo(i) {{
  idx = i;
  const traces = buildTraces(i);
  const err = D.eucl_errors[i].toFixed(4);
  layout.title.text = 't = ' + i + '  |  err = ' + err;
  Plotly.react('maze', traces, layout, config);
  updateMetrics(i);
}}

function togglePlay() {{
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
  if (playing) runAnim();
  else if (timer) {{ clearTimeout(timer); timer = null; }}
}}

function resetAnim() {{
  playing = false;
  document.getElementById('playBtn').textContent = 'Play';
  if (timer) {{ clearTimeout(timer); timer = null; }}
  stepTo(0);
}}

function runAnim() {{
  if (!playing || idx >= D.n - 1) {{
    playing = false;
    document.getElementById('playBtn').textContent = 'Play';
    return;
  }}
  stepTo(idx);
  idx = Math.min(D.n - 1, idx + speed);
  timer = setTimeout(runAnim, delay);
}}

updateMetrics(0);
</script>
"""

components.html(html_code, height=720, scrolling=False)

# --- Final charts (static, shown below) ---
# Only render when user clicks "Show results"
if st.button("Afficher les resultats finaux"):
    st.markdown("---")
    st.subheader("Resultats finaux")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            create_trajectory_aggregate(data['y_test'], data['y_pred'], N - 1),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            create_mse_components(
                data['se_x'], data['se_y'], data['se_pos'],
                data['cum_mse_x'], data['cum_mse_y'], data['cum_mse_pos'],
                N - 1,
            ),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            create_mean_eucl_error(data['cum_mean_eucl'], N, N - 1),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            create_zone_accuracy(data['cum_zone_acc'], N, N - 1),
            use_container_width=True,
        )
