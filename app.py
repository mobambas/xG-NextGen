# app.py – xG‑NextGen Interactive & Batch Demo

import os
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score

# —————————————————————————————————————————————
# Constants & Model Loading
# —————————————————————————————————————————————

MODEL_PATH      = 'models/xgboost_model.json'
FEATURE_COLUMNS = [
    'goal_difference','is_home','minute','distance','angle',
    'defenders_in_5m','gk_distance','abs_goal_diff',
    'n_prev_passes','angular_pressure'
]

@st.cache_resource
def load_model():
    mdl = xgb.XGBClassifier()
    mdl.load_model(MODEL_PATH)
    return mdl

@st.cache_resource
def load_shap_explainer(_model):
    # initialize on a zero‐filled sample
    sample = pd.DataFrame([{c: 0 for c in FEATURE_COLUMNS}])
    return shap.Explainer(_model, sample)

model     = load_model()
explainer = load_shap_explainer(model)


# —————————————————————————————————————————————————
# Sidebar: Single‑Shot Sliders & PDP Toggle
# —————————————————————————————————————————————————

st.sidebar.title("Single‑Shot Demo")

# -- Game Context
gd     = st.sidebar.slider("Goal difference before shot", -5, 5, 0)
home   = st.sidebar.checkbox("Home team shooting?", value=True)
minute = st.sidebar.slider("Match minute", 0.0, 95.0, 45.0)

# -- Shot Geometry
x = st.sidebar.slider("X coordinate", 0.0, 120.0, 60.0)
y = st.sidebar.slider("Y coordinate", 0.0, 80.0, 40.0)
goal_x, goal_y = 120, 40
distance = np.hypot(x - goal_x, y - goal_y)
angle    = np.degrees(np.arctan2(abs(y - goal_y), goal_x - x))

# -- Defensive Pressure
defs             = st.sidebar.slider("Defenders within 5 m", 0, 10, 1)
gk_dist          = st.sidebar.slider("Goalkeeper distance", 0.0, 50.0, 16.0)
angular_pressure = st.sidebar.slider("Angular defensive pressure", 0.0, 1.5, 0.0, step=0.01)

# -- Possession Build‑Up
n_prev = st.sidebar.slider("# Passes in last 5 events", 0, 5, 1)

# -- PDP toggle
show_pdp = st.sidebar.checkbox("Show Goal‑Difference PDP", value=False)

# Build single‑shot feature dict
single_features = {
    'goal_difference':  gd,
    'is_home':          int(home),
    'minute':           minute,
    'distance':         distance,
    'angle':            angle,
    'defenders_in_5m':  defs,
    'gk_distance':      gk_dist,
    'abs_goal_diff':    abs(gd),
    'n_prev_passes':    n_prev,
    'angular_pressure': angular_pressure
}

single_df = pd.DataFrame([single_features])


# —————————————————————————————————————————————————
# Main: Single‑Shot Prediction + SHAP
# —————————————————————————————————————————————————

st.title("xG‑NextGen Interactive & Batch Demo")

st.subheader("Single‑Shot Prediction")
st.write(single_df)

xg_prob = model.predict_proba(single_df[FEATURE_COLUMNS])[0, 1]
st.metric("Predicted xG", f"{xg_prob:.3f}")

st.subheader("SHAP Waterfall Explanation")
shap_vals = explainer(single_df[FEATURE_COLUMNS])
fig_wf = shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value, shap_vals.values[0],
    feature_names=FEATURE_COLUMNS, show=False
)
st.pyplot(fig_wf)


# —————————————————————————————————————————————————
# Optional: Partial Dependence Plot for Goal Difference
# —————————————————————————————————————————————————

if show_pdp:
    st.markdown("---")
    st.subheader("Partial‑Dependence: Goal Difference → xG")

    # define grid of goal differences
    gd_vals = np.arange(-5, 6, 1)
    pdp_df = pd.DataFrame([
        {
            **{k: single_features[k] for k in single_features if k != 'goal_difference'},
            'goal_difference': gd,
            'abs_goal_diff': abs(gd)
        }
        for gd in gd_vals
    ])
    # predict
    pdp_df['xG'] = model.predict_proba(pdp_df[FEATURE_COLUMNS])[:, 1]

    # plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(gd_vals, pdp_df['xG'], marker='o')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_xlabel("Goal Difference")
    ax.set_ylabel("Predicted xG")
    ax.set_title("Partial Dependence on Goal Difference")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)


st.markdown("---")


# —————————————————————————————————————————————————
# Sidebar: Batch CSV Upload
# —————————————————————————————————————————————————

st.sidebar.title("Batch Upload")
upload = st.sidebar.file_uploader("Upload shots CSV", type=["csv"])

if upload:
    st.subheader("Batch Predictions")
    batch = pd.read_csv(upload)

    # check required columns
    missing = [c for c in FEATURE_COLUMNS if c not in batch.columns]
    if missing:
        st.error(f"Missing feature columns: {missing}")
    else:
        batch['xG'] = model.predict_proba(batch[FEATURE_COLUMNS])[:, 1]
        st.write(batch)

        if 'goal' in batch.columns:
            brier = brier_score_loss(batch['goal'], batch['xG'])
            st.write(f"**Batch Brier score:** {brier:.4f}")
            try:
                auc = roc_auc_score(batch['goal'], batch['xG'])
                st.write(f"**Batch AUC‑ROC:** {auc:.4f}")
            except:
                pass

        st.subheader("Predicted xG Distribution")
        st.bar_chart(batch['xG'])

    st.markdown("---")
    st.caption("Batch predictions on uploaded CSV")
