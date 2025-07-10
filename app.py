# app.py – xG‑NextGen Interactive & Batch Demo (Option 2 patch)

import os
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import brier_score_loss, roc_auc_score

# —————————————————————————————————————————————
# Constants & Model Loading
# —————————————————————————————————————————————

MODEL_PATH     = 'models/xgboost_model.json'
# UI knows all 13, but model was trained on only 10:
UI_FEATURES = [
    'goal_difference','is_home','minute','distance','angle',
    'defenders_in_5m','gk_distance','abs_goal_diff',
    'n_prev_passes','angular_pressure',
    'assist_Cross','assist_Through Ball','assist_Other'
]

@st.cache_resource
def load_model():
    m = xgb.XGBClassifier()
    m.load_model(MODEL_PATH)
    return m

@st.cache_resource
def load_shap_explainer(_model):
    # build explainer on a dummy DataFrame with exactly the model's feature names
    trained_feats = _model.get_booster().feature_names
    sample = pd.DataFrame([{c:0 for c in trained_feats}])
    return shap.Explainer(_model, sample)

model       = load_model()
explainer   = load_shap_explainer(model)
trained_feats = model.get_booster().feature_names  # the 10 features your model expects

# —————————————————————————————————————————————————
# Sidebar: Single‐Shot Sliders
# —————————————————————————————————————————————————

st.sidebar.title("Single‐Shot Demo")

st.sidebar.subheader("Game Context")
gd     = st.sidebar.slider("Goal difference before shot", -5, 5, 0)
home   = st.sidebar.checkbox("Home team shooting?", value=True)
minute = st.sidebar.slider("Match minute", 0.0, 95.0, 45.0)

st.sidebar.subheader("Shot Geometry")
x = st.sidebar.slider("X coordinate", 0.0, 120.0, 60.0)
y = st.sidebar.slider("Y coordinate", 0.0, 80.0, 40.0)
goal_x, goal_y = 120, 40
distance = np.hypot(x - goal_x, y - goal_y)
angle    = np.degrees(np.arctan2(abs(y - goal_y), goal_x - x))

st.sidebar.subheader("Defensive Pressure")
defs             = st.sidebar.slider("Defenders within 5 m", 0, 10, 1)
gk_dist          = st.sidebar.slider("Goalkeeper distance", 0.0, 50.0, 16.0)
angular_pressure = st.sidebar.slider(
    "Angular defensive pressure", 0.0, 1.5, 0.0, step=0.01
)

st.sidebar.subheader("Possession Build‑Up")
n_prev = st.sidebar.slider("# passes in last 5 events", 0, 5, 1)

st.sidebar.subheader("Assist Type")
assist = st.sidebar.selectbox(
    "Assist type", ['None','Cross','Through Ball','Other']
)

# assemble the full UI DataFrame (13 cols)
ui_dict = {
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
for opt in ['Cross','Through Ball','Other']:
    ui_dict[f'assist_{opt}'] = int(assist == opt)

single_df = pd.DataFrame([ui_dict])


# —————————————————————————————————————————————————
# Main: Single‐Shot Prediction + SHAP
# —————————————————————————————————————————————————

st.title("xG‑NextGen Interactive & Batch Demo")

st.subheader("Single‐Shot Input")
st.write(single_df)

# SUBSET to exactly what the model expects:
X_single = single_df[trained_feats]

xg_prob = model.predict_proba(X_single)[0, 1]
st.metric("Predicted xG", f"{xg_prob:.3f}")

st.subheader("SHAP Waterfall Explanation")
shap_vals = explainer(X_single)
fig = shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value,
    shap_vals.values[0],
    feature_names=trained_feats,
    show=False
)
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

    # ensure we drop any extra columns before predict
    missing = [c for c in trained_feats if c not in batch.columns]
    if missing:
        st.error(f"Missing feature columns: {missing}")
    else:
        X_batch = batch[trained_feats]
        batch['xG'] = model.predict_proba(X_batch)[:, 1]
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
