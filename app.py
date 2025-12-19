# app.py – xG-NextGen Interactive & Batch Demo (Safe Booster-first loader)
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import brier_score_loss, roc_auc_score
import json
from pathlib import Path
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------------
# Model path and safe loader
# ---------------------------------------------------------------------
MODEL_PATH = Path("models/xgboost_model.json")

@st.cache_resource  # comment out during debugging if you want fresh loads each time
def load_model_safe():
    """
    Load the XGBoost model defensively:
      - Load as a Booster first (this avoids calling XGBClassifier.load_model which can fail).
      - Inspect scikit metadata and attach Booster to a wrapper if appropriate.
      - Return either a wrapper (XGBClassifier/XGBRegressor) or a raw Booster.
    """
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))  # your diagnostics showed this works

    meta = None
    meta_str = booster.attr("scikit_learn")
    if meta_str:
        try:
            meta = json.loads(meta_str)
        except Exception:
            meta = None

    # If metadata says classifier/regressor, attach to a wrapper to keep sklearn API
    if meta and meta.get("_estimator_type") == "classifier":
        clf = XGBClassifier()
        clf._Booster = booster
        # restore classes_ if present in metadata (optional)
        if "classes_" in meta:
            clf.classes_ = meta["classes_"]
        return clf

    if meta and meta.get("_estimator_type") == "regressor":
        reg = XGBRegressor()
        reg._Booster = booster
        return reg

    # fallback: return raw Booster
    return booster

# ---------------------------------------------------------------------
# Helpers: feature names, safe predictions, SHAP explainer creation
# ---------------------------------------------------------------------
def get_trained_feature_names(model_obj):
    """Return list of feature names stored in the Booster (works for wrapper or raw Booster)."""
    if isinstance(model_obj, xgb.Booster):
        return list(model_obj.feature_names or [])
    else:
        try:
            return list(model_obj.get_booster().feature_names or [])
        except Exception:
            return []

def predict_pos_proba(model_obj, X, feature_names):
    """
    Return probability for the positive class for each row in X.
    - If model_obj is a Booster -> use DMatrix + booster.predict()
    - If model_obj is a sklearn wrapper -> try predict_proba; if that errors, fallback to booster
    """
    if isinstance(model_obj, xgb.Booster):
        dm = xgb.DMatrix(X, feature_names=feature_names)
        preds = model_obj.predict(dm)
        return np.asarray(preds)
    else:
        try:
            return np.asarray(model_obj.predict_proba(X)[:, 1])
        except Exception:
            # fallback to underlying booster (some wrappers with only _Booster attached might need this)
            booster = model_obj.get_booster()
            dm = xgb.DMatrix(X, feature_names=feature_names)
            preds = booster.predict(dm)
            return np.asarray(preds)

def make_shap_explainer(model_obj, feature_names):
    """
    Create a SHAP explainer that works for either Booster or wrapper.
    For Booster, we wrap a predict function that accepts DataFrame/array.
    """
    sample = pd.DataFrame([{c: 0 for c in feature_names}])
    if isinstance(model_obj, xgb.Booster):
        def pred_fn(X):
            dm = xgb.DMatrix(X, feature_names=feature_names)
            return model_obj.predict(dm)
        return shap.Explainer(pred_fn, sample)
    else:
        # wrapper case: try direct explainer; if it fails, fall back to booster pred_fn
        try:
            return shap.Explainer(model_obj, sample)
        except Exception:
            booster = model_obj.get_booster()
            def pred_fn(X):
                dm = xgb.DMatrix(X, feature_names=feature_names)
                return booster.predict(dm)
            return shap.Explainer(pred_fn, sample)

# ---------------------------------------------------------------------
# Load model & explainer
# ---------------------------------------------------------------------
# Load model (safe)
model_obj = load_model_safe()

# Determine feature names used by model
trained_feats = get_trained_feature_names(model_obj)
if not trained_feats:
    st.error("Model has no recorded feature names. Check that the saved model contains feature metadata.")
    st.stop()

# Build SHAP explainer (not cached to avoid pickling issues)
explainer = make_shap_explainer(model_obj, trained_feats)

# ---------------------------------------------------------------------
# Sidebar: Single-Shot Inputs (UI)
# ---------------------------------------------------------------------
st.sidebar.title("Single-Shot Demo")

# Game context
gd     = st.sidebar.slider("Goal difference before shot", -5, 5, 0)
home   = st.sidebar.checkbox("Home team shooting?", value=True)
minute = st.sidebar.slider("Match minute", 0.0, 95.0, 45.0)

# Shot geometry
x = st.sidebar.slider("X coordinate", 0.0, 120.0, 60.0)
y = st.sidebar.slider("Y coordinate", 0.0, 80.0, 40.0)
goal_x, goal_y = 120, 40
distance = np.hypot(x - goal_x, y - goal_y)
angle    = np.degrees(np.arctan2(abs(y - goal_y), goal_x - x))

# Defensive pressure
defs             = st.sidebar.slider("Defenders within 5 m", 0, 10, 1)
gk_dist          = st.sidebar.slider("Goalkeeper distance", 0.0, 50.0, 16.0)
angular_pressure = st.sidebar.slider(
    "Angular defensive pressure", 0.0, 1.5, 0.0, step=0.01
)

# Possession build-up
n_prev = st.sidebar.slider("# Passes in last 5 events", 0, 5, 1)

# Assist type (UI only)
assist = st.sidebar.selectbox(
    "Assist type", ['None','Cross','Through Ball','Other']
)

# Assemble UI dict (13 total)
ui = {
    'goal_difference':  gd,
    'is_home':          int(home),
    'minute':           minute,
    'distance':         distance,
    'angle':            angle,
    'defenders_in_5m':  defs,
    'gk_distance':      gk_dist,
    'abs_goal_diff':    abs(gd),
    'n_prev_passes':    n_prev,
    'angular_pressure': angular_pressure,
}

# one-hot assist
for opt in ['Cross','Through Ball','Other']:
    ui[f'assist_{opt}'] = int(assist == opt)

single_df = pd.DataFrame([ui])

# ---------------------------------------------------------------------
# Main: Single-Shot Prediction & SHAP
# ---------------------------------------------------------------------
st.title("xG-NextGen Interactive & Batch Demo")

st.subheader("Single-Shot Input")
st.write(single_df)

# **SUBSET** to exactly the features the model was trained on
# If user input has all features, this will succeed; otherwise model will error earlier.
try:
    X_single = single_df[trained_feats]
except KeyError as e:
    st.error(f"Input is missing required model features: {e}")
    st.stop()

# Predict xG using the safe helper
try:
    xg_prob = float(predict_pos_proba(model_obj, X_single, trained_feats)[0])
    st.metric("Predicted xG", f"{xg_prob:.3f}")
except Exception as e:
    st.error("Error computing prediction. See logs for details.")
    st.exception(e)
    st.stop()

# SHAP explanation on the exact same subset
st.subheader("SHAP Waterfall Explanation")
try:
    shap_vals = explainer(X_single)
    # waterfall_legacy expects (expected_value, shap_values, feature_names)
    fig = shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_vals.values[0],
        feature_names=trained_feats,
        show=False
    )
    st.pyplot(fig)
except Exception as e:
    st.error("SHAP explanation failed to render.")
    st.exception(e)

st.markdown("---")

# ---------------------------------------------------------------------
# Sidebar: Batch CSV Upload
# ---------------------------------------------------------------------
st.sidebar.title("Batch Upload")
upload = st.sidebar.file_uploader("Upload shots CSV", type=["csv"])

if upload:
    st.subheader("Batch Predictions")
    batch = pd.read_csv(upload)

    # ensure required cols
    missing = [c for c in trained_feats if c not in batch.columns]
    if missing:
        st.error(f"Missing columns for model: {missing}")
    else:
        X_batch = batch[trained_feats]
        try:
            batch['xG'] = predict_pos_proba(model_obj, X_batch, trained_feats)
            st.write(batch)
        except Exception as e:
            st.error("Error computing batch predictions.")
            st.exception(e)
            st.stop()

        if 'goal' in batch.columns:
            brier = brier_score_loss(batch['goal'], batch['xG'])
            st.write(f"**Batch Brier score:** {brier:.4f}")
            try:
                auc = roc_auc_score(batch['goal'], batch['xG'])
                st.write(f"**Batch AUC-ROC:** {auc:.4f}")
            except Exception:
                pass

        st.subheader("Predicted xG Distribution")
        st.bar_chart(batch['xG'])

        st.markdown("---")
        st.caption("Batch predictions on uploaded CSV")
