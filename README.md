# xG‑NextGen: Improving Expected Goals with AI

**Inspired by the highs and lows I experienced as a lifelong Manchester United supporter during the 2024/25 season**, this repository brings together my passion for football and my expertise in data science to tackle one of the game’s most controversial metrics: Expected Goals (xG).

---

## 📖 Project Introduction

Conventional xG models—built almost exclusively on basic shot location and angle—often misrepresent true scoring probabilities, leaving both fans and analysts frustrated when “expected” goals don’t materialize on the scoresheet. Having watched Manchester United generate xG north of 2.0 on numerous occasions yet leave Old Trafford empty‑handed, I set out to diagnose and remedy these blind spots.

**xG‑NextGen** is an end‑to‑end research pipeline that uses state‑of‑the‑art machine learning to:

- **Integrate Rich Contextual Features:** defender proximity, goalkeeper positioning, shot velocity, assist type, pre‑shot sequences, and match state.  
- **Model Player and Keeper Skill:** hierarchical embeddings capture individual finishing and saving ability, rather than treating all shooters and keepers as interchangeable.  
- **Apply Robust Calibration:** out‑of‑sample isotonic regression and Platt scaling ensure probabilities align with real‑world outcomes, especially at the extremes.  
- **Adapt Across Leagues and Seasons:** hierarchical Bayesian priors and team‑season embeddings allow the model to evolve with tactical trends.  

---

## 📂 Repository Structure

xG‑NextGen/
├── app.py # Streamlit demo: interactive & batch prediction
├── requirements.txt # Python dependencies
├── scripts/ # Utility scripts (JSON loading, freeze‑frame parsing)
│ └── utils.py
├── notebooks/ # Jupyter notebooks for each pipeline stage
│ ├── 01_data_cleaning.ipynb
│ ├── 02_feature_eng.ipynb
│ └── 03_modeling.ipynb
├── data/ # Not tracked (see .gitignore)
│ ├── raw/ # StatsBomb JSON downloads
│ └── processed/ # cleaned CSVs: shots.csv, freeze_frames.csv, features.csv
├── models/ # trained model artifacts (e.g. xgboost_model.json)
└── outputs/ # figures, SHAP values, logs, calibration plots

yaml
Copy
Edit

> _Note: `data/raw/` and `data/processed/` are excluded from version control (see `.gitignore`)._

---

## 🔧 Setup & Installation

1. **Clone this repo**  
   ```bash
   git clone https://github.com/mobambas/xG-NextGen.git
   cd xG-NextGen
Create & activate a Python environment

bash
Copy
Edit
python3.9 -m venv venv
source venv/bin/activate
Install dependencies

bash
Copy
Edit
pip install --upgrade pip
pip install -r requirements.txt
Download StatsBomb Open Data
We leverage the free StatsBomb open‑data repository for training and testing:
🔗 https://github.com/statsbomb/open-data

Download the events/*.json and lineups/*.json files for your chosen competitions.

Place them under:

bash
Copy
Edit
data/raw/events/
data/raw/lineups/
Run Data Cleaning
Open and execute all cells in notebooks/01_data_cleaning.ipynb. This will:

Unzip & flatten raw JSON

Compute goal_difference, is_home, player positions

Export shots.csv & freeze_frames.csv to data/processed/

Engineer Features
Open and execute notebooks/02_feature_eng.ipynb. This will:

Load cleaned CSVs

Compute time, distance, angle, defensive pressure, assist type, pre‑shot sequence features

One‑hot encode categoricals

Save features.csv to data/processed/

Train & Evaluate Models
Open and run notebooks/03_modeling.ipynb. This will:

Split features.csv into train/test

Train XGBoost (and benchmarks)

Evaluate AUC‑ROC (~0.876), Brier score, log loss, calibration

Save best model to models/xgboost_model.json and SHAP outputs to outputs/

Launch the Streamlit Demo

bash
Copy
Edit
streamlit run app.py
Interactive mode: adjust the sidebar sliders for single‑shot predictions

Batch mode: upload a CSV of shot‑level features (same columns as trained features)

🌟 Key Results & Metrics
AUC‑ROC: 0.876

Brier Score: 0.0686

Log Loss: 0.2361

Top SHAP Features:

goal_difference

angle

gk_distance

distance

minute

🎯 Project Goals
Reduce the gap between expected and actual goals.

Explain every prediction, showing why a chance is rated at a given probability.

Provide an open‑source foundation for future xG research and operational use.

Join me in redefining what it means to measure a “good chance” in the beautiful game.
Pull requests, issues, or feature ideas are warmly welcome!

makefile
Copy
Edit
::contentReference[oaicite:0]{index=0}







Sources
