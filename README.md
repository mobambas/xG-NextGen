# xG-NextGen: Improving Expected Goals with AI

**Inspired by the highs and lows I experienced as a lifelong Manchester United supporter during the 2024/25 season**, this repository brings together my passion for football and my expertise in data science to tackle one of the game’s most controversial metrics: Expected Goals (xG).

## 📖 Project Introduction

Conventional xG models—built almost exclusively on basic shot location and angle—often misrepresent true scoring probabilities, leaving both fans and analysts frustrated when "expected" goals don’t materialize on the scoresheet. Having watched Manchester United generate xG north of 2.0 on numerous occasions yet leave Old Trafford empty-handed, I set out to diagnose and remedy these blind spots.

xG-NextGen is an end-to-end research pipeline that uses state-of-the-art machine learning to:

* **Integrate Rich Contextual Features:** defender proximity, goalkeeper positioning, shot velocity, assist type, pre-shot sequences, and match state.
* **Model Player and Keeper Skill:** hierarchical embeddings capture individual finishing and saving ability, rather than treating all shooters and keepers as interchangeable.
* **Apply Robust Calibration:** out-of-sample isotonic regression and Platt scaling ensure probabilities align with real-world outcomes, especially at the extremes.
* **Adapt Across Leagues and Seasons:** hierarchical Bayesian priors and team-season embeddings allow the model to evolve with tactical trends.

## 🌟 Inspiration & Motivation

1. **Fan’s Frustration, Researcher’s Curiosity:** Tweets from the “xG Philosophy” community and academic critiques challenged me to look beneath the surface of the numbers I loved.
2. **Bridging the Gap:** My goal is to transform xG from a frequently criticized statistic into a trusted, explainable metric that coaches, analysts, and fans can rely on.

## 📂 Pipeline Overview

1. **Data Collection & Cleaning**: Flatten StatsBomb JSONs and other event data into a shot-level dataset.
2. **Feature Engineering**: Over 20 features spanning shot specifics, build-up context, defensive freeze-frames, and player embeddings.
3. **Model Building**: Benchmark logistic regression, XGBoost, convolutional spatial models, and hierarchical Bayesian frameworks.
4. **Calibration & Evaluation**: Use Brier score, reliability diagrams, log loss, and out-of-sample calibration techniques.
5. **Explainability & Deployment**: Visualize with SHAP and deploy an interactive Streamlit demo for real-time analysis.

## 🎯 Project Goals

* Dramatically reduce the gap between expected and actual goals.
* Offer deep transparency into each prediction, explaining *why* a chance is rated its probability.
* Provide an open-source foundation for future xG research and operational use.

*Join me in redefining what it means to measure a “good chance” in the beautiful game.*
