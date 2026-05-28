<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&size=13&duration=2500&pause=800&color=4A90D9&center=true&vCenter=true&width=600&lines=Machine+Learning+Engineer+%7C+East+Africa;825-feature+ensembles+%E2%86%92+production+APIs;SHAP+%7C+Optuna+%7C+FastAPI+%7C+Docker+%7C+Railway;OOF+score+0.19144+%E2%80%94+top+5%25+model+composite" alt="Typing SVG" />

</div>

---

# Henry Otsyula

**Machine Learning Engineer** — I build production-grade ML systems that go from raw data to deployed API, with interpretability baked in at every stage. My work sits at the intersection of rigorous statistical modelling, scalable engineering, and domain-driven feature design.

Currently focused on financial risk intelligence for emerging markets.

---

## Featured Work

### 💧 Liquidity Stress Early Warning System
**Zindi AI4EAC Finance Challenge — Production Deployment**

> Predicts 30-day mobile money liquidity stress from 6 months of transaction history. 5-model heterogeneous ensemble with SHAP interpretability and a live FastAPI service on Railway.

| | |
|---|---|
| **OOF Composite Score** | 0.19144 &nbsp;(`0.6×LogLoss + 0.4×(1−AUC)`, lower is better) |
| **Ensemble AUC** | 0.9052 |
| **Engineered Features** | 825 from 183 raw columns — 23 feature blocks |
| **Models** | LightGBM · XGBoost · CatBoost · TabNet · Logistic Regression |
| **Deployment** | FastAPI · Docker · Railway (live) |

**Key technical achievements:**
- Designed a three-cache feature engineering architecture (value / volume / highest\_amount) where adding a new cache type auto-propagates through all 9 temporal blocks with zero code changes
- Identified and quantified the **balance-lag false-negative pattern** via SHAP analysis: customers whose `inflow_slope < −5,000` but `balance_slope > 0` are systematically misclassified — the production API flags this with a rule override
- `balance_slope` (mean |SHAP| = 0.793) shows a near-vertical bifurcation at zero — the single most dominant feature by a factor of 2.76× over rank 2
- Proved stacking loses to weighted averaging in this configuration: LogReg/TabNet probability compression (max OOF 0.740/0.700 vs GBM 0.87) produces Platt slope = 5.99 on the meta-model output — a structural limitation, not a hyperparameter problem
- Resolved a production deployment crash (`KeyError: 118`) traced to Git LFS pointer stubs being served as real model files; fixed by deleting `.gitattributes` and committing all artefacts as real binary objects

[![Live API](https://img.shields.io/badge/Live%20API-Railway-0B0D0E?style=flat-square&logo=railway)](https://liquidity-stress-early-warning-prediction-system-production.up.railway.app)
[![Swagger Docs](https://img.shields.io/badge/Swagger-Docs-85EA2D?style=flat-square&logo=swagger&logoColor=black)](https://liquidity-stress-early-warning-prediction-system-production.up.railway.app/docs)
[![Repo](https://img.shields.io/badge/GitHub-Repo-181717?style=flat-square&logo=github)](https://github.com/theerealhenry/liquidity-stress-early-warning)

<details>
<summary><strong>Screenshots — click to expand</strong></summary>

**SHAP Global Feature Impact (beeswarm)**
![SHAP beeswarm](https://raw.githubusercontent.com/theerealhenry/liquidity-stress-early-warning/main/docs/images/SHAP_global_summary_plot.png)

**balance_slope Dependence Plot — the dominant signal**
![SHAP dependence](https://github.com/theerealhenry/theerealhenry/blob/main/assets/SHAP_dependence_plot_for_balance_slope.png)

**SHAP Theme Importance**
![SHAP themes](https://github.com/theerealhenry/theerealhenry/blob/main/assets/shap_theme_importance.png)

**Live Swagger UI**
![Swagger](https://github.com/theerealhenry/theerealhenry/blob/main/assets/swagger_ui.png)

**MLflow — 8 experiment groups, 50 runs**
![MLflow](https://github.com/theerealhenry/theerealhenry/blob/main/assets/mlflow_ui.png)

**Railway Production Deployment**
![Railway](https://github.com/theerealhenry/theerealhenry/blob/main/assets/railway_deploy_1.png)

</details>

---

### 💳 Credit Risk Intelligence System

> End-to-end loan default prediction platform with real-time SHAP explainability, threshold optimisation, and an interactive Streamlit dashboard.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://loan-default-risk-app.streamlit.app/)
[![Repo](https://img.shields.io/badge/GitHub-Repo-181717?style=flat-square&logo=github)](https://github.com/theerealhenry/Loan-Default-Prediction-Project)

---

### 📊 SME Financial Health Prediction

> Production-ready pipeline for predicting SME financial health categories using ensemble modelling, probability blending, and threshold optimisation.

[![Repo](https://img.shields.io/badge/GitHub-Repo-181717?style=flat-square&logo=github)](https://github.com/theerealhenry)

---

## Tech Stack

**Core ML**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-02569B?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-FF6600?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-1.x-FFD700?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TabNet](https://img.shields.io/badge/pytorch--tabnet-4.1-blueviolet?style=flat-square)

**Optimisation & Explainability**

![Optuna](https://img.shields.io/badge/Optuna-TPE%20%2B%20Pruner-6DB3F2?style=flat-square)
![SciPy](https://img.shields.io/badge/SciPy-Nelder--Mead-8CAAE6?style=flat-square&logo=scipy)
![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-FF6B6B?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?style=flat-square&logo=mlflow)

**Data**

![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-PostgreSQL-336791?style=flat-square&logo=postgresql&logoColor=white)

**APIs & Deployment**

![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Containerisation-2496ED?style=flat-square&logo=docker&logoColor=white)
![Railway](https://img.shields.io/badge/Railway-Cloud-0B0D0E?style=flat-square&logo=railway&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

**Tooling**

![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![YAML](https://img.shields.io/badge/YAML-Config-CB171E?style=flat-square)
![joblib](https://img.shields.io/badge/joblib-Serialisation-lightgrey?style=flat-square)
![SQLite](https://img.shields.io/badge/SQLite-Optuna%20Storage-003B57?style=flat-square&logo=sqlite&logoColor=white)

---

## How I Work

**Feature engineering is a first-class system.** Features are designed for extensibility, tested for single-row correctness at inference time, and version-controlled with a hard contract enforced at `transform()`. The engineering pipeline is stateless and deterministic — same input always produces identical output.

**Calibration before ensembling.** Every base model gets Platt calibration evaluated against OOF predictions before being handed to the ensemble layer. The competition weighting (60% LogLoss) means miscalibrated probabilities kill your score even with strong AUC.

**OOF-only meta-models.** Stacking meta-models are trained exclusively on out-of-fold predictions — never on in-sample data. Nested cross-validation for any stacking layer.

**Deployment is part of the model spec.** Every project includes containerisation, health checks, startup verification, and structured logging. The API is not an afterthought — it is designed alongside the inference pipeline.

**Interpretability informs the model, not just explains it.** SHAP analysis on the AI4EAC project identified a systematic false-negative failure mode that wasn't visible in aggregate metrics. The fix is encoded directly into the production API as a rule override.

---

## GitHub Stats

<div align="center">

![Stats](https://github-readme-stats.vercel.app/api?username=theerealhenry&show_icons=true&theme=default&hide_border=true&count_private=true)
&nbsp;
![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=theerealhenry&layout=compact&theme=default&hide_border=true)

</div>

<div align="center">

![Streak](https://streak-stats.demolab.com/?user=theerealhenry&theme=default&hide_border=true)

</div>

---

## Contact

**Email:** henryotsyula01@gmail.com  
**LinkedIn:** [henry-otsyula-datascientist](https://linkedin.com/in/henry-otsyula-datascientist)  
**Portfolio:** [datascienceportfol.io/otsyulahenry](https://www.datascienceportfol.io/otsyulahenry)

Open to roles in machine learning engineering, applied AI, and data science — particularly where interpretability and production deployment matter.

![Visitors](https://komarev.com/ghpvc/?username=theerealhenry&color=4A90D9&style=flat-square)
