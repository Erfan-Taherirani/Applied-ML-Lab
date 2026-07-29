# Modeling the Relationship Between Generative AI Usage and Student Academic Performance

An end-to-end, reproducible supervised machine learning project that investigates how Generative AI (GenAI) usage patterns relate to university students' academic performance, well-being, and study behavior — and predicts students' post-semester GPA from academic background, study habits, and AI usage characteristics.

> **Note:** This project models statistical associations between AI usage and GPA. It does **not** establish a causal relationship between AI usage and academic performance.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Workflow](#workflow)
- [Key Insights](#key-insights)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Installation & Usage](#installation--usage)
- [Project Status & Limitations](#project-status--limitations)
- [Roadmap](#roadmap)
- [Contact](#contact)

---

## Project Overview

This project demonstrates a complete, portfolio-quality Data Science workflow, including:

- Exploratory Data Analysis (EDA)
- Statistical reasoning
- Reproducible preprocessing
- Feature engineering
- Modular, reusable code organization (`src/` and `utils/`)
- Interpretable regression modeling
- Professional documentation

The primary objective is **not** to chase the absolute best predictive score, but to demonstrate sound Machine Learning practice and clear technical communication — from raw data to a validated, interpretable model.

## Problem Statement

The goal is to **predict a student's post-semester GPA** using their academic background, study behavior, and Generative AI usage characteristics, and to extract meaningful insights about how these factors relate to academic and well-being outcomes.

- **Target variable:** `post_semester_gpa`
- **Task type:** Supervised regression

## Dataset

- **Source:** [AI Impact on Students — Kaggle](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students)
- **Nature:** Real-world data (not synthetic)
- **License:** No license specified by the original source
- **Size:** 50,000 observations, 15 predictors before encoding (21 after encoding), no missing values
- **Engineered target-adjacent feature:** `gpa_difference` (`post_semester_gpa - pre_semester_gpa`)

**Feature groups:**

| Group | Features |
|---|---|
| 🪪 Identifier | `Student_ID` |
| 🎓 Academic Profile | `Major_Category`, `Year_of_Study`, `Pre_Semester_GPA`, `Post_Semester_GPA` |
| 🤖 AI Behavior | `Weekly_GenAI_Hours`, `Primary_Use_Case`, `Prompt_Engineering_Skill`, `Tool_Diversity`, `Paid_Subscription` |
| 📚 Study Habits | `Traditional_Study_Hours`, `Perceived_AI_Dependency` |
| 🏛️ Institutional | `Institutional_Policy` |
| 🧠 Well-being | `Anxiety_Level_During_Exams`, `Skill_Retention_Score`, `Burnout_Risk_Level` |

## Repository Structure

```
├── data/
│   ├── raw/                        # Original and lightly-processed raw data
│   │   ├── ai_student_impact_dataset (1).csv
│   │   └── data.npz
│   └── processed/                  # Encoded / transformed data ready for modeling
│       ├── processed_data.npz
│       └── polynomial_features.npz
│
├── notebooks/
│   ├── eda.ipynb                   # Exploratory Data Analysis
│   ├── statistics.ipynb            # Statistical testing (in progress)
│   └── modeling.ipynb              # Model development & evaluation
│
├── src/                            # Project-specific, reusable pipeline code
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── visualization.py
│
├── utils/                          # General-purpose helper functions
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── visualization.py
│
├── requirements.txt
└── README.md
```

> Note: In this repository's flat file listing, `src` and `utils` modules are prefixed (`src_*.py`, `utils_*.py`) to avoid filename collisions; the tree above reflects the intended folder layout.

**Design rationale:**
- `data/` is kept separate for raw vs. processed artifacts, allowing multiple versions of the data to be tested across models without overwriting source files.
- `notebooks/` hosts the analytical narrative (EDA, statistics, modeling) — each notebook has a single, focused responsibility to keep technical review straightforward.
- `src/` contains project-specific pipeline logic, keeping notebooks clean and the workflow reproducible.
- `utils/` contains general-purpose functions reused across different Data Science projects.

## Workflow

**EDA → Statistical Analysis → Feature Engineering → Model Development → Evaluation**

1. **Preprocessing:** Type correction, feature name normalization, and derivation of `gpa_difference`.
2. **EDA:** Univariate distributions, categorical breakdowns, and bivariate/multivariate relationships between AI usage, study habits, well-being, and GPA outcomes.
3. **Feature Engineering:** One-hot encoding for nominal categories, ordinal encoding for ordered categories, and standard scaling for continuous features.
4. **Modeling:** Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor, compared on held-out test data.
5. **Validation:** 5-fold cross-validation on the training set, followed by a single held-out test set evaluation (80/20 split) to confirm generalization.

## Key Insights

- Students whose primary AI use case is **debugging/troubleshooting** tend to show larger GPA gains and better skill retention than students who mainly use AI for **direct answer generation**.
- Higher **prompt engineering skill** is associated with larger GPA improvements.
- Students under a **strict institutional ban** on GenAI tend to show smaller GPA gains than students under more permissive institutional policies.
- **Perceived AI dependency** is associated with higher exam anxiety and higher burnout risk.
- Students combining **lower traditional study hours** with **higher weekly GenAI usage** show the highest burnout risk levels.
- `Pre_Semester_GPA`, `Traditional_Study_Hours`, and `Year_of_Study` consistently emerge as the strongest predictors of `post_semester_gpa` across all three models.

## Model Performance

Best-performing model: **Gradient Boosting Regressor**

| Metric | Value |
|---|---|
| MSE | 0.021 |
| RMSE | 0.148 |
| MAE | 0.116 |
| R² | 0.908 |

Cross-validated training performance and held-out test performance were closely aligned, indicating the model generalizes well without overfitting. Results were achieved without extensive hyperparameter tuning, in line with the project's emphasis on a clean and interpretable workflow over marginal performance gains.

## Tech Stack

- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Category Encoders

See [`requirements.txt`](requirements.txt) for exact versions.

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Erfan-Taherirani/Applied-ML-Lab.git
cd Applied-ML-Lab

# Install dependencies
pip install -r requirements.txt

# Launch the notebooks
jupyter notebook notebooks/
```

## Project Status & Limitations

This project is **feature-complete** and currently undergoing final technical review before publication.

- Statistical hypothesis testing is still being finalized and is not yet comprehensive.
- Some methodological choices (e.g., polynomial feature construction, multicollinearity handling) may benefit from additional validation.
- Reported relationships between AI usage and GPA are **associational**, not causal — the dataset does not support causal inference.

## Roadmap

- [ ] Complete statistical hypothesis testing
- [ ] SHAP-based model interpretability analysis
- [ ] Residual diagnostics
- [ ] Robustness / ablation study
- [ ] Preprocessing pipeline refinements

## Contact

**Author:** Erfan Taherirani

- 📧 Email: e.taherirani81@gmail.com
- 💼 LinkedIn: [www.linkedin.com/in/erfan-taherirani](www.linkedin.com/in/erfan-taherirani)
- 🐙 GitHub: https://github.com/Erfan-Taherirani
- 📱 Telegram ID: @www_ErfanT_ir

---

*Feedback and suggestions from you are welcome.*
