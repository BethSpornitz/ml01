# 🩺 Midterm – Diabetes Classification (Pima Indians Dataset)

**Author:** Beth Spornitz  
**Date:** November 8, 2025  

---

## 🎯 Project Overview
This project uses the **Pima Indians Diabetes dataset** to build and evaluate machine learning models that predict whether a person has diabetes (1) or does not (0).

Models evaluated:
- **Decision Tree Classifier**
- **Support Vector Machine (SVM – RBF Kernel)**

Each model is tested on three cases:

| Case | Features Used     |
|------|--------------------|
| 1    | `BMI`              |
| 2    | `Glucose`          |
| 3    | `Glucose + BMI`    |

---

## 📁 Dataset
**Source:**  
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  

**File Used:**  
`data/diabetes.csv`

---

## 🔗 Project Files 
| File Type      | Path to Update |
|----------------|------------------|
| ✅ Notebook    | `[ADD_PATH_TO_NOTEBOOK_HERE.ipynb]` |
| ✅ Peer Review | `[ADD_PATH_TO_PEER_REVIEW_HERE.md]` |

---

## ⚙️ Workflow 1 – Set Up Machine
Make sure the following are installed:
- VS Code (with Python, Jupyter, Pylance, Ruff extensions)
- Git
- uv (Python environment manager)

---

## ⚙️ Workflow 2 – Set Up Project

### 2.1 Clone Repository
```bash
git clone https://github.com/BethSpornitz/ml-bethspornitz
```

### 2.2 Create & Activate Environment
```bash
uv venv
uv python pin 3.12
uv sync --extra dev --extra docs --upgrade
uv run pre-commit install
uv run python --version
```

Activate:
```bash
.\.venv\Scripts\activate   # Windows PowerShell
# or
source .venv/bin/activate  # macOS/Linux
```

---

## ⚙️ Workflow 3 – Daily Development
```bash
git pull
uv sync --extra dev --extra docs --upgrade
uvx ruff check --fix
uv run pre-commit run --all-files
git add .
uv run pytest
```

---

## ⚙️ Workflow 4 – Commit & Push
```bash
git add .
git commit -m "Midterm: diabetes classification progress"
git push -u origin main
```

---

## ⚙️ Workflow 5 – Documentation
If MkDocs is configured:
- Documentation builds automatically from `docs/` when pushing to GitHub.
- Configured using `mkdocs.yml`.

---

## 🧩 How the Project Works
| Step | Description |
|------|-------------|
| 1 | Load `diabetes.csv` dataset |
| 2 | Replace biologically impossible zero values with `NaN` (Glucose, BloodPressure, SkinThickness, Insulin, BMI) |
| 3 | Impute missing values with median |
| 4 | Create optional `BMI_Category` feature (for interpretation only) |
| 5 | Perform Stratified 80/20 train-test split |
| 6 | Train Decision Tree and SVM (RBF kernel) models on 3 feature cases |
| 7 | Evaluate accuracy, precision, recall, F1-score |
| 8 | Summarize results in table and interpret findings |

---

## 📊 Model Performance Summary (Test Set)
| Model Type        | Case | Features Used     | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score | Notes |
|-------------------|------|-------------------|----------|----------------------|------------------|----------|-------|
| Decision Tree     | 1    | BMI               | 0.62     | 0.45                 | 0.37             | 0.41     | BMI alone is weak |
| Decision Tree     | 2    | Glucose           | 0.67     | 0.55                 | 0.31             | 0.40     | Glucose improves performance |
| Decision Tree     | 3    | Glucose + BMI     | 0.71     | 0.59                 | 0.56             | 0.57     | ✅ Best Decision Tree |
| SVM (RBF Kernel)  | 1    | BMI               | 0.65     | 0.50                 | 0.15             | 0.23     | Low recall |
| SVM (RBF Kernel)  | 2    | Glucose           | 0.69     | 0.61                 | 0.37             | 0.46     | Better than BMI |
| SVM (RBF Kernel)  | 3    | Glucose + BMI     | 0.70     | 0.62                 | 0.39             | 0.48     | Best SVM case |

---

## 💡 Key Insights
- **Glucose is the strongest single predictor**.
- **BMI alone performs poorly**, but improves in combination with Glucose.
- **Decision Tree (Glucose + BMI)** gives the highest accuracy (~71%).
- **SVM models have low recall without scaling.**
- **Stratified train-test split preserves class distribution (~65% non-diabetic / 35% diabetic).**

---

## 🧾 Acknowledgements
- Instructor: **Dr. Denise Case**  
- Dataset Source: **UCI / Kaggle – Pima Indians Diabetes Database**  
- Template: `applied-ml-template`  
- Tools Used: Python, uv, Scikit-Learn, VS Code, Git, MkDocs  

---
