# 🧠 Midterm – Classification Analysis (Diabetes & Mushroom Datasets)
**Author:** Beth Spornitz  
**Date:** November 8, 2025  

---

## 📌 Overview
This repository contains two completed machine learning classification projects that follow the midterm requirements with correct numbered sections and reflection responses after each section.

| Dataset | Type | Goal |
|---------|------|------|
| Diabetes (Pima Indians) | Binary Classification | Predict whether a patient has diabetes |
| Mushroom (UCI Dataset) | Binary Classification | Predict whether a mushroom is edible or poisonous |

---

## 📁 Dataset Information

| Dataset | Source | File |
|---------|--------|------|
| Diabetes | Kaggle | `data/diabetes.csv` |
| Mushroom | UCI ML Repository | `data/mushrooms.csv` |

---

## ⚙️ Workflow 1 – Set Up Machine

Make sure you have these installed first:
- ✅ VS Code (with Extensions: Python, Jupyter, Pylance, Ruff)
- ✅ Git
- ✅ uv (Python environment & dependency manager)

---

## ⚙️ Workflow 2 – Set Up Project

### ✅ 2.1 Clone the Repository
```bash
git clone https://github.com/BethSpornitz/ml-bethspornitz
```

### ✅ 2.2 Create and Activate Virtual Environment
```bash
uv venv
uv python pin 3.12
uv sync --extra dev --extra docs --upgrade
uv run pre-commit install
uv run python --version
```

Activate the environment:
```bash
.\.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate  # macOS/Linux
```

---

## ⚙️ Workflow 3 – Daily Workflow
```bash
git pull
uv sync --extra dev --extra docs --upgrade
uvx ruff check --fix
uv run pre-commit run --all-files
git add .
uv run pytest
```

---

## ⚙️ Workflow 4 – Save and Push Work
```bash
git add .
git commit -m "Update midterm classification"
git push -u origin main
```

---

## 🍩 Project 1 – Diabetes Classification

### 🔬 Feature Cases
| Case | Features Used     |
|------|--------------------|
| 1    | BMI                |
| 2    | Glucose            |
| 3    | Glucose + BMI      |

### 📊 Model Results — Decision Tree & SVM (Test Set)

| Model Type       | Case | Features       | Accuracy | Precision | Recall | F1-Score |
|------------------|------|----------------|----------|-----------|--------|----------|
| Decision Tree    | 1    | BMI            | 0.62     | 0.45      | 0.37   | 0.41     |
| Decision Tree    | 2    | Glucose        | 0.67     | 0.55      | 0.31   | 0.40     |
| ✅ Decision Tree | 3    | Glucose + BMI  | **0.71** | **0.59**  | **0.56** | **0.57** |
| SVM (RBF)        | 1    | BMI            | 0.65     | 0.50      | 0.15   | 0.23     |
| SVM (RBF)        | 2    | Glucose        | 0.69     | 0.61      | 0.37   | 0.46     |
| SVM (RBF)        | 3    | Glucose + BMI  | 0.70     | 0.62      | 0.39   | 0.48     |

---

## 🍄 Project 2 – Mushroom Classification

### 🔬 Feature Cases
| Case | Features Used         |
|------|------------------------|
| 1    | Odor                  |
| 2    | Gill Size             |
| 3    | Odor + Gill Size      |

### 📊 Model Results (Test Set)

| Model Type        | Case | Accuracy | Precision | Recall | F1-Score |
|-------------------|------|----------|-----------|--------|----------|
| Decision Tree     | 1    | 0.986    | 0.987     | 1.000  | 0.987    |
| Decision Tree     | 2    | 0.774    | 0.891     | 0.604  | 0.720    |
| Decision Tree     | 3    | 0.986|   |   0.987   | 1.00*  | 0.987    |
| SVM (RBF Kernel)  | 3    | 0.986    | 1.000     | 0.971  | 0.985    |
| Neural Network    | 3    | 0.986    | 1.000     | 0.971  | 0.985    |

---

## 💡 Key Insights

### Diabetes Dataset:
✔ Glucose is the strongest single predictor of diabetes  
✔ BMI alone performs poorly but helps when combined  
✔ Best performing model: **Decision Tree with Glucose + BMI (71% accuracy)**  
✔ SVM struggles with recall without scaling  

### Mushroom Dataset:
✔ Odor is almost a perfect predictor  
✔ Combining Odor + Gill Size leads to ~98.6% accuracy  
✔ Decision Tree, SVM, and Neural Network all perform extremely well  

---

## 📁 Repository Structure

| File | Purpose |
|------|---------|
| `notebooks/ml_midterm_diabetes.ipynb` | Diabetes classification notebook |
| `notebooks/ml_midterm_mushroom.ipynb` | Mushroom classification notebook |
| `data/diabetes.csv` | Diabetes dataset |
| `data/mushrooms.csv` | Mushroom dataset |
| `peer_review.md` | Peer review template |
| `README.md` | This file |

---

## 🧾 Acknowledgements
- Instructor: **Dr. Denise Case**  
- Dataset Sources: **Kaggle** and **UCI Machine Learning Repository**  
- Tools Used: Python, uv, pandas, scikit-learn, Jupyter, VS Code, Git  

---
