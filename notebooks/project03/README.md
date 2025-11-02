# Ml03 – Titanic Classification Models

**Author:** Beth Spornitz  
**Date:** November 1,  2025  

---

## 🎯 Project Overview

This project uses the Titanic dataset from Seaborn to build and evaluate **three machine learning classification models**:

- **Decision Tree**
- **Support Vector Machine (SVM)**
- **Neural Network (MLPClassifier)**

Each model predicts whether a passenger **survived (1)** or **did not survive (0)** using three input cases:

| Case | Features Used |
|------|----------------|
| 1    | `alone` |
| 2    | `age` |
| 3    | `age + family_size` |

The notebook includes:
✔ Feature engineering  
✔ Stratified train/test split  
✔ Decision Trees, SVMs, Neural Networks  
✔ Confusion matrices and decision boundaries  
✔ Performance summary table  
✔ Reflections after each section  

---

## ⚙️ Workflow 1. Set Up Machine

Make sure you have:

- VS Code (with Python, Jupyter, Pylance, Ruff extensions)
- Git  
- uv (Python environment and dependency manager)

---

## ⚙️ Workflow 2. Set Up Project

1. **Clone repository**
```bash
git clone https://github.com/BethSpornitz/ml-bethspornitz
```

2. **Create and activate environment**
```bash
uv venv
uv python pin 3.12
uv sync --extra dev --extra docs --upgrade
uv run pre-commit install
uv run python --version
```

3. **Activate (Windows)**
```bash
.\.venv\Scripts\activate
```

---

## ⚙️ Workflow 3. Daily Workflow

### 3.1 Pull latest updates
```bash
git pull
```

### 3.2 Keep things clean
```bash
uv sync --extra dev --extra docs --upgrade
uvx ruff check --fix
uv run pre-commit run --all-files
git add .
```

---

## ⚙️ Workflow 4. Save and Push Work

```bash
git add .
git commit -m "Update Project 3 models and results"
git push -u origin main
```

---

## ⚙️ Workflow 5. Documentation

Pushing to GitHub automatically builds your **MkDocs website** from `docs/` using `mkdocs.yml`.

---

## 🧩 How the Project Works

| Step | Description |
|------|-------------|
| 1 | Load Titanic dataset using seaborn |
| 2 | Clean missing values (age = median, embark_town = mode) |
| 3 | Add new feature: `family_size = sibsp + parch + 1` |
| 4 | Encode categories (`sex`, `embarked`, `alone`) to numbers |
| 5 | Define feature cases (alone, age, age + family_size) |
| 6 | Use `StratifiedShuffleSplit` for 80/20 train/test |
| 7 | Train Decision Tree models |
| 8 | Plot confusion matrices + decision trees |
| 9 | Train SVM models and visualize support vectors |
| 10 | Train Neural Network for Case 3 |
| 11 | Create summary table + reflections |

---

## 📊 Model Performance Summary (Test Results)

| Model Type | Case | Features Used | Accuracy | Precision | Recall | F1-Score | Notes |
|------------|------|----------------|----------|-----------|--------|----------|-------|
| **Decision Tree** | 1 | alone | 63% | 64% | 63% | 63% | Good balance |
| | 2 | age | 61% | 58% | 61% | 55% | Age alone weak |
| | 3 | age + family_size | 59% | 57% | 59% | 57% | Overfit (train=77%) |
| **SVM (RBF)** | 1 | alone | 63% | 64% | 63% | 63% | Similar to DT |
| | 2 | age | 63% | 66% | 63% | 52% | High precision, low recall |
| | 3 | age + family_size | 63% | 66% | 63% | 52% | Similar behavior |
| **Neural Network (MLP)** | 3 | age + family_size | **66%** | **65%** | **66%** | **65%** | Best overall |

---

## 💡 Key Insights

✅ Neural Network performed best: **66% accuracy**  
✅ Decision Trees easy to read but overfit on Case 3  
✅ `alone` works better than `age` alone  
✅ More features ≠ better accuracy — Case 3 overfit on training  
✅ SVMs are stable but miss many survivors (lower recall)


## 🚀 Next Steps

- Add features like `sex`, `fare`, `pclass`  
- Push summary table to README and MkDocs site  

---

## 🧾 Acknowledgements

- Instructor: **Dr. Denise Case**  
- Base Template: `applied-ml-template`  
- Dataset: **Titanic (Seaborn)**  
- Tools: VS Code, uv, Git, MkDocs, Scikit-Learn  


