# 🩺 Midterm Project – Diabetes Classification (Pima Indians)

**Author:** Beth Spornitz  
**Date:** November 3, 2025

---

## ⚙️ WORKFLOW 1. Set Up Machine

1. Verify You Have These Installed  
   - **VS Code** (with **Python**, **Pylance**, **Jupyter**, and **Ruff** extensions)  
   - **Git**  
   - **uv** – the environment and dependency manager

---

## ⚙️ WORKFLOW 2. Set Up Project

Once your environment is ready, follow these steps to set up your project.

### 2.1 Clone the Repository
```bash
git clone https://github.com/BethSpornitz/ml-bethspornitz
```

### 2.2 Create and Configure Your Virtual Environment (uv)
```bash
uv venv
uv python pin 3.12
uv sync --extra dev --extra docs --upgrade
uv run pre-commit install
uv run python --version
```

### 2.3 Activate the Environment
```bash
.\.venv\Scripts\activate   # Windows PowerShell
# or
source .venv/bin/activate  # macOS/Linux
```

---

## ⚙️ WORKFLOW 3. Daily Workflow

### 3.1 Pull the Latest Updates from GitHub
```bash
git pull
```

### 3.2 Run Checks as You Work
Keep your environment and code clean by running these commands regularly:
```bash
git pull
uv sync --extra dev --extra docs --upgrade
uv cache clean
git add .
uvx ruff check --fix
uvx pre-commit autoupdate
uv run pre-commit run --all-files
git add .
uv run pytest
```

---

## ⚙️ WORKFLOW 4. Version Control

After making progress, save and push your work.

### 4.1 Git Add, Commit, and Push
```bash
git add .
git commit -m "Midterm: diabetes classification updates"
git push -u origin main
```

---

## ⚙️ WORKFLOW 5. Build Documentation

Your project automatically builds a professional documentation site using **MkDocs** whenever you push to GitHub.  
The site is built from the `docs/` folder and configured by `mkdocs.yml`.

---

## 🧩 How the Project Works

**Step 1: Load Data** – Uses the **Pima Indians Diabetes** dataset (`data/diabetes.csv`) https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?select=diabetes.csv   
**Step 2: Explore** – Inspect column types, class balance, and distributions.  
**Step 3: Clean** – Replace *biologically impossible zeros* with `NaN` and **impute medians** for:  
`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`.  
**Step 4: Engineer** – Create **BMI_Category** (0=Underweight, 1=Normal, 2=Overweight, 3=Obese) for interpretability.  
**Step 5: Split** – **Stratified 80/20** split to preserve class ratios.  
**Step 6: Model** – Train and evaluate:
- **Decision Tree** (Cases: BMI, Glucose, Glucose+BMI)
- **SVC (RBF)** (Cases: BMI, Glucose, Glucose+BMI)

---

## 📈 Example Output

**Class Distribution:** ~**65%** no diabetes (0), **35%** diabetes (1).  
**Data Cleaning:** All invalid zeros in clinical fields replaced and **no missing values remain**.  
**Engineered Features:** `BMI_Category` added (used for interpretation; modeling used continuous BMI).  
**Modeling Cases:**  
- **Case 1:** BMI  
- **Case 2:** Glucose  
- **Case 3:** Glucose + BMI

**Verified Performance Summary (test set):**
| Model Type        | Case | Features Used     | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Notes |
|-------------------|------|-------------------|----------|----------------------|------------------|---------------------|-------|
| Decision Tree     | 1    | BMI               | 0.62     | 0.45                 | 0.37             | 0.41                | BMI alone is weak |
| Decision Tree     | 2    | Glucose           | 0.67     | 0.55                 | 0.31             | 0.40                | Glucose improves prediction |
| Decision Tree     | 3    | Glucose + BMI     | 0.71     | 0.59                 | 0.56             | 0.57                | ✅ Best Decision Tree |
| SVC (RBF Kernel)  | 1    | BMI               | 0.65     | 0.50                 | 0.15             | 0.23                | Very low recall |
| SVC (RBF Kernel)  | 2    | Glucose           | 0.69     | 0.61                 | 0.37             | 0.46                | Better than BMI |
| SVC (RBF Kernel)  | 3    | Glucose + BMI     | 0.70     | 0.62                 | 0.39             | 0.48                | Best SVC case |

**Interpretation:**
- **Glucose is the strongest single predictor**; BMI alone underperforms.
- **Combining Glucose + BMI (Case 3)** yields the best overall performance.
- **Decision Tree Case 3** achieved **~71% accuracy** and the **best recall** for diabetics among the tested models.
- **SVC (RBF)** benefits from **feature scaling** (not applied here to mirror the Project 3 pattern), which likely limits its recall.

---

## 🧪 Reproducibility Notes

- **Random State:** Stratified splits use a fixed `random_state=42`.  
- **Imputation:** Medians are computed **after** replacing zeros with `NaN`.  
- **Consistency:** Code structure mirrors the **Titanic Project 3** style (Cases 1–3, same evaluation flow).

---

## 🧾 Acknowledgements

- **Instructor:** Dr. Denise Case  
- **Base Template:** `applied-ml-template`  
- **Dataset:** Pima Indians Diabetes (UCI/Kaggle)

---
