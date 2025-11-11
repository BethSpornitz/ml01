# ML04 – Titanic Fare Prediction (Regression)
**Author:** Beth Spornitz  
**Date:** November 10, 2025  

---

## Project Overview
This project continues the Titanic series, shifting from classification to regression. Instead of predicting survival, we predict a continuous numeric target, passenger fare,  using different sets of input features.  

The project compares four regression approaches:  
- **Linear Regression**  
- **Ridge Regression**  
- **Elastic Net Regression**  
- **Polynomial Regression**  

Each model predicts `fare` based on several feature cases:  

| Case | Features Used |
|------|----------------|
| 1 | `age` |
| 2 | `family_size` |
| 3 | `age + family_size` |
| 4 | `sex_num`  |

The notebook includes:  
✔ Data preparation and feature creation  
✔ Multiple regression models  
✔ Regularization and polynomial expansion  
✔ Model comparison table  
✔ Visualizations and reflections  

---

## 📊 Dataset
The dataset is loaded directly from Seaborn:  

~~~python
import seaborn as sns
titanic = sns.load_dataset("titanic")
~~~

Key fields used:  
- **fare** — target variable (continuous)  
- **age**, **sex**, **sibsp**, **parch** — base features  
- **family_size** — engineered feature (`sibsp + parch + 1`)  
- **sex_num** — numeric encoding for sex (0 = male, 1 = female)  

---

## ⚙️ Workflow 1. Set Up Machine
Make sure you have:  
- VS Code (with Python, Jupyter, Pylance, and Ruff extensions)  
- Git  
- uv (Python environment and dependency manager)  

---

## ⚙️ Workflow 2. Set Up Project

1) **Clone repository**
~~~bash
git clone https://github.com/BethSpornitz/ml-bethspornitz
~~~

2) **Create and activate environment**
~~~bash
uv venv
uv python pin 3.12
uv sync --extra dev --extra docs --upgrade
uv run pre-commit install
uv run python --version
~~~

3) **Activate (Windows)**
~~~bash
.\.venv\Scripts\activate
~~~

---

## ⚙️ Workflow 3. Daily Workflow

### 3.1 Pull latest updates
~~~bash
git pull
~~~

### 3.2 Keep things updated
~~~bash
uv sync --extra dev --extra docs --upgrade
uvx ruff check --fix
uv run pre-commit run --all-files
git add .
~~~

---

## ⚙️ Workflow 4. Save and Push Work
~~~bash
git add .
git commit -m "Update Project 4 regression models and results"
git push -u origin main
~~~

---

## ⚙️ Workflow 5. Documentation
Pushing to GitHub automatically builds your **MkDocs** website from the `docs/` folder using `mkdocs.yml`.  

---

## 🧩 How the Project Works

| Step | Description |
|------|--------------|
| 1 | Load Titanic dataset from Seaborn |
| 2 | Fill missing values (`age` = median) and drop rows with missing `fare` |
| 3 | Create new feature `family_size = sibsp + parch + 1` |
| 4 | Encode `sex` to numeric (`sex_num`) |
| 5 | Define feature cases (age, family_size, both, and with sex) |
| 6 | Split data into 80/20 train/test sets |
| 7 | Train Linear, Ridge, Elastic Net, and Polynomial Regression models |
| 8 | Compare models using R², RMSE, and MAE |
| 9 | Visualize cubic and higher-degree polynomial fits |
| 10 | Reflect on findings and model behavior |

---

## 📈 Model Performance Summary (Case 4 – Comparison)

| Model Type | R² | RMSE | MAE | Notes |
|-------------|----|------|-----|-------|
| Linear Regression | 0.099 | 36.10 | 24.24 | Base model |
| Ridge Regression | 0.099 | 36.10 | 24.24 | Regularization had no effect |
| Elastic Net | 0.068 | 36.71 | 24.33 | Slightly worse fit |
| Polynomial (degree=3) | 0.099 | 36.10 | 24.24 | No improvement |

---

## 💡 Key Insights
- The models explained only about 10% of the variation in fares.  
- Sex had the strongest effect, with female passengers typically paying higher fares.  
- Adding model complexity or regularization did not improve accuracy.  
- Fare was difficult to predict because it depends on categorical factors like `pclass` and `embarked` that weren’t included in this case.  
- Extreme fare outliers made the predictions less stable.  

---

## 🚀 Next Steps
- Add features such as `pclass` and `embarked` to better explain fare differences.  
- Try predicting age instead of fare for a smoother target.   
- Experiment with advanced models (e.g., Random Forest, Gradient Boost) for comparison.  

---

## 🧾 Acknowledgements
- Instructor: **Dr. Denise Case**  
- Base Template: `applied-ml-template`  
- Dataset: **Titanic (Seaborn)**  
- Tools: VS Code, uv, Git, MkDocs, Scikit-Learn  
