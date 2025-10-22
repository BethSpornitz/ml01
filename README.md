# Ml01 – California Housing Price Prediction  

**Author:** Beth Spornitz  
**Date:** October 22, 2025  

---

## Project Overview  

This project predicts median house prices in California using the California Housing Dataset from Scikit-learn.  
It demonstrates fundamental machine learning workflows including:
- Data exploration  
- Visualization  
- Feature selection  
- Model training and evaluation  
- Documentation with MkDocs  
- Version control using Git and GitHub  

The model uses Linear Regression to find patterns between income, rooms, and house prices.

---

## ⚙️ WORKFLOW 1. Set Up Machine  

### 1. Verify You Have These Installed
- **VS Code** (with Python, Pylance, Jupyter, and Ruff extensions)
- **Git**
- **uv** – the environment and dependency manager

## ⚙️ WORKFLOW 2. Set Up Project  
Once your environment is ready, follow these steps to set up your project.  

1. Clone the Repository
```git clone https://github.com/BethSpornitz/ml01.git
```

2. Create and Activate Your Virtual Environment
```
uv venv  
uv python pin 3.12  
uv sync --extra dev --extra docs --upgrade  
uv run pre-commit install  
uv run python --version
```    

Activate it:
```
.\.venv\Scripts\activate
```   

## ⚙️ WORKFLOW 3. Daily Workflow  
When working on the project, always start by opening the project folder in VS Code, not your global Repos folder.

3.1 Git Pull from GitHub  
```
git pull
```     
3.2 Run Checks as You Work  
```
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
💡 Run uv run pre-commit run --all-files twice if the first run fixes files automatically.  

3.3 Build Project Documentation  
```
uv run mkdocs build --strict  
uv run mkdocs serve
```    

3.4 Execute  
Run your analysis script directly from VS Code terminal:  
```
uv run python notebooks/project01/ml01.py
```   

## ⚙️ WORKFLOW 4. Version Control  
After making progress, save and push your work.  

4.1 Git Add, Commit, and Push  
```
git add .  
git commit -m "Your update description here"  
git push -u origin main
```      


## ⚙️ WORKFLOW 5. Build Documentation  
Your project automatically builds a professional documentation site using MkDocs whenever you push to GitHub.  

The site is built from the docs/ folder and configured by mkdocs.yml.  

🧩 How the Project Works  
Step	Task	Description  
1	Load Data	Uses fetch_california_housing from Scikit-learn  
2	Explore	Check column types, nulls, and summary statistics  
3	Visualize	Create histograms, boxenplots, and scatterplots  
4	Model	Linear Regression with MedInc and AveRooms  
5	Evaluate	R², MAE, RMSE used to assess performance  

📈 Example Model Output  
Metric	Description	Example Result  
R²	Variance explained by model	~0.62  
MAE	Avg. prediction error	~0.55  
RMSE	Penalized large errors	~0.73  

Interpretation:  
Median income (MedInc) has the strongest relationship to house prices, while average rooms (AveRooms) adds moderate predictive power.  

🧾 Acknowledgements  
Instructor: Dr. Denise Case  

Base Template: applied-ml-template  

Dataset: California Housing (Scikit-learn)  
