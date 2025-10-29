Ml02 – Titanic Data Features

Author: Beth Spornitz
Date: October 27, 2025

Project Overview

This project explores and prepares the Titanic dataset from Seaborn for machine learning analysis. It demonstrates essential data preparation workflows including: data inspection and visualization, handling missing values, feature engineering and encoding, feature selection and justification, splitting data into training and test sets, and documentation and version control using Git and GitHub. The goal is to create a clean, fully numeric dataset ready for classification modeling to predict passenger survival.

⚙️ WORKFLOW 1. Set Up Machine
1. Verify You Have These Installed  
    - VS Code (with Python, Pylance, Jupyter, and Ruff extensions)  
    -  Git  
    - uv – the environment and dependency manager  

⚙️ WORKFLOW 2. Set Up Project  

Once your environment is ready, follow these steps to set up your project.

1.  Clone the Repository: git clone https://github.com/BethSpornitz/ml-bethspornitz

2. Create and Activate Your Virtual Environment  
   ```bash
   uv venv
   uv python pin 3.12
   uv sync --extra dev --extra docs --upgrade
   uv run pre-commit install
   uv run python --version
   ```

3.  Activate it: 
```
..venv\Scripts\activate
```

## ⚙️ WORKFLOW 3. Daily Workflow
When working on the project, always start by opening the **project folder** in VS Code — *not* your global `Repos` folder.

### 3.1 Pull the Latest Updates from GitHub
```
git pull
```
### 3.2 Run Checks as You Work
Keep your environment and code clean by running these commands regularly:

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


⚙️ WORKFLOW 4. Version Control

After making progress, save and push your work.
### 4.1 Git Add, Commit, and Push: 
```
git add . 
git commit -m "Add Titanic project updates"  
git push -u origin main
```

⚙️ WORKFLOW 5. Build Documentation

Your project automatically builds a professional documentation site using MkDocs whenever you push to GitHub. The site is built from the docs/ folder and configured by mkdocs.yml.

🧩 How the Project Works  
Step 1: Load Data – Uses the Titanic dataset directly from Seaborn  
Step 2: Explore – Inspect column types, missing values, and correlations  
Step 3: Visualize – Scatter plots, histograms, and count plots  
Step 4: Clean – Fill missing values using median and mode  
Step 5: Engineer – Create family_size and encode categorical variables  
Step 6: Split – Use both random and stratified train/test splits  

📈 Example Output  
Survivors: Percentage of passengers who survived (~38%)  
Missing Values: Cleaned age and embark_town fields (0 remaining)  
Engineered Features: family_size, encoded sex, embarked, and alone (verified)  
Class Balance: Stratified split maintains survival ratios (balanced train/test sets)  

Interpretation: Sex and passenger class are the strongest predictors of survival, consistent with historical reports. Stratified sampling preserves class proportions, ensuring fair model evaluation.

🧾 Acknowledgements
Instructor: Dr. Denise Case
Base Template: applied-ml-template
Dataset: Titanic (Seaborn)