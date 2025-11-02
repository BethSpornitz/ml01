# Ml03 – Titanic Classification Models

**Author:** Beth Spornitz  
**Date:** November 2025  

---

## 🎯 Project Overview

This project uses the Titanic dataset from Seaborn to **build and evaluate three machine learning classification models**:

- **Decision Tree**
- **Support Vector Machine (SVM)**
- **Neural Network (MLPClassifier)**

Each model tries to predict whether a Titanic passenger **survived (1)** or **did not survive (0)** using three different input cases:

| Case | Features Used          |
|------|-------------------------|
| 1    | `alone`                |
| 2    | `age`                  |
| 3    | `age + family_size`   |

The project includes:  
✔ Feature engineering and encoding  
✔ Stratified train/test splits  
✔ Decision Trees + Confusion Matrices  
✔ SVM models with support vector visualization  
✔ Neural Network with decision surface visualization  
✔ Performance summary in a Markdown table  
✔ Reflections after each section  

---

## ⚙️ Workflow 1. Set Up Machine

Make sure you have:

- **VS Code** (with Python, Jupyter, Pylance, Ruff extensions)
- **Git**
- **uv** (Python environment + dependency manager)

---

## ⚙️ Workflow 2. Set Up Project

1. **Clone your repository**
```bash
git clone https://github.com/BethSpornitz/ml-bethspornitz
