# Project 5 – Ensemble Machine Learning (Wine Dataset)
**Author:** Beth Spornitz  
**Date:** November 2025  
---

## Project Overview
This project explores **ensemble machine learning** methods using the **UCI Wine Quality** dataset.  
The goal is to predict the perceived quality of red wine (low / medium / high) from 11 physicochemical features such as acidity, alcohol, sulphates, and density.  

Ensemble models combine the outputs of multiple algorithms to create stronger, more generalizable predictors.  
We evaluated several ensemble techniques and focused on two high-performing approaches:  

- **Gradient Boosting** – Sequentially trains weak learners to correct earlier errors.  
- **Voting Classifier (DT + SVM + NN)** – Combines diverse models by averaging their probabilities.  

---

## Dataset Information
**Source:** [UCI Machine Learning Repository – Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  
- 1599 samples of Portuguese red wine  
- 11 numeric features + 1 target variable (`quality`)  

### Feature Examples
| Feature | Description |
|:--|:--|
| fixed acidity | Tartaric acid level |
| volatile acidity | Acetic acid (vinegar) |
| citric acid | Adds freshness |
| residual sugar | Sweetness after fermentation |
| chlorides | Salt content |
| free/total SO₂ | Microbial protection |
| density | Related to sugar content |
| pH | Acidity indicator |
| sulphates | Antioxidant and stabilizer |
| alcohol | % ABV |

### Target Transformation
The original quality (0–10) was grouped into three classes:  
- **Low:** 3–4  **Medium:** 5–6  **High:** 7–8  

---

## ⚙️ Workflow Summary
1. **Load and inspect** dataset, confirm shape and missing values.  
2. **Prepare data:** created categorical and numeric quality labels.  
3. **Select features and target.**  
4. **Split train/test** with stratified sampling to preserve class balance.  
5. **Train and evaluate models:** Gradient Boosting and Voting Classifier.  
6. **Compare results,** analyze feature importance and performance gaps.  
7. **Interpret insights and form conclusions.**  

---

## 📊 Model Results

### Gradient Boosting (100 Estimators)
| Metric | Train | Test |
|:--|:--|:--|
| Accuracy | 0.960 | 0.856 |
| F1 Score | 0.958 | 0.841 |

**Confusion Matrix (visualized in bar + heatmap):**  
Most “medium” wines were classified correctly. Misclassifications mainly occurred between adjacent categories (medium ↔ high).  

**Feature Importance (top predictors):** `alcohol`, `sulphates`, `volatile acidity`  

---

### Voting Classifier (DT + SVM + NN)
| Metric | Train | Test |
|:--|:--|:--|
| Accuracy | 0.892 | 0.835 |
| F1 Score | 0.884 | 0.827 |

**Observation:** The Voting Classifier was slightly less accurate but more balanced across classes, suggesting better generalization.  

---

## 🧩 Comparison and Visualization
- **Feature Importance Plot:** Showed that alcohol and sulphates positively influenced predicted quality, while volatile acidity had a negative effect.  
- **Permutation Importance / Partial Dependence Plots:** Demonstrated how quality scores increased with higher alcohol and moderate sulphates.  
- **Gap Analysis:** Train–Test Accuracy Gap = 0.104 for Gradient Boosting → Model generalized well with mild overfitting.  

---

## Conclusions and Insights
Gradient Boosting produced the highest accuracy and F1 scores, capturing complex non-linear relationships between chemical features and perceived quality.  
The Voting Classifier offered stable results and reduced variance through model diversity.  

**Best Model:** Gradient Boosting  
**Why:** High predictive power, small generalization gap, interpretable feature importance.  

If this were used by a winery, they could focus on monitoring alcohol, sulphates, and volatile acidity levels to consistently produce higher-quality wine.  

---

## 🏁 Next Steps
- Tune hyperparameters (n_estimators, learning_rate, max_depth).  
- Apply cross-validation for robust performance estimation.  
- Use SHAP or Permutation Importance for deeper interpretability.  
- Extend analysis to white wine dataset for comparison.  

---

## 📂 Repository Structure
notebooks/
└── project05/
├── ensemble_bethspornitz.ipynb
├── data/
│ └── winequality-red.csv
└── README.md