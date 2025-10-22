# ML01 – California Housing Price Prediction

**Author:** Beth Spornitz  
**Date:** October 22, 2025  

---

## Overview
This project predicts **median house prices in California** using Scikit-learn’s California Housing dataset.  

It demonstrates:  
- Data exploration and visualization  
- Feature selection  
- Linear regression modeling  
- Evaluation using R², MAE, and RMSE metrics  
- Automated documentation with MkDocs  

---

## Quick Links
- [Notebook: ML01](https://github.com/BethSpornitz/ml01/blob/main/notebooks/project01/ml01.ipynb)
- [Project README](https://github.com/BethSpornitz/ml01/blob/main/README.md)
- [GitHub Repository](https://github.com/BethSpornitz/ml01)

---

## Summary of Methods
| Step | Task | Description |
|------|------|--------------|
| 1 | Load Data | Loads the California housing dataset using `fetch_california_housing()` |
| 2 | Explore | Displays column types, missing values, and summary statistics |
| 3 | Visualize | Creates histograms, boxenplots, and scatterplots for features |
| 4 | Model | Fits a Linear Regression model using `MedInc` and `AveRooms` |
| 5 | Evaluate | Calculates R², MAE, and RMSE to assess prediction accuracy |

---

## Example Results
| Metric | Meaning | Example Value |
|---------|----------|---------------|
| R² | How well the model explains price variation | 0.62 |
| MAE | Average error between predictions and actual prices | 0.55 |
| RMSE | Larger errors penalized more heavily | 0.73 |

Interpretation:  
Median income (`MedInc`) shows the strongest correlation with house prices.  
Average rooms (`AveRooms`) adds moderate predictive power.

---

## Acknowledgments
Instructor: **Dr. Denise Case**  
Base Template: *applied-ml-template*  
Dataset: *California Housing Dataset (Scikit-learn)*  

