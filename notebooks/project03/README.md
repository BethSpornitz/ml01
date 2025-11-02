# 🛳 Project 3 – Titanic Classification Models  
**Author:** Beth Spornitz  
**Goal:** Predict which passengers survived the Titanic using 3 machine learning models and compare their performance.

---

## 📁 Dataset
We used the built-in Titanic dataset from Seaborn. Features were cleaned and prepared by:
- Filling missing ages with the median  
- Filling missing towns with the mode  
- Creating `family_size = sibsp + parch + 1`  
- Converting `sex`, `embarked`, `alone` into numeric codes  

---

## 📊 Model Performance Summary

| Model Type            | Case   | Features Used        | Accuracy | Precision | Recall | F1-Score | Notes |
|-----------------------|--------|------------------------|----------|-----------|--------|----------|-------|
| Decision Tree         | Case 1 | alone                 | 63%      | 64%       | 63%    | 63%      | Most balanced/simple model |
|                       | Case 2 | age                   | 61%      | 58%       | 61%    | 55%      | Age alone is weak predictor |
|                       | Case 3 | age + family_size     | 59%      | 57%       | 59%    | 57%      | Overfitting (train=77%)     |
| SVM (RBF Kernel)      | Case 1 | alone                 | 63%      | 64%       | 63%    | 63%      | Similar to Decision Tree    |
|                       | Case 2 | age                   | 63%      | 66%       | 63%    | 52%      | High precision, low recall  |
|                       | Case 3 | age + family_size     | 63%      | 66%       | 63%    | 52%      | Same pattern as Case 2      |
| Neural Network (MLP)  | Case 3 | age + family_size     | 66%      | 65%       | 66%    | 65%      | Best overall performance     |

---

## ✅ Key Insights (Human Language)

- Using **just one feature (alone or age)** gives weak results.  
- **Adding family size helps a little**, but models are still limited.  
- **Neural Network did worse** than Decision Tree and SVM (probably needs more features like sex, class, fare).  
- The models often predict “did not survive” because most people actually died (class imbalance).

---

## ⚠ Challenges I Faced
- Understanding what each model was doing “behind the scenes”  
- Confusion about how to convert categorical values (male → 0, female → 1)  
- Decision trees looked messy — had to change font size, dpi, etc.  
- Making sure I didn’t break instructor’s original code structure

---

## 🚀 Next Steps (Improvements)
✔ Add better features: `sex`, `pclass`, `fare`  
✔ Try scaling features for SVM/Neural Network  
✔ Try hyperparameters (max_depth, C, gamma, hidden layers, etc.)  
✔ Use cross-validation instead of one train/test split  
✔ Try more advanced models (Random Forest, Gradient Boosting)

---

## 📎 Files In This Project
