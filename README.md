# 🏠 House Price Prediction

A machine learning project that predicts house sale prices using regression models.  
Built with Python, scikit-learn, and real-world housing data from Kaggle.

---

## 📌 Project Overview

Predicting house prices is a classic regression problem in machine learning.  
In this project, I trained and compared **3 regression models** on the famous  
[Kaggle House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)  
to find which one best predicts sale prices.

---

## 🗂️ Project Structure

```
house-price-prediction/
│
├── house_price_prediction.ipynb   ← Main notebook (EDA + Modeling)
├── train.csv                      ← Dataset (download from Kaggle)
└── README.md                      ← You are here
```

---

## 📊 Dataset

| Detail | Info |
|---|---|
| Source | Kaggle — House Prices: Advanced Regression Techniques |
| Rows | 1,460 houses |
| Columns | 79 features |
| Target | `SalePrice` (house sale price in USD) |

---

## 🔄 Workflow

```
Step 1 → Import Libraries
Step 2 → Load Dataset
Step 3 → Exploratory Data Analysis (EDA)
Step 4 → Data Preprocessing
Step 5 → Train 3 Models + Compare
Step 6 → Feature Importance (Random Forest)
Step 7 → Actual vs Predicted + Residual Plot
Step 8 → Final Summary
```

---

## 📈 EDA Highlights

- **SalePrice** is right-skewed → Log transformation makes it normally distributed
- **OverallQual** (overall quality) has the strongest correlation with price
- **GrLivArea** (above-ground living area) is the #2 predictor
- Top features: `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `1stFlrSF`

---

## 🤖 Models Trained

| Model | R² Score | RMSE |
|---|---|---|
| Linear Regression | Baseline | Higher |
| Decision Tree | Better | Lower |
| **Random Forest** | **Best 🏆** | **Lowest** |

> **Random Forest** wins — it handles non-linear relationships and feature interactions  
> better than Linear Regression and avoids the overfitting issue of a single Decision Tree.

---

## 🏆 Key Findings

- `OverallQual` is the single most important feature for predicting house prices
- `GrLivArea` (living area sq ft) is the 2nd most important feature
- Random Forest outperformed all other models on both R² Score and RMSE
- Tree-based models are more suitable for this dataset due to non-linear patterns

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Machine learning models |
| Google Colab | Development environment |

---

## 🚀 How to Run

**1. Open in Google Colab**
Click the link below to open the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZExKkalHtoG_w_rSUd9HUXFODmsCDpuy)

**2. Download the dataset**
Download `train.csv` from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) and upload it to your Colab session.

**3. Run all cells**
Go to **Runtime → Run All** and you're done!

---

## 📬 Connect with Me

- 💼 [LinkedIn] https://linkedin.com/in/sharda-vatsal-bhat-73b037295
- 🐙 [GitHub] https://github.com/Sharda2004196

---

*Built by Sharda Vatsal Bhat — Aspiring AI/ML Engineer*
