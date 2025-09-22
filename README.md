# 🏡 Boston House Price Prediction

Predict house prices in Boston using a Gradient Boosting Regressor deployed as an interactive Streamlit web app.

---

## 📊 Project Overview

This project demonstrates an **end-to-end machine learning workflow**:

### Data Analysis & Cleaning
- Handled missing values:
  - **Mean** for normally distributed data  
  - **Median** for skewed data  
  - **Mode** for categorical columns  
- Handled outliers using **IQR (Interquartile Range)**

### Feature Engineering
- Created `house_age = year_sold - year_built`  
- Label encoded categorical features (`view`, `condition`, `waterfront`) to reduce dimensionality

### Scaling
- Used **RobustScaler** for numeric features

---

## 🔧 Technologies Used
- Python 3.x  
- Pandas, NumPy  
- Scikit-learn (GradientBoostingRegressor, LabelEncoder, RobustScaler)  
- Joblib (for model & scaler serialization)  
- Streamlit (for deployment)  
- Matplotlib & Seaborn (for visualization)  

---

## 📈 Model Performance
- **Model:** Gradient Boosting Regressor  
- **R² Score:** 0.77 → explains ~77% of variance in house prices  

---

## 🚀 Deployment

The app is deployed with **Streamlit**. Users can input house features and get **real-time price predictions**.

### Run Locally
1. **Clone the repository:**
```bash
git clone https://github.com/siddhantrane31/House-price-prediction.git
cd House-price-prediction
