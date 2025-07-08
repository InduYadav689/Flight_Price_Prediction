# ✈️ Flight Price Prediction using Machine Learning

This project predicts flight ticket prices based on features such as airline, duration, source/destination, stops, and booking lead time. Using a dataset of over *3 million real-world records, several regression algorithms were applied to develop an accurate and scalable solution. The model is deployed using **Streamlit* for real-time interaction.

---

## 📌 Key Features

- Data cleaning and preprocessing
- Feature engineering
- Trained and compared multiple regression models:
  - Random Forest
  - XGBoost
  - Decision Tree
  - CatBoost
- Evaluation metrics:
  - R² (R-squared)
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - Both Train-Test and Cross- validation methods for checking Accuracy
- Interactive *Streamlit app* interface

---

## 📊 Dataset Summary

- *Total Rows*: ~3,152,000
- *Columns*:
  - Serial Number
  - Airline
  - Flight Number
  - Duration
  - Source City
  - Destination City
  - Stops
  - Days Left
  - Price

> The dataset is publicly available in this repository under flight_prediction_data.csv.

---

## 🚀 How to Run This Project

```bash
# 1. Clone the repository
# ✈️ Flight Price Prediction using Machine Learning

This project predicts flight ticket prices based on features such as airline, duration, source/destination, stops, and booking lead time. Using a dataset of over *3.15 million real-world records, several regression algorithms were applied to develop an accurate and scalable solution. The model is deployed using **Streamlit* for real-time interaction.

---

## 📌 Key Features

- Data cleaning and preprocessing
- Feature engineering
- Trained and compared multiple regression models:
  - Random Forest
  - XGBoost
  - Decision Tree
  - CatBoost
- Evaluation metrics:
  - R² (R-squared)
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
- Interactive *Streamlit app* interface

---

## 📊 Dataset Summary

- *Total Rows*: ~300152
- *Columns*:
  - Serial Number
  - Airline
  - Flight Number
  - Duration
  - Source City
  - Destination City
  - Stops
  - Days Left
  - Price

> The dataset is publicly available in this repository under dataset.csv.

---
### 📦 Trained Model (Download Link)

Due to GitHub's file size restrictions, the trained machine learning model (random_forest_model.pkl, approx. 875 MB) has been securely hosted on Google Drive.

You can download the model using the link below:

🔗 [Download Random Forest Model (flight.pkl - 875 MB)](https://drive.google.com/file/d/1YOqv7RgPatoYhkMJrzc8A_vScoAD4cGa/view?usp=sharing)

#### 📁 About the Model:
- The model was trained on a large dataset of over *3 million flight records*
- It uses a *Random Forest Regressor* for flight price prediction
- The model was selected based on the best performance metrics among several algorithms including XGBoost, Decision Tree, and CatBoost
- Evaluation metrics include: *R² Score, **MAE, **MSE, **RMSE*
- The .pkl file can be loaded in any Python environment using:
  
```python
import pickle

with open(flight.pkl', 'rb') as f:
    model = pickle.load(f)

## 🚀 How to Run This Project

```bash
# 1. Clone the repository
(https://github.com/InduYadav689/Flight_Price_Prediction)

# 2. Install required packages
pip install -r requirements.txt
like catboost, xgboost, matplotlib, seaborn

# 3. Run the Streamlit app
streamlit run app.py

# 2. Install required packages
pip install -r requirements.txt
like joblib, pickle for deploying the model in frontend.

# 3. Run the Streamlit app
streamlit run app.py
