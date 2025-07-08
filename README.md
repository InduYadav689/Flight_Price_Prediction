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

> The dataset is publicly available in this repository under dataset.csv.

---

## 🚀 How to Run This Project

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/Flight_Price_Prediction.git

# 2. Install required packages
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
