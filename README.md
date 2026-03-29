# Retail Sales Forecasting & Analytics API

End-to-end time-series forecasting project that builds a retail sales model and exposes it as a REST API using FastAPI.

---

## Problem

Forecast future retail sales using historical time-based data.

---

## Dataset

Synthetic retail time-series dataset generated programmatically.

File:
data/retail.csv

---

## Pipeline

1. Generate time-series data
2. Perform SQL-style analysis using pandas
3. Feature engineering (year, month, day, dayofweek)
4. Add lag features (lag_1, lag_2, lag_3)
5. Train forecasting models
6. Evaluate using MAE and RMSE
7. Save model
8. Deploy via FastAPI

---

## Analysis Performed

- Total sales
- Average daily sales
- Monthly sales aggregation
- Top 10 highest sales days
- Trend visualization
- Actual vs predicted comparison

---

## Models Compared

### Linear Regression + Lag Features
MAE: 10.33  
RMSE: 12.95  

### XGBoost
MAE: 13.11  
RMSE: 16.41  

Result: Linear model outperformed XGBoost on this dataset.

---

## API

Run locally:

uvicorn api:app --reload

Open:

http://127.0.0.1:8000/docs

---

## Example Request

{
  "year": 2021,
  "month": 6,
  "day": 10,
  "dayofweek": 3,
  "lag_1": 200,
  "lag_2": 190,
  "lag_3": 180
}

---

## Example Response

{
  "predicted_sales": 172.26
}

---

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- FastAPI
- Uvicorn

---

## Key Takeaways

- Built full forecasting pipeline
- Used lag features for time-series modeling
- Compared multiple models
- Deployed model as REST API
- Demonstrated end-to-end ML workflow

---

## Author

Syed Asif Raza