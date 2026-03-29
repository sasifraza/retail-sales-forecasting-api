from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model/retail_forecast_model.pkl")


@app.get("/")
def home():
    return {"message": "Retail Forecast API running"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"predicted_sales": float(pred)}