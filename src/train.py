import pandas as pd
import numpy as np


from xgboost import XGBRegressor 


from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# Load Data 

df =pd.read_csv("data/retail.csv",parse_dates=["date"])

# Create time-based features 

df["year"] = df["date"].dt.year
df["month"]= df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek

# Create lag features (IMPORTANT)
df["lag_1"] = df["sales"].shift(1)
df["lag_2"] = df["sales"].shift(2)
df["lag_3"] = df["sales"].shift(3)

# Drop missing values created by lag
df = df.dropna().reset_index(drop=True)

#  Sort by time 

df =df.sort_values("date").reset_index(drop=True)

# Features and target 

X = df[["year", "month", "day", "dayofweek", "lag_1", "lag_2", "lag_3"]]
y= df["sales"]

# Time-based train/test split 

split_index = int(len(df)*0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Train model 

from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train,y_train)

# predict 
preds = model.predict(X_test)

# Metrics 

mae = mean_absolute_error(y_test,preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("Train size", len(X_train))
print("Test size", len(X_test))

print (f"MAE, {mae:.2f}")
print(f"RMSE , {rmse:.2f}")

# Save Model 

os.makedirs("model", exist_ok= True)
joblib.dump(model, "model/retail_forecast_model.pkl")
print("Model saved to model/retail_forecast_model.pkl")

#  Plot  actual versus predicted 

import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label = "Actual")
plt.plot(preds, label ="Predicted")
plt.title("Actual Versus Predicted Sales")
plt.xlabel("Test Period")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()


