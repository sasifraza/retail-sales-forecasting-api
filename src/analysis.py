import pandas as pd

# load dataset 

df =pd.read_csv("data/retail.csv", parse_dates=["date"])

# create time-based features (SQL-style)

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

print("Dataset shape:")
print(df.shape)

print("\n First 5 rows")
print(df.head())

# Total sales (SQL:SUM)
print("\n Total sales:")
print(df["sales"].sum())

# Average sales (SQL: AVG)

print("\n Average daily sales")
print(df["sales"].mean())

# Monthly aggregation (SQL, Group by)
print("\n Monthly sales summary")
monthly_sales= (
df.groupby(["year","month"],as_index=False)["sales"]
.sum()
.sort_values(["year", "month"])    
)

print(monthly_sales)

# Top 10 highest sales days (SQL: order by DESC LIMIT 10)

print("\n Top 10 highest -sales days:")

top_days = df.sort_values("sales", ascending=False).head(10)
print(top_days[["date", "sales"]])

# Data Visualiztion 
# plot daily sales trend
import matplotlib.pyplot as plt
# plot daily sales trends 
plt.figure()
plt.plot(df["date"],df["sales"])
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# Plot monthly aggregates sales 
plt.figure()
plt.plot(monthly_sales["month"],monthly_sales["sales"])
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()


