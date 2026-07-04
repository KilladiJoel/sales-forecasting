import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
print("Loading historical processed data...")
X = pd.read_csv("data/X_data.csv")
y = pd.read_csv("data/y_data.csv")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training the production-ready Random Forest model...")
best_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) 
best_model.fit(X_train, y_train.values.ravel())
predictions = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, predictions))
r2 = r2_score(y_val, predictions)
print("\n================ HISTORICAL MODEL PERFORMANCE ================")
print(f"Model Architecture: Random Forest Regressor")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} units")
print("==============================================================")
print("\n📅 Generating future timeline for next month...")
future_dates = pd.date_range(start="2026-08-01", end="2026-08-31", freq='D')
unique_stores = X['store_nbr'].unique()
future_records = []
for store in unique_stores:
    for date in future_dates:
        future_records.append({
            'date': date,
            'store_nbr': store,
            'onpromotion': 0  
        })

df_future = pd.DataFrame(future_records)
store_profiles = X[['store_nbr', 'city', 'type_store']].drop_duplicates()
df_future = pd.merge(df_future, store_profiles, on='store_nbr', how='left')
df_future['day_of_week'] = df_future['date'].dt.dayofweek
df_future['month'] = df_future['date'].dt.month
df_future['day'] = df_future['date'].dt.day
if 'type_holiday' in X.columns:
    df_future['type_holiday'] = 0
X_future = df_future.drop(columns=['date'])
X_future = X_future[X.columns]
print("🔮 Simulating future demand from memory splits...")
df_future['predicted_sales'] = best_model.predict(X_future)
total_stock_needed = df_future.groupby('store_nbr')['predicted_sales'].sum().reset_index()
total_stock_needed.columns = ['Store Number', 'Total Predicted Stock Needed']
total_stock_needed['Total Predicted Stock Needed'] = total_stock_needed['Total Predicted Stock Needed'].round().astype(int)

print("\n================ TOTAL STOCK NEEDED FOR NEXT MONTH ================")
print(total_stock_needed.to_string(index=False))
print("===================================================================")