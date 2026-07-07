from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

X = pd.read_csv("data/X_data.csv")
y = pd.read_csv("data/y_data.csv")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
model = DecisionTreeRegressor(random_state=41)
model.fit(x_train, y_train.values.ravel())

unique_stores = sorted([int(s) for s in X['store_nbr'].unique()])
store_profiles = X[['store_nbr', 'city', 'type_store']].drop_duplicates()

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_store = None
    total_stock = None
    avg_daily = None
    dates_js = []
    sales_js = []
    
    if request.method == 'POST':
        selected_store = int(request.form.get('store_nbr'))
        
        future_dates = pd.date_range(start="2026-08-01", end="2026-08-31", freq='D')
        future_records = [{'date': d, 'store_nbr': selected_store, 'onpromotion': 0} for d in future_dates]
        df_future = pd.DataFrame(future_records)
        
        df_future = pd.merge(df_future, store_profiles, on='store_nbr', how='left')
        df_future['day_of_week'] = df_future['date'].dt.dayofweek
        df_future['month'] = df_future['date'].dt.month
        df_future['day'] = df_future['date'].dt.day
        
        if 'type_holiday' in X.columns:
            df_future['type_holiday'] = 0
            
        X_future = df_future.drop(columns=['date'])[X.columns]
        df_future['predicted_sales'] = model.predict(X_future)
        
        raw_total = int(df_future['predicted_sales'].sum().round())
        raw_avg = int(df_future['predicted_sales'].mean().round())
        
        total_stock = f"{raw_total:,}"
        avg_daily = f"{raw_avg:,}"
        
        dates_js = df_future['date'].dt.strftime('%Y-%m-%d').tolist()
        sales_js = [float(val) for val in df_future['predicted_sales'].round(1).tolist()]

    return render_template('index.html', 
                           stores=unique_stores, 
                           selected_store=selected_store,
                           total_stock=total_stock,
                           avg_daily=avg_daily,
                           dates_js=dates_js,
                           sales_js=sales_js)

if __name__ == '__main__':
    app.run(debug=True)