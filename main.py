if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor  
    from sklearn.metrics import mean_squared_error, r2_score
    X = pd.read_csv("data/X_data.csv")
    y = pd.read_csv("data/y_data.csv")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training the production-ready Random Forest model (this might take a minute)...")
    best_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) 
    best_model.fit(X_train, y_train.values.ravel())
    predictions = best_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)
    print("\n================ FINAL MODEL PERFORMANCE ================")
    print(f"Final Model: Random Forest Regressor")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f} units")
    print("=========================================================")