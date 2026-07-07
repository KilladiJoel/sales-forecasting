import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
print("Loading data...")
X = pd.read_csv("../data/X_data.csv")
y = pd.read_csv("../data/y_data.csv")
sample_fraction = 0.10
sample_indices = X.sample(frac=sample_fraction, random_state=41).index
X = X.loc[sample_indices]
y = y.loc[sample_indices]
print("Downsampled dataset size:", X.shape[0], "rows")
print("Splitting data...")
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
print("Scaling features...")
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)
k_value = [5, 15, 30]
for k in k_value:
    print(f"\nEvaluating K = {k}...")
    knn = KNeighborsRegressor(n_neighbors=k, n_jobs=-1, algorithm='kd_tree')
    print("Training model...")
    knn.fit(x_train_s, y_train)
    print("Running predictions...")
    y_pred = knn.predict(x_test_s)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"Results for K={k}:")
    print("R2 Score :", r2)
    print("RMSE     :", rmse)
print("\nAll steps completed!")