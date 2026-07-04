import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
X=pd.read_csv("../data/X_data.csv")
y=pd.read_csv("../data/y_data.csv")
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=41)
model=StandardScaler()
x_train_s=model.fit_transform(x_train)
x_test_s=model.transform(x_test)
k_value=[5,15,30]
for k in k_value:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(x_train_s, y_train)
    y_pred = knn.predict(x_test_s)
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_test,y_pred)
    print("Model Performance:")
    print("R2 Score:",r2)
    print("RMSE",rmse)