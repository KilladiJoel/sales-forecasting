import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score
X=pd.read_csv("../data/X_data.csv")
y=pd.read_csv("../data/y_data.csv")
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=41)
model=DecisionTreeRegressor(random_state=42)
model.fit(x_train,y_train)
print("Model Trained")
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
print("Model Performance:")
print("R2 Score:",r2)
print("RMSE",rmse)