import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
train_df=pd.read_csv("../data/train.csv")
store_df=pd.read_csv("../data/stores.csv")
holidays_df=pd.read_csv("../data/holidays_events.csv")
merge_df=pd.merge(train_df,store_df,on="store_nbr",how="left")
final_df=pd.merge(merge_df,holidays_df,on="date",how="left",suffixes=("_store",'_holiday'))
final_df["type_holiday"] = final_df["type_holiday"].fillna('Normal Day')
final_df["city"]=final_df["city"].astype("category").cat.codes
final_df["type_store"]=final_df["type_store"].astype("category").cat.codes
final_df["type_holiday"]=final_df["type_holiday"].astype("category").cat.codes
features=["store_nbr","city","type_store","type_holiday"]
X=final_df[features]
y=final_df["sales"]
X.to_csv("../data/X_data.csv",index=False)
y.to_csv("../data/y_data.csv",index=False)