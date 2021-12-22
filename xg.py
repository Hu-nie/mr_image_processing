from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error, mean_squared_log_error
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
df = pd.read_csv("test.csv")


X = df["x"]
y = df["y"]

X = X.values
y = y.values

xgb_r = XGBRegressor(random_state=200)

xgb_r.fit(X,np.log(y))

y_pred_sgv = xgb_r.predict()
