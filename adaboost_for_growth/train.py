#%%
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#%%
df=pd.read_excel('out/preprocessed/growth_file.xlsx')
data=df.fillna(-1).values
#%%
feature_col=8
X=data[:,:feature_col]
y=data[:,feature_col:]

#%%
x_train,x_valid,y_train,y_valid=train_test_split(X,y,test_size=0.3,random_state=42)

#%%
model=AdaBoostRegressor(DecisionTreeRegressor(max_depth=11),n_estimators=117,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_valid)
mse=mean_squared_error(y_valid,y_pred)
print(f'Mean Squared Error: {mse}')

#%%
import joblib
joblib.dump(model,'adaboost_for_growth/adaboost_for_growth.joblib')