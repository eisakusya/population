#%%
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#%%
df=pd.read_excel('preprocessed/growth_file.xlsx')
data=df.values

#%%
feature_number=7
X=data[:,:feature_number]
Y=data[:,feature_number:]

x_train,x_valid,y_train,y_valid=train_test_split(X,Y,test_size=0.3,random_state=42)

#%%
model1=AdaBoostRegressor(DecisionTreeRegressor(max_depth=8),n_estimators=117,random_state=42)
model2=AdaBoostRegressor(DecisionTreeRegressor(max_depth=6),n_estimators=275,random_state=42)

y1_train,y1_valid=y_train[:,0],y_valid[:,0]
y2_train,y2_valid=y_train[:,1],y_valid[:,1]

model1.fit(x_train,y1_train)
model2.fit(x_train,y2_train)

y1_pred=model1.predict(x_valid)
y2_pred=model2.predict(x_valid)

mse1=mean_squared_error(y1_valid,y1_pred)
mse2=mean_squared_error(y2_valid,y2_pred)

print(f'Permanent Mean Squared Error 1: {mse1}')
print(f'Permanent Mean Squared Error 2: {mse2}')

#%%
import joblib
joblib.dump(model1,'adaboost/model_for_permanent.joblib')
joblib.dump(model2,'adaboost/model_for_register.joblib')