#%%
import numpy as np
import pandas as pd

#%%
df=pd.read_excel('out/preprocessed/file_withnan.xlsx')
df=df.fillna(-1)
data=df.values

#%%
row_del=[]
for i in range(data.shape[0]):
    if -1.0 in data[i,:]:
        row_del.append(i)

data_update=np.delete(data,row_del,axis=0)

#%%
citys=[i for i in range(1,41)]
city_dict={}
for city in citys:
    condition= data_update[:,0]==float(city)
    data_part=data_update[condition]
    city_dict[city]=data_part

#%%
city_data=[]
for city in city_dict.values():
    if len(city)!=0:
        all_growth=[]
        for i in range(2,11):#column
            growth_list=[]
            for j in range(1,city.shape[0]-1):#row
                growth=(city[j,i]-city[j-1,i])/city[j-1,i]
                growth_list.append(growth)
            all_growth.append(growth_list)

        city_data.append(all_growth)

#%%
cityarray_list=[]
for city in city_data:
    city_array=np.array(city).T
    cityarray_list.append(city_array)

con_array=np.concatenate(cityarray_list,axis=0)

#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X=con_array[:,:7]
y=con_array[:,7:]
X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_valid)
mse=mean_squared_error(y_valid,y_pred)
print(f'Mean Squared Error: {mse}')

#%%
import joblib
joblib.dump(model,'randomforest_for_growth.joblib')

#%%
df=pd.DataFrame(con_array)
df.to_excel('out/preprocessed/growth_file.xlsx',index=False)