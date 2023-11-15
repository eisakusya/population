#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
df=pd.read_excel('preprocessed/file.xlsx')

#%%
#切片取出第一座城市的数据
city1=df.iloc[:22,:]
city1=city1.fillna(0)

#%%
year=[i for i in range(6,22)]
val=[v for v in city1['urbanization_rate'].tolist() if v!=0]

#%%
from sklearn.linear_model import LinearRegression
y_2d=np.array(year).reshape(-1,1)
v_2d=np.array(val).reshape(-1,1)
model=LinearRegression()
model.fit(y_2d,v_2d)

val_pre=model.predict(np.array([i for i in range(6)]).reshape(-1,1))

#%%
city1.loc[[i for i in range(6)],'urbanization_rate']=val_pre
