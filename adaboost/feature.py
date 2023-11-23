#%%
import numpy as np
import pandas as pd

#%%
df=pd.read_excel('out/preprocessed/density.xlsx')
df=df.fillna(-1)
data=df.values
citys=[i for i in range(1,41)]
city_data=[]

#%%
for city in citys:
    row_beg=22*city-22
    row_end=22*city
    data_part=data[row_beg:row_end,:]
    row_del=[]
    for i in range(data_part.shape[0]):
        if -1.0 in data_part[i,:]:
            row_del.append(i)
    data_part=np.delete(data_part,row_del,axis=0)

    city_data.append(data_part)

#%%
Growth=[]
for city in city_data:
    growth=[]
    for i in range(1,city.shape[0]):
        g=(city[i,2]-city[i-1,2])/city[i-1,2]
        growth.append(g)

    growth_arr=np.array(growth)
    Growth.append(growth_arr)

#%%
for i in range(len(Growth)):
    Growth[i]=Growth[i].T

con_array=np.concatenate(Growth,axis=0)

#%%
con_array=np.transpose(con_array)

#%%
df=pd.DataFrame(con_array)
df.to_excel()