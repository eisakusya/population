# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# %%
df = pd.read_excel('out/preprocessed/file_withnan.xlsx')
data = df.fillna(-1).values

# %%
citys = [i for i in range(1, 41)]
city = 1
row_begin = 22 * city - 22
row_end = 22 * city
partition = data[row_begin:row_end, :]
row_del=[]
for i in range(partition.shape[0]):
    if -1.0 in partition[i,:]:
        row_del.append(i)
#partition=np.delete(partition,row_del,axis=0)

