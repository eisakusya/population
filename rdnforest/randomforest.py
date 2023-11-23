#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

#%%
df=pd.read_excel('preprocessed/file_handwork.xlsx')
df=df.fillna(-1)
data=df.values

#%%
pop_per=[]
pop_reg=[]
citys = [i for i in range(1, 41)]
for city in citys:
    part=data[city * 22 - 22:city * 22,:]

    row_del=[]
    for i in range(part.shape[0]):
        if -1.0 in part[i,:]:
            row_del.append(i)

    update_part=np.delete(part,row_del,axis=0)
    x=update_part[:,1]
    y_1=update_part[:,10]
    y_2=update_part[:,11]

    x_train,x_valid=train_test_split(x,test_size=0.2,random_state=42)
    y1_train,y1_valid,y2_train,y2_valid=train_test_split(y_1,y_2,test_size=0.2,random_state=42)

    model_per=RandomForestRegressor(n_estimators=100,random_state=42)
    model_reg=RandomForestRegressor(n_estimators=100,random_state=43)

    model_per.fit(x_train.reshape(-1,1),y1_train)

    model_reg.fit(x_train.reshape(-1,1),y2_train)

    per_pred=model_per.predict(x_valid.reshape(-1,1))
    reg_pred=model_reg.predict(x_valid.reshape(-1,1))

    mse1 = mean_squared_error(y1_valid, per_pred)
    print(f'Permanent Mean Squared Error: {mse1}')

    mse2 = mean_squared_error(y2_valid, reg_pred)
    print(f'Registered Mean Squared Error: {mse2}')

    permanent=model_per.predict(np.array([22.0]).reshape(-1,1))
    register=model_reg.predict(np.array([22.0]).reshape(-1,1))

    pop_per.append(permanent[0])
    pop_reg.append(register[0])

    input("Press Enter to continue...")

#%%
output=pd.DataFrame({
    'permanent':pop_per,
    'registered':pop_reg
})
output.to_excel('result.xlsx',index=False)