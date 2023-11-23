#%%
import joblib
import numpy as np
import pandas as pd

#%%
df=pd.read_excel('out/preprocessed/output1123.xlsx')

#%%
data=df.values
dataset=[]
for i in range(data.shape[0]):
    sample = []
    for j in range(1,data.shape[1],2):
        feature_i=data[i,j-1]
        feature_j=data[i,j]
        rate = (feature_j - feature_i) / feature_i
        if feature_i==0 and feature_j==0:
            rate=0
        sample.append(rate)
    dataset.append(sample)

#%%
dataset=np.array(dataset)

#%%
model=joblib.load('adaboost_for_growth/adaboost_for_growth.joblib')
y_pred=model.predict(dataset)

#%%
df=pd.read_excel('out/preprocessed/file_withnan.xlsx')
data=df.fillna(-1).values

#%%
pop_pred=[]
for i in range(1,41):
    row=22*i-2
    per_pop=data[row,10]
    pred_p=per_pop*((1+y_pred[i-1])**2)
    pop_pred.append(pred_p)

result=np.array(pop_pred)
#%%
df=pd.DataFrame(result)
df.to_excel('out/result_adaboost_growth.xlsx',index=False)