#%%
import joblib
import numpy as np
import pandas as pd

#%%
df=pd.read_excel('preprocessed/output1120.xlsx')

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
model1=joblib.load('adaboost/model_for_permanent.joblib')
model2=joblib.load('adaboost/model_for_register.joblib')

y1_pred=model1.predict(dataset)
y2_pred=model2.predict(dataset)

#%%
df=pd.read_excel('preprocessed/file_withnan.xlsx')
data=df.fillna(-1).values

#%%
pop_pred=[]
for i in range(1,41):
    row=22*i-2
    per_pop=data[row,9]
    reg_pop=data[row,10]
    spl_pred=[per_pop*((1+y1_pred[i-1])**2),reg_pop*((1+y1_pred[i-1])**2)]
    pop_pred.append(spl_pred)

result=np.array(pop_pred)
df=pd.DataFrame(result)
df.to_excel('out/result_adaboost.xlsx',index=False)