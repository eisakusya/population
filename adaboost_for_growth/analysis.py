#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df=pd.read_excel('out/preprocessed/growth_file.xlsx')
data=df.fillna(-1).values
citys=[i for i in range(1,41)]

#%%
for city in citys:
    row_beg=22*city-22
    row_end=22*city
    part=data[row_beg:row_end,:]
    df_part=pd.DataFrame(part)
    cor_ma=df_part.corr()

    label=df.columns.values
    plt.clf()
    plt.figure(figsize=(10,8))
    sns.heatmap(cor_ma, annot=True, cmap='viridis', fmt=".2f", xticklabels=label,yticklabels=label)
    plt.title('Heatmap')
    plt.show()
    plt.savefig('out/heatmap/city'+str(city)+'.jpeg')

    input('Enter')
#%%
label=['urban','salary','unemployment','employment','2_employment','3_employment','density','registered','permanent_population']
cor_ma=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(cor_ma,annot=True, cmap='viridis', fmt=".2f", xticklabels=label,yticklabels=label)
plt.title('Heatmap')
plt.show()

