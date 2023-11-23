#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
df=pd.read_excel('preprocessed/file_handwork.xlsx')
df=df.fillna(-1)
data=df.values

#%%
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

    plt.scatter(x.tolist(),y_1.tolist(),color='b',label='permanent')
    plt.scatter(x.tolist(),y_2.tolist(),color='r',label='registered')
    plt.legend()
    plt.savefig('pic/population_city'+str(city)+'.jpeg')
    plt.show()
    input("Press Enter to continue...")
