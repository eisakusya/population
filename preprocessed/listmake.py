#%%
citys=[i for i in range(1,41)]
city_num=[]
for city in citys:
    city_num=city_num+([city]*22)

#%%
import pandas as pd
df=pd.DataFrame({
    'city':city_num
})
df.to_excel('city.xlsx')