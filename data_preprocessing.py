# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append('./')

# %%
df = pd.read_excel('disposableIcome.xlsx')
# %%
citys = [num for num in range(1, 41)]
years = [num for num in range(21)]

# %%

df = pd.read_excel('urbanizationRate.xlsx')

# %%
data = df.values
for i in range(data.shape[0]):
    oristr = data[i, 0]
    data[i, 0] = oristr[4:]

    data[i, 1] = data[i, 1] - 2001

# %%
# 城市化率数据处理
urbanization_rate = [0] * 880  # 40city*22year
for i in range(data.shape[0]):
    city_num = int(data[i, 0])
    year_num = data[i, 1]
    urban_rate = data[i, 2]
    urbanization_rate[(city_num - 1) * 22 + (year_num)] = urban_rate

# %%
df = pd.read_excel('salary.xlsx')
df = df.fillna(0)
data = df.values

# %%
# 工资水平数据处理
salary_list = [0] * 880
for i in range(1, data.shape[1]):
    for j in range(data.shape[0]):
        city_num = i
        year_num = j
        salary = data[j, i]

        salary_list[(city_num - 1) * 22 + year_num] = salary

# %%
Data = pd.DataFrame({
    'urbanization_rate': urbanization_rate,
    'salary': salary_list
})
Data.to_excel('preprocessed/file.xlsx')

# %%
# 读取人口规模数据
df = pd.read_excel('populationScale.xlsx')
df = df.fillna(0)
data = df.values

# %%
# 预处理
for i in range(data.shape[0]):
    oristr = data[i, 0]
    data[i, 0] = oristr[4:]

    data[i, 1] = data[i, 1] - 2001

# %%
# 读入
permanent_re = [0] * 880
register_re = [0] * 880
for i in range(data.shape[0]):
    year_num = data[i, 1]
    city_num = int(data[i, 0])
    val1 = data[i, 2]
    val2 = data[i, 3]

    permanent_re[(city_num - 1) * 22 + year_num] = val1
    register_re[(city_num - 1) * 22 + year_num] = val2

# %%
# 读入人口密度数据
df = pd.read_excel('populationDensity.xlsx')
df = df.fillna(0)
data = df.values

# %%
# 预处理
row_del = []
for i in range(data.shape[0]):
    if data[i, 0] == 2000:
        row_del.append(i)

data = np.delete(data, row_del, axis=0)

# %%
for i in range(data.shape[0]):
    oristr = data[i, 1]
    data[i, 1] = oristr[4:]

    data[i, 0] = data[i, 0] - 2001

# %%
pop_density = [0] * 880
for i in range(data.shape[0]):
    year_num = data[i, 0]
    city_num = int(data[i, 1])
    val = data[i, 2]

    pop_density[(city_num - 1) * 22 + year_num] = val

# %%
df = pd.read_excel('disposableIcome.xlsx')
df = df.fillna(0)
data = df.values

# %%
disposable_in = [0] * 880
for i in range(data.shape[0]):
    oristr = data[i, 0]

    for j in range(1, data.shape[1]):
        city_num = int(oristr[4:])
        year_num = j + 13
        val = data[i, j]

        disposable_in[(city_num - 1) * 22 + year_num] = val

# %%
df = pd.read_excel('disposableIcome.xlsx', sheet_name='consumption_expenditure')
df = df.fillna(0)
data = df.values
data = np.delete(data, 0, axis=0)

# %%
consumption_ex = [0] * 880
for i in range(data.shape[0]):
    oristr = data[i, 0]

    for j in range(1, data.shape[1]):
        city_num = int(oristr[4:])
        year_num = j + 13
        val = data[i, j]

        consumption_ex[(city_num - 1) * 22 + year_num] = val

# %%
df = pd.read_excel('disposableIcome.xlsx', sheet_name='towner_ ConsumptionExpenditures')
df = df.fillna(0)
data = df.values
data = np.delete(data, 0, axis=0)

# %%
towner_consumption = [0] * 880
for i in range(data.shape[0]):
    oristr = data[i, 0]

    for j in range(1, data.shape[1]):
        city_num = int(oristr[4:])
        year_num = j + 9
        val = data[i, j]

        towner_consumption[(city_num - 1) * 22 + year_num] = val

# %%
df = pd.read_excel('disposableIcome.xlsx', sheet_name='rural_ConsumptionExpenditures')
df = df.fillna(0)
data = df.values
data = np.delete(data, 0, axis=0)

rural_consumption = [0] * 880
for i in range(data.shape[0]):
    oristr = data[i, 0]

    for j in range(1, data.shape[1]):
        city_num = int(oristr[4:])
        year_num = j + 9
        val = data[i, j]

        rural_consumption[(city_num - 1) * 22 + year_num] = val

# %%
df = pd.read_excel('disposableIcome.xlsx', sheet_name='towner_disposableIncome')
df = df.fillna(0)
data = df.values
data = np.delete(data, 0, axis=0)

towner_income = [0] * 880
for i in range(data.shape[0]):
    oristr = data[i, 0]

    for j in range(1, data.shape[1]):
        city_num = int(oristr[4:])
        year_num = j + 9
        val = data[i, j]

        towner_income[(city_num - 1) * 22 + year_num] = val

# %%
df = pd.read_excel('disposableIcome.xlsx', sheet_name='rural_disposableIncome')
df = df.fillna(0)
data = df.values
data = np.delete(data, 0, axis=0)

rural_income = [0] * 880
for i in range(data.shape[0]):
    oristr = data[i, 0]

    for j in range(1, data.shape[1]):
        city_num = int(oristr[4:])
        year_num = j + 9
        val = data[i, j]

        rural_income[(city_num - 1) * 22 + year_num] = val

#%%
Data = pd.DataFrame({
    'urbanization_rate': urbanization_rate,
    'salary': salary_list,
    'disposable_income':disposable_in,
    'consumption_expenditure':consumption_ex,
    'towner_income':towner_income,
    'rural_income':rural_income,
    'towner_expenditure':towner_consumption,
    'rural_expenditure':rural_consumption,
    'permanent_resident':permanent_re,
    'registered_resident':register_re,
    'population_density':pop_density
})
Data.to_excel('preprocessed/file.xlsx')