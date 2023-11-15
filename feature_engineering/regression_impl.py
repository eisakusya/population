# %%
import pandas as pd
import numpy as np
import joblib
import sys

sys.path.append('./')
from sklearn.linear_model import LinearRegression

# %%
citys = [i for i in range(1, 41)]
df = pd.read_excel('../preprocessed/file.xlsx')

for city in citys:
    df_part = df.iloc[city * 22 - 22:city * 22, :]
    df_part = df_part.interpolate()

    # 城镇化率回归填充
    data = df_part.fillna(-1).values

    year = data[:, 1]
    urban = data[:, 2]
    empty_data = []
    for i in range(len(urban)):
        if int(urban[i]) == -1:
            empty_data.append(i)
    if len(empty_data) != 0:
        year = np.delete(year, empty_data)
        urban = np.delete(urban, empty_data)
        model_urban = LinearRegression()
        model_urban.fit(year.reshape(-1, 1), urban.reshape(-1, 1))

        x_pre = np.array(empty_data).reshape(-1, 1)
        y_pre = model_urban.predict(x_pre)

        df_part.loc[[(city - 1) * 22 + i for i in empty_data], 'urbanization_rate'] = y_pre

    # 生活质量回归
    data = df_part.fillna(-1).values
    X = data[:, 2:4]

    load_model = joblib.load('../linear_regression.joblib')
    y_pre = load_model.predict(X)

    df_part.iloc[:, 4:10] = y_pre

    df_part.to_excel('../feature_engineering/temp.xlsx', index=False)
    input("Press Enter to continue...")
