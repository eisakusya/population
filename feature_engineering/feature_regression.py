# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# %%
all = set([i for i in range(1, 41)])
remain = all - {2, 5, 8, 12, 13, 15, 18, 22, 24, 26, 28, 30, 32, 33, 34, 36, 37, 39, 40}
remain

# %%
df = pd.read_excel('preprocessed/file.xlsx')
df = df.fillna(-1)
data = df.values

# %%
row_del = []
for i in range(data.shape[0]):
    if int(data[i, 0]) not in remain:
        row_del.append(i)

data_updated = np.delete(data, row_del, axis=0)

# %%
row_del.clear()
for i in range(data_updated.shape[0]):
    if -1.0 in data_updated[i, :]:
        row_del.append(i)

data_updated = np.delete(data_updated, row_del, axis=0)

# %%
col_del = [0, 1]
data_updated = np.delete(data_updated, col_del, axis=1)

x = data_updated[:, 0:2]
y = data_updated[:, 2:8]

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_valid)
mse = mean_squared_error(y_valid, y_pred)
print(f'Mean Squared Error: {mse}')

# %%
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

model_svm = SVR(kernel='linear')
multioutput_model = MultiOutputRegressor(model_svm)
multioutput_model.fit(x_train, y_train)
y_svmpred = multioutput_model.predict(x_valid)
mse = mean_squared_error(y_valid, y_svmpred)
print(f'Mean Squared Error: {mse}')
# 经对比发现，线性模型足以
# %%
import joblib

joblib.dump(model, 'linear_regression.joblib')
