import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from lstm.lstm_model import LSTMModel

input_size = 1
output_size = 8
data_size=22
citys = [i for i in range(1, 41)]

df=pd.read_excel('preprocessed/file_handwork.xlsx')

for city in citys:
    df_part = df.iloc[city * 22 - 22:city * 22, :10]
    data=df_part.values

    feature=data[:,1]
    label=data[:,2:]

    feature=torch.from_numpy(feature).float()
    label=torch.from_numpy(label).float()

    train_size=int(0.8*data_size)
    x_train,x_valid=feature[:train_size],feature[train_size:]
    y_train,y_valid=label[:train_size],label[:train_size]

    hidden_size=8
    model=LSTMModel(input_size,hidden_size,output_size)
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)

    epoch_num=15
    for epoch in range(epoch_num):
        model.train()
        optimizer.zero_grad()
        x_train=torch.unsqueeze(x_train,0)
        # x_train = torch.unsqueeze(x_train, -1)
        # x_train
        output=model(x_train)
        loss=criterion(output,y_train)
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch}/{epoch_num}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        x_valid=torch.unsqueeze(x_valid,0)
        #x_valid=torch.unsqueeze(x_valid,-1)
        output_valid=model(x_valid)
        loss_valid=criterion(output_valid,y_valid)
        print(f'Test Loss: {loss_valid.item():.4f}')

    new_data=np.array([2023-2001])
    new_tensor=torch.from_numpy(new_data)
    new_tensor=torch.unsqueeze(new_tensor,0)
    #new_tensor=torch.unsqueeze(new_tensor,-1)
    with torch.no_grad():
        predict=model(new_tensor)

    predict
    input("Press Enter to continue...")