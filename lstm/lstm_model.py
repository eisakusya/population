import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out=out[:, -1]
        # out=torch.unsqueeze(out,0)
        # out=torch.unsqueeze(out,-1)
        # out
        # out = self.fc(out)  # 取 LSTM 输出的最后一个时间步
        return out