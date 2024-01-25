import torch
import torch.nn as nn
import torch.nn.functional as F


class rnn(nn.Module):
    def __init__(self):
        super(rnn, self).__init__()

        self.lstm = nn.LSTM(input_size=3 * 64 * 64, hidden_size=1024, num_layers=1, batch_first=True)
        self.BN = nn.BatchNorm1d(1024)

        self.fc = nn.Linear(1024, 10)
        self.out = nn.Linear(10, 2)

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        x, (h_n, h_c) = self.lstm(x)
        x = torch.squeeze(x, dim=1)
        x = self.BN(x)
        x = F.relu(self.fc(x))
        x = self.out(x)
        # x = F.softmax(x, dim=1)
        return x
