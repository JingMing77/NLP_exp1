import torch
import torch.nn as nn
import torch.nn.functional as F


class dnn(nn.Module):
    def __init__(self):
        super(dnn, self).__init__()

        self.fc1 = nn.Linear(3 * 64 * 64, 1024)  # input:3*64*64, output:1024

        self.fc2 = nn.Linear(1024, 512)  # input:1024, output:512

        self.fc3 = nn.Linear(512, 2)  # input:512, output:2

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        # x = F.softmax(x, dim=1)

        return x

