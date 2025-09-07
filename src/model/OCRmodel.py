import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cv1 = nn.Conv2d(3, 64, 3)
        self.cv2 = nn.Conv2d(64, 128, 5)
        self.cv3 = nn.Conv2d(128, 256, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(2304, 128)
        self.out = nn.Linear(128, 61)

    def forward(self, x):
        x = self.pool(F.relu(self.cv1(x)))
        x = self.pool(F.relu(self.cv2(x)))
        x = self.pool(F.relu(self.cv3(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.linear1(x))
        x = self.out(x)
        return x
