import torch.nn as nn
import torch.nn.functional as F


class classifier(nn.Module):
    def __init__(self, isDropOut = True):
        super(classifier, self).__init__()
        self.conv1 = nn.Conv2d(1,32,5,padding=2,stride=1)
        self.pool = nn.MaxPool2d(2,stride=2)
        self.conv2 = nn.Conv2d(32,64,5,padding=2,stride=1)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024,10)
        self.relu = nn.ReLU(inplace=True);

    def forward(self, x):
       
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(-1, 7*7*64)

        x = self.relu(self.fc1(x))
        x = (self.fc2(x))
       
        return x

