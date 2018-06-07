import torch.nn as nn
import torch.nn.functional as F


class classifier(nn.Module):
    def __init__(self, isDropOut = True):
        super(classifier, self).__init__()
        self.conv1 = nn.Conv2d(5,32,3,padding=2,stride=2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(5,64,1,padding=2,stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        ## WRITE YOUR CODE HERE ##
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print("X is " + x);
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

