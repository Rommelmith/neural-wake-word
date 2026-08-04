import torch
import torch.nn as nn


class WakeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 5), stride=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(5, 5), stride=1)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(5, 5), stride=1)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(540, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x