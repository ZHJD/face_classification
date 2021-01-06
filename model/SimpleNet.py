import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_process import TrainSet


class SimpleNet(nn.Module):

    def __init__(self, in_channels, num_classes=68):
        super().__init__()

        self.conv1_1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))

        x = self.classifier(x)

        return x

if __name__ == "__main__":
    net = SimpleNet(1)
    test = torch.ones((1, 1, 32, 32))
    y_hat = net(test)
    print(y_hat.shape)
    print(net)