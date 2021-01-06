import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_process import TrainSet


class Linear(nn.Module):

    def __init__(self, in_channels, num_classes=68):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32, num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    net = Linear(1)
    test = torch.ones((1, 1, 32, 32))
    y_hat = net(test)
    print(y_hat.shape)
    print(net)