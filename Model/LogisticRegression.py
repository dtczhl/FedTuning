"""
    Logistic regression
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class LogisticRegression(nn.Module):
    def __init__(self, num_input_feature, depth, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 200),
            nn.ReLU(),
            nn.Linear(200, num_classes)
        )

        # num_input_feature = 28 * 28
        #
        # self.flatten = nn.Flatten()
        # self.linear = nn.Linear(num_input_feature, num_classes)

    def forward(self, x):
        # out = self.flatten(x)
        # out = self.linear(out)
        return self.layers(x)


if __name__ == '__main__':

    model = LogisticRegression(1, -1, 62)
    summary(model, (1, 28, 28))
