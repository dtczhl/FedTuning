"""
    Defined in Table 4 of the paper ADAPTIVE FEDERATED OPTIMIZATION
    <https://arxiv.org/pdf/2003.00295.pdf>
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class EmnistCR(nn.Module):
    def __init__(self, num_input_feature, depth, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_input_feature, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(p=0.25)
        self.flatten1 = nn.Flatten()
        self.dense1 = nn.Linear(in_features=9216, out_features=128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.maxpool1(out)
        # out = self.dropout1(out)
        out = self.flatten1(out)
        out = self.dense1(out)
        # out = self.dropout2(out)
        out = self.dense2(out)
        out = F.softmax(out, dim=1)
        return out


if __name__ == '__main__':

    model = EmnistCR(1, -1, 62)
    summary(model, (1, 28, 28))
