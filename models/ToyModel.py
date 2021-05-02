import torch.nn as nn
import torch.nn.functional as F


class ConvBrunch(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(ConvBrunch, self).__init__()
        padding = (kernel_size - 1) // 2
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU())

    def forward(self, x):
        return self.out_conv(x)


class ToyModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ToyModel, self).__init__()
        self.block1 = nn.Sequential(
            ConvBrunch(3, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBrunch(64, 128, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBrunch(128, 256, 3),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.global_avg_pool(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x
