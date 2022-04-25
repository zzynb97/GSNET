import torch
from torch import nn
import torch.nn.functional as F


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=stride)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(out + x)

# blk = Residual(3, 6, use_1x1conv=True, stride=2)
# x = torch.rand((4, 3, 6, 6))
# print(blk(x).shape)

net = nn.Sequential(
    # nn.Conv2d(3, 1, kernel_size=1),##flag
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

def resnet_block(in_channels, out_channels, num_residuals,
                 first_block=False):
    if first_block:
        assert in_channels == out_channels

    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels,
                                use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module('resnet_block2', resnet_block(64, 128, 2))
net.add_module('resnet_block3', resnet_block(128, 256, 2))
# net.add_module('resnet_block4', resnet_block(256, 512, 2))
net.add_module('fc', nn.Sequential(
    nn.AvgPool2d(2),
    FlattenLayer(),
    nn.Linear(256, 10)
))

# x = torch.rand((1, 1, 28, 28))
# for name, layer in net.named_children():
#     x = layer(x)
#     print(name, 'output shape:\t', x.shape)
# class AAnet(nn.Module):
#     def __init__(self):
#         super(AAnet, self).__init__()
#
#     def forward(self, x):
#         return net(x)

