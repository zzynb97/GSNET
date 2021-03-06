import torch
from torch import nn
import torch.nn.functional as F


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Inception(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.p1_1 = nn.Sequential(nn.Conv2d(in_c, c1, kernel_size=1),
                                  nn.BatchNorm2d(c1))

        self.p2_1 = nn.Sequential(nn.Conv2d(in_c, c2[0], kernel_size=1),
                                  nn.BatchNorm2d(c2[0]))
        self.p2_2 = nn.Sequential(nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
                                  nn.BatchNorm2d(c2[1]))

        self.p3_1 = nn.Sequential(nn.Conv2d(in_c, c3[0], kernel_size=1),
                                  nn.BatchNorm2d(c3[0]))
        self.p3_2 = nn.Sequential(nn.Conv2d(c3[0], c3[1], kernel_size=3, padding=1),
                                  nn.BatchNorm2d(c3[1]))

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Sequential(nn.Conv2d(in_c, c4, kernel_size=1),
                                  nn.BatchNorm2d(c4))

    def forward(self, x):
        p1 = F.leaky_relu(self.p1_1(x))
        p2 = F.leaky_relu(self.p2_2(F.leaky_relu(self.p2_1(x))))
        p3 = F.leaky_relu(self.p3_2(F.leaky_relu(self.p3_1(x))))
        p4 = F.leaky_relu((self.p4_2(self.p4_1(x))))
        return torch.cat((p1, p2, p3, p4), dim=1)


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7,
                             stride=2, padding=3),
                   nn.LeakyReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2,
                                padding=1)
                   )

b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                   )

b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   GlobalAvgPool2d()
                   )

net = nn.Sequential(b1, b2, b3, b4, b5,
                    FlattenLayer(),
                    nn.Linear(1024, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 2)
                    )

# x = torch.rand(1, 1, 32, 32)
# for name, blk in net.named_children():
#     x = blk(x)
#     print(name, 'output shape:\t', x.shape)