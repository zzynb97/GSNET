import torch
import torch.nn as nn

class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.cnnlayer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.001),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.001),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.001),
            nn.MaxPool2d(2)
        )
        self.fclayer = nn.Sequential(
            nn.Linear(128*4*4, 128),
            nn.LeakyReLU(negative_slope=0.001),
            nn.Linear(128, 16),
            nn.LeakyReLU(negative_slope=0.001),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.cnnlayer(x)
        x = x.view((x.shape[0], -1))
        x = self.fclayer(x)
        return x

net = CNNnet()
# x = torch.rand((10000, 1, 28, 28))
# for name, layer in net.named_children():
#     x = layer(x)
#     print(name, 'output shape:\t', x.shape)