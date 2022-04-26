import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision


def load_mnist():
    
    train_data = MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

    # test_data = MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    train_data.data = train_data.data

    return train_data, train_data.data




# def load_norb():
    