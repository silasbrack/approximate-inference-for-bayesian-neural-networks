import torch
from torch import nn
from torch.nn import functional as F


def DenseNet(num_classes: int = 10):
    dims = (1, 28, 28)
    hidden_size = 64
    channels, width, height = dims
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(channels * width * height, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, num_classes),
        nn.LogSoftmax(dim=-1),
    )
