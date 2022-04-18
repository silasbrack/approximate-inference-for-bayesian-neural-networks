import torch.nn as nn


def ConvNet(num_classes: int = 10):
    return nn.Sequential(
        nn.Conv2d(1, 32, 4),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(11 * 11 * 32, 128),
        nn.ReLU(),
        # nn.Dropout(0.5),
        nn.Linear(128, num_classes),
        nn.LogSoftmax(-1),
    )
    # return nn.Sequential(
    #     nn.Conv2d(1, 6, 5),
    #     nn.ReLU(),
    #     nn.MaxPool2d((2, 2)),
    #     nn.Conv2d(6, 16, 5),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
    #     nn.Flatten(),
    #     nn.Linear(16 * 4 * 4, 120),
    #     nn.ReLU(),
    #     nn.Linear(120, 84),
    #     nn.ReLU(),
    #     nn.Linear(84, num_classes),
    #     nn.LogSoftmax(-1),
    # )
