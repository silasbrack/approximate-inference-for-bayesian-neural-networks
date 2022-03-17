import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvNet(n_classes: int = 10):
    return nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_classes),
            nn.LogSoftmax(-1),
        )

# class ConvNet(nn.Module):
#     def __init__(self, n_classes: int = 10):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(1, 6, 5),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             nn.Conv2d(6, 16, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(16 * 4 * 4, 120),
#             nn.ReLU(),
#             nn.Linear(120, 84),
#             nn.ReLU(),
#             nn.Linear(84, n_classes),
#         )

#     def forward(self, x):
#         x = self.model(x)
#         return F.log_softmax(x, dim=1)
    # def __init__(self, n_classes: int = 10):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(1, 6, 5)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 4 * 4, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, n_classes)

    # def forward(self, x):
    #     x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    #     x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    #     x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x
