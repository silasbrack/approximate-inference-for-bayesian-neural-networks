from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet18


def ResNet18(num_classes: int = 10):
    model = resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, num_classes)
    return model
# class ResNet18(nn.Module):
#     def __init__(self, num_classes: int = 10):
#         super().__init__()
#         self.model = resnet18(pretrained=False, num_classes=num_classes)
#         self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         self.model.fc = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = self.model(x)
#         return F.log_softmax(x, dim=1)
