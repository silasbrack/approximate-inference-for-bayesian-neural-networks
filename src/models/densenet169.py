from torch import nn
from torch.nn import functional as F
from torchvision.models.densenet import densenet169


def DenseNet169(num_classes: int = 10):
    model = densenet169(pretrained=False, num_classes=10)
    model.features.conv0 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    return model
