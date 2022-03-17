from torch import nn
from torch.nn import functional as F
from torchvision.models.densenet import densenet169


class DenseNet169(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = densenet169(pretrained=False, num_classes=10)
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        print(self.model)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)
