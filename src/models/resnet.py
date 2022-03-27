from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock


# def ResNet18(num_classes: int = 10):
#     model = resnet18(pretrained=False, num_classes=num_classes)
#     model.conv1 = nn.Conv2d(
#         1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
#     )
#     model.fc = nn.Linear(512, num_classes)
#     return model

class ResNet18(ResNet):
    def __init__(
        self,
        n_classes=10,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3), bias=False)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self._forward_impl(x)
