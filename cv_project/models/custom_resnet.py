from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            num_channels,
            kernel_size=3,
            padding=1,
            stride=strides,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1, bias=False
        )
        # 通道数或空间尺寸变化时，捷径分支也需要投影。
        if use_1x1conv or input_channels != num_channels or strides != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    num_channels,
                    kernel_size=1,
                    stride=strides,
                    bias=False,
                ),
                nn.BatchNorm2d(num_channels),
            )
        else:
            self.downsample = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.downsample is not None:
            X = self.downsample(X)
        Y += X
        return F.relu(Y)


class CustomResNet(nn.Module):
    """使用命名层组织的自定义 ResNet18，支持 fc 和 layer4 微调。"""

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.apply(init_weights)

    @staticmethod
    def _make_layer(input_channels, num_channels, stride):
        return nn.Sequential(
            Residual(input_channels, num_channels, strides=stride),
            Residual(num_channels, num_channels),
        )

    def forward(self, X):
        X = self.maxpool(self.relu(self.bn1(self.conv1(X))))
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.avgpool(X)
        return self.fc(X.flatten(1))


def model_factory(num_classes: int) -> CustomResNet:
    return CustomResNet(num_classes)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
