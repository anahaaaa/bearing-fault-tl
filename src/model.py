# ==================================================
# 1D ResNet for Bearing Fault Diagnosis
# Input : (batch, 1, 1024)
# Output: (batch, 10)
# ==================================================

import torch
import torch.nn as nn


# --------------------------------------------------
# Residual Block
# --------------------------------------------------
class ResidualBlock1D(nn.Module):
    """
    Basic 1D residual block:
    Conv -> BN -> ReLU -> Conv -> BN + shortcut
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# --------------------------------------------------
# ResNet1D
# --------------------------------------------------
class ResNet1D(nn.Module):
    """
    ResNet18-style 1D network
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.in_channels = 64

        # Initial stem
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool1d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Residual stages
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.fc = nn.Linear(512, num_classes)

    # ----------------------------------------------
    # Build layers
    # ----------------------------------------------
    def _make_layer(self, out_channels, blocks, stride):

        layers = []

        layers.append(
            ResidualBlock1D(
                self.in_channels,
                out_channels,
                stride
            )
        )

        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                ResidualBlock1D(
                    self.in_channels,
                    out_channels,
                    stride=1
                )
            )

        return nn.Sequential(*layers)

    # ----------------------------------------------
    # Forward
    # ----------------------------------------------
    def forward(self, x):

        # Input: (B,1,1024)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)     # (B,512,1)

        x = x.squeeze(-1)           # (B,512)

        x = self.fc(x)              # (B,10)

        return x


# --------------------------------------------------
# Factory Function
# --------------------------------------------------
def build_model(num_classes=10):
    """
    Returns ResNet1D model
    """

    model = ResNet1D(num_classes=num_classes)

    return model
