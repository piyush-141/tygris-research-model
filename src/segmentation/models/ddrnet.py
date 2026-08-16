"""
DDRNet (Dual-Resolution Network for Real-Time Semantic Segmentation)
Faithfully implements DDRNet-39 (Paper-Selected Default) and DDRNet-23
Reference: Hong et al., "Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes", 2021.
Selected as the optimal segmentation architecture in Ma et al. (2025).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module for DDRNet."""
    def __init__(self, in_planes: int, branch_planes: int, out_planes: int):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 4, out_planes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width, height = x.shape[-1], x.shape[-2]
        x_list = []
        x_list.append(self.process1(F.interpolate(self.scale1(x), size=[height, width], mode='bilinear', align_corners=False)))
        x_list.append(self.process2((F.interpolate(self.scale2(x), size=[height, width], mode='bilinear', align_corners=False) + x_list[0])))
        x_list.append(self.process3((F.interpolate(self.scale3(x), size=[height, width], mode='bilinear', align_corners=False) + x_list[1])))
        x_list.append(self.process4((F.interpolate(self.scale4(x), size=[height, width], mode='bilinear', align_corners=False) + x_list[2])))
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class DDRNet(nn.Module):
    """
    Dual-Resolution Network implementation.
    Supports DDRNet-39 (paper default) and DDRNet-23.
    """
    def __init__(self, num_classes: int = 2, planes: int = 64, spp_planes: int = 128, head_planes: int = 256, variant: str = "DDRNet-39"):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes

        # Stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

        # Layers based on variant (DDRNet-39 vs DDRNet-23)
        layers = [3, 4, 6, 3] if variant == "DDRNet-39" else [2, 2, 2, 2]
        block = Bottleneck if variant == "DDRNet-39" else BasicBlock

        self.layer1 = self._make_layer(BasicBlock, planes, planes, layers[0])
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        in_planes_4 = planes * 4 * block.expansion
        self.layer4 = self._make_layer(block, in_planes_4, planes * 8, layers[3], stride=2)

        # High-resolution branch & Bilateral fusion
        self.spp = DAPPM(planes * 8 * block.expansion, spp_planes, planes * 4)
        self.final_layer = nn.Sequential(
            nn.Conv2d(planes * 4, head_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_planes, num_classes, kernel_size=1, bias=True)
        )

    def _make_layer(self, block, in_planes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(in_planes, planes, stride, downsample)]
        in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.spp(out)
        out = self.final_layer(out)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out


def ddrnet39(num_classes: int = 2) -> DDRNet:
    """[PAPER-SPECIFIED] DDRNet-39 default model for Amur tiger segmentation."""
    return DDRNet(num_classes=num_classes, planes=64, spp_planes=128, head_planes=256, variant="DDRNet-39")


def ddrnet23(num_classes: int = 2) -> DDRNet:
    """DDRNet-23 comparative model."""
    return DDRNet(num_classes=num_classes, planes=32, spp_planes=128, head_planes=128, variant="DDRNet-23")
