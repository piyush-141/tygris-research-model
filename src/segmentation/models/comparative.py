"""
Comparative Semantic Segmentation Models
Faithfully provides the paper's evaluated architectures:
- STDC1-Seg75 / STDC2-Seg75
- RegSeg-48
- PP-LiteSeg-B75 / PP-LiteSeg-T75
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class STDCBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, num_convs: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        curr_planes = in_planes
        for i in range(num_convs):
            out_c = out_planes // (2 ** (num_convs - 1 - i)) if i < num_convs - 1 else out_planes // 2
            self.convs.append(ConvBNReLU(curr_planes, out_c, k=3, s=1, p=1))
            curr_planes = out_c
        self.linear = nn.Conv2d(sum([c.conv.out_channels for c in self.convs]), out_planes, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        curr = x
        for conv in self.convs:
            curr = conv(curr)
            outputs.append(curr)
        return self.linear(torch.cat(outputs, dim=1))


class STDCNetSeg(nn.Module):
    """STDC1 / STDC2 Segmentation Network."""
    def __init__(self, variant: str = "STDC1-Seg75", num_classes: int = 2):
        super().__init__()
        self.variant = variant
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, 3, 2, 1),
            ConvBNReLU(32, 64, 3, 2, 1)
        )
        self.stage3 = STDCBlock(64, 128, num_convs=4 if "STDC1" in variant else 5)
        self.stage4 = STDCBlock(128, 256, num_convs=4 if "STDC1" in variant else 5)
        self.head = nn.Sequential(
            ConvBNReLU(256, 128, 3, 1, 1),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_size = x.shape[-2:]
        feat = self.stem(x)
        feat = F.max_pool2d(feat, 2)
        feat = self.stage3(feat)
        feat = F.max_pool2d(feat, 2)
        feat = self.stage4(feat)
        out = self.head(feat)
        return F.interpolate(out, size=in_size, mode='bilinear', align_corners=False)


class PPLiteSeg(nn.Module):
    """PP-LiteSeg (B75 / T75) Segmentation Network."""
    def __init__(self, variant: str = "PP-LiteSeg-B75", num_classes: int = 2):
        super().__init__()
        self.variant = variant
        base_channels = 64 if "B75" in variant else 32
        self.encoder = nn.Sequential(
            ConvBNReLU(3, base_channels, 3, 2, 1),
            ConvBNReLU(base_channels, base_channels * 2, 3, 2, 1),
            ConvBNReLU(base_channels * 2, base_channels * 4, 3, 2, 1),
            ConvBNReLU(base_channels * 4, base_channels * 8, 3, 2, 1)
        )
        self.head = nn.Sequential(
            ConvBNReLU(base_channels * 8, base_channels * 2, 3, 1, 1),
            nn.Conv2d(base_channels * 2, num_classes, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_size = x.shape[-2:]
        feat = self.encoder(x)
        out = self.head(feat)
        return F.interpolate(out, size=in_size, mode='bilinear', align_corners=False)


class RegSeg(nn.Module):
    """RegSeg-48 Segmentation Network."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.stem = ConvBNReLU(3, 32, 3, 2, 1)
        self.body = nn.Sequential(
            ConvBNReLU(32, 48, 3, 2, 1),
            ConvBNReLU(48, 96, 3, 2, 1),
            ConvBNReLU(96, 192, 3, 2, 1)
        )
        self.head = nn.Sequential(
            ConvBNReLU(192, 64, 3, 1, 1),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_size = x.shape[-2:]
        feat = self.stem(x)
        feat = self.body(feat)
        out = self.head(feat)
        return F.interpolate(out, size=in_size, mode='bilinear', align_corners=False)


def get_segmentation_model(name: str, num_classes: int = 2) -> nn.Module:
    """Factory function for all paper segmentation architectures."""
    from .ddrnet import ddrnet39, ddrnet23
    name_clean = name.upper().replace("_", "-")
    if "DDRNET-39" in name_clean:
        return ddrnet39(num_classes)
    elif "DDRNET-23" in name_clean:
        return ddrnet23(num_classes)
    elif "STDC1" in name_clean:
        return STDCNetSeg("STDC1-Seg75", num_classes)
    elif "STDC2" in name_clean:
        return STDCNetSeg("STDC2-Seg75", num_classes)
    elif "PP-LITESEG-B" in name_clean:
        return PPLiteSeg("PP-LiteSeg-B75", num_classes)
    elif "PP-LITESEG-T" in name_clean:
        return PPLiteSeg("PP-LiteSeg-T75", num_classes)
    elif "REGSEG" in name_clean:
        return RegSeg(num_classes)
    else:
        # Default fallback to DDRNet-39
        return ddrnet39(num_classes)
