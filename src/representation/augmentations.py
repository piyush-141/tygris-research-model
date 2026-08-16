"""
Tiger Re-Identification Augmentation Pipeline
Faithfully implements Section 11 of the paper:
- Random Crop
- Random Grayscale
- Random Color Jitter
- Random Lighting
- RandAugment
"""

import random
from typing import Dict, Any, Tuple
import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance


class RandomLighting:
    """[PAPER-SPECIFIED] Random lighting variation simulation for camera traps."""
    def __init__(self, factor_range: Tuple[float, float] = (0.7, 1.3)):
        self.factor_range = factor_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > 0.5:
            factor = random.uniform(*self.factor_range)
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(factor)
        return img


def get_paper_reid_transforms(
    input_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    config_params: Dict[str, Any] = None
) -> T.Compose:
    """
    [PAPER-SPECIFIED AUGMENTATIONS]
    Constructs the exact augmentation sequence specified in Section 11.
    """
    if config_params is None:
        config_params = {}

    crop_scale = config_params.get("crop_scale", (0.8, 1.0))
    jitter_b = config_params.get("jitter_brightness", 0.2)
    jitter_c = config_params.get("jitter_contrast", 0.2)
    jitter_s = config_params.get("jitter_saturation", 0.2)
    jitter_h = config_params.get("jitter_hue", 0.1)

    if is_training:
        transforms_list = [
            T.RandomResizedCrop(input_size, scale=crop_scale),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=jitter_b, contrast=jitter_c, saturation=jitter_s, hue=jitter_h),
            RandomLighting(factor_range=(0.8, 1.2)),
            T.RandomGrayscale(p=0.2),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        transforms_list = [
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

    return T.Compose(transforms_list)
