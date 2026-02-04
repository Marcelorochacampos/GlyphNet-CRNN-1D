import torchvision.transforms.functional as F
import random
from torchvision import transforms
from PIL import Image

class ResizeKeepAspectRatio:
    def __init__(self, height=32, max_width=256):
        self.height = height
        self.max_width = max_width

    def __call__(self, img):
        w, h = img.size
        new_w = int(self.height * w / h)

        if new_w > self.max_width:
            new_w = self.max_width

        img = img.resize((new_w, self.height), Image.BILINEAR)

        pad_width = self.max_width - new_w
        if pad_width > 0:
            img = F.pad(img, (0, 0, pad_width, 0), fill=255)

        return img


def data_augmentation_pipeline():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3
            )
        ], p=0.5),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.2),

        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=2,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05)
            )
        ], p=0.3),

        ResizeKeepAspectRatio(height=32, max_width=256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def base_pipeline():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        ResizeKeepAspectRatio(height=32, max_width=256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])