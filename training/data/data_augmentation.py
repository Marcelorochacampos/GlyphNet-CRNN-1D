import torchvision.transforms.functional as F
import random
from torchvision import transforms
from PIL import Image

class ResizeKeepAspectRatio:
    def __init__(self, height=64, max_width=1024):
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


def data_augmentation_pipeline(configuration):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=configuration["color_jitter"]["brightness"],
                contrast=configuration["color_jitter"]["contrast"]
            )
        ], p=configuration["color_jitter"]["probability"]),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=configuration["gaussian_blur"]["kernel_size"])
        ], p=configuration["gaussian_blur"]["probability"]),

        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=configuration["affine"]["degrees"],
                translate=(configuration["affine"]["translate"], configuration["affine"]["translate"]),
                scale=(configuration["affine"]["scale"][0], configuration["affine"]["scale"][1])
            )
        ], p=configuration["affine"]["probability"]),

        ResizeKeepAspectRatio(height=64, max_width=1024),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def base_pipeline():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        ResizeKeepAspectRatio(height=64, max_width=1024),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])