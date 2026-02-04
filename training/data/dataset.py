import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class OCRDataset(Dataset):
    def __init__(self, dir_path, charset, transform=None, index=None):
        if not os.path.isdir(dir_path):
            raise ValueError("Invalid path, directory does not exist")

        if not isinstance(charset, dict):
            raise ValueError("charset must be a dict: {char: index}")

        if "<unk>" not in charset:
            raise ValueError("charset must contain '<unk>' token")
        
        if index is not None:
            if not isinstance(index, (list, tuple)):
                raise ValueError("index must be a list or tuple of indices")

            self.index = index
        else:
            self.index = None

        self.samples = []

        for item in os.listdir(dir_path):
            sample_dir = os.path.join(dir_path, item)

            img_path = os.path.join(sample_dir, "image.png")
            lbl_path = os.path.join(sample_dir, "label.txt")

            if os.path.isfile(img_path) and os.path.isfile(lbl_path):
                self.samples.append({
                    "image": img_path,
                    "label": lbl_path
                })

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found in dataset directory")

        self.transform = transform
        self.charset = charset
        self.unk_idx = charset["<unk>"]

    def __getitem__(self, idx):
        if self.index is not None:
            real_idx = self.index[idx]
        else:
            real_idx = idx

        sample = self.samples[real_idx]
        image = Image.open(sample["image"]).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        with open(sample["label"], "r", encoding="utf-8") as f:
            text = f.read()

        text = text.strip().lower()
        text = " ".join(text.split())

        label = torch.tensor(
            [
                self.charset[c] if c in self.charset else self.unk_idx
                for c in text
            ],
            dtype=torch.long
        )

        return image, label

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        return len(self.samples)
