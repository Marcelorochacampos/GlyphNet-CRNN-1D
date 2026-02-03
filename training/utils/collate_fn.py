import torch

def collate_fn(batch):
    images, labels = zip(*batch)

    images = torch.stack(images, dim=0)

    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat(labels)

    return images, labels, target_lengths