import os
import torch
import json
from torch.utils.data import DataLoader
from training.data.dataset import OCRDataset
from training.data.data_augmentation import data_augmentation_pipeline, base_pipeline

from model.crnn import CRNNNetwork
from utils.collate_fn import collate_fn

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

def main():
    with open("./config/charset.json", "r") as f:
        charset = json.load(f)

    train_transforms = data_augmentation_pipeline()
    test_transforms = base_pipeline()
    val_transforms = base_pipeline()

    dataset = OCRDataset("../data-collection/collected-data", charset)

    N = len(dataset)

    generator = torch.Generator().manual_seed(42)
    index = torch.randperm(N, generator=generator).tolist()

    train_size = int(N * 0.8)
    test_size = int(N * 0.1)
    val_size = int(N - train_size - test_size)

    train_idx = index[:train_size]
    val_idx = index[train_size:train_size+val_size]
    test_idx = index[train_size+val_size:]

    train_ds = OCRDataset("../data-collection/data", charset, transform=train_transforms, index=train_idx)
    test_ds = OCRDataset("../data-collection/data", charset, transform=test_transforms, index=test_idx)
    val_ds = OCRDataset("../data-collection/data", charset, transform=val_transforms, index=val_idx)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CRNNNetwork(
        num_classes=len(charset), # Charset size
        img_height=32,
        cnn_out_channels=512,
        rnn_hidden_size=256,
        rnn_num_layers=2
    )
    model.to(device)

    blank_idx = charset["<blank>"]
    criterion = torch.nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0

        print("\nEpoch {epoch}/{epochs}".format(epoch=epoch,epochs=EPOCHS))
        for images, labels, target_lengths in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()

            logits = model(images)
            log_probs = logits.log_softmax(2)

            T, B, _ = log_probs.shape
            input_lengths = torch.full(
                size=(B,),
                fill_value=T,
                dtype=torch.long,
                device=device
            )

            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                target_lengths
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels, target_lengths in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                target_lengths = target_lengths.to(device)

                logits = model(images)
                log_probs = logits.log_softmax(2)

                T, B, _ = log_probs.shape
                input_lengths = torch.full(
                    size=(B,),
                    fill_value=T,
                    dtype=torch.long,
                    device=device
                )

                loss = criterion(
                    log_probs,
                    labels,
                    input_lengths,
                    target_lengths
                )

                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Val Loss: {val_loss:.4f}")

    # Test
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for images, labels, target_lengths in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(images)
            log_probs = logits.log_softmax(2)

            T, B, _ = log_probs.shape
            input_lengths = torch.full(
                size=(B,),
                fill_value=T,
                dtype=torch.long,
                device=device
            )

            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                target_lengths
            )

            test_loss += loss.item()

    test_loss /= len(test_dataloader)
    print(f"\nFinal Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()