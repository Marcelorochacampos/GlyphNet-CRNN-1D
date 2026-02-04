import os
import torch
import json
from torch.utils.data import DataLoader
from data.dataset import OCRDataset
from data.data_augmentation import data_augmentation_pipeline, base_pipeline

from model.crnn import CRNNNetwork
from utils.collate_fn import collate_fn

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3


def main():
    charset_path = "./config/charset.json"
    dataset_path = "../data-collection/collected-data"
    checkpoint_path = "./checkpoint"

    print("\n" + "=" * 80)
    print("[INFO] Starting OCR Training Pipeline")
    print("=" * 80)

    print(f"[INFO] Loading charset from: {charset_path}")
    with open(charset_path, "r") as f:
        charset = json.load(f)

    print(f"[INFO] Charset size: {len(charset)} tokens")

    print("[INFO] Initializing data pipelines")
    train_transforms = data_augmentation_pipeline()
    test_transforms = base_pipeline()
    val_transforms = base_pipeline()

    print(f"[INFO] Loading dataset from: {dataset_path}")
    dataset = OCRDataset(dataset_path, charset)
    N = len(dataset)

    print(f"[INFO] Total samples found: {N}")

    generator = torch.Generator().manual_seed(42)
    index = torch.randperm(N, generator=generator).tolist()

    train_size = int(N * 0.8)
    test_size = int(N * 0.1)
    val_size = int(N - train_size - test_size)

    print(
        f"[INFO] Dataset split -> "
        f"Train: {train_size} | Val: {val_size} | Test: {test_size}"
    )

    train_idx = index[:train_size]
    val_idx = index[train_size:train_size + val_size]
    test_idx = index[train_size + val_size:]

    print("[INFO] Creating dataset subsets")
    train_ds = OCRDataset(dataset_path, charset, transform=train_transforms, index=train_idx)
    test_ds = OCRDataset(dataset_path, charset, transform=test_transforms, index=test_idx)
    val_ds = OCRDataset(dataset_path, charset, transform=val_transforms, index=val_idx)

    print("[INFO] Initializing DataLoaders")
    train_dataloader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Initializing model")
    model = CRNNNetwork(
        num_classes=len(charset),
        img_height=32,
        cnn_out_channels=512,
        rnn_hidden_size=256,
        rnn_num_layers=2
    ).to(device)

    blank_idx = charset["<blank>"]
    criterion = torch.nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("[INFO] Training configuration:")
    print(f"       Epochs: {EPOCHS}")
    print(f"       Batch size: {BATCH_SIZE}")
    print(f"       Learning rate: {LR}")

    print("\n" + "=" * 80)
    print("[TRAIN] Starting training loop")
    print("=" * 80)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        print(f"\n[TRAIN] Epoch {epoch}/{EPOCHS}")

        for batch_idx, (images, labels, target_lengths) in enumerate(train_dataloader, start=1):
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

            if batch_idx % 50 == 0 or batch_idx == len(train_dataloader):
                avg_loss = train_loss / batch_idx
                progress = (batch_idx / len(train_dataloader)) * 100
                print(
                    f"[TRAIN] Batch {batch_idx:5d}/{len(train_dataloader)} "
                    f"({progress:6.2f}%) | "
                    f"Avg Loss: {avg_loss:.4f}"
                )

        train_loss /= len(train_dataloader)
        print(f"[TRAIN] Epoch {epoch} completed | Avg Loss: {train_loss:.4f}")

        os.makedirs(checkpoint_path, exist_ok=True)
        model_path = os.path.join(checkpoint_path, f"ocr_epoch_{epoch}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] Checkpoint saved: {model_path}")

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
        print(f"[VAL] Epoch {epoch} | Avg Loss: {val_loss:.4f}")

    print("\n" + "=" * 80)
    print("[TEST] Running final evaluation")
    print("=" * 80)

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
    print(f"[TEST] Final Test Loss: {test_loss:.4f}")

    print("\n" + "=" * 80)
    print("[INFO] Training pipeline finished successfully")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
