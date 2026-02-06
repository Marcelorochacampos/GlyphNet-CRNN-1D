import os
import torch
import json
import yaml
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from data.dataset import OCRDataset
from data.data_augmentation import data_augmentation_pipeline, base_pipeline

from utils.generate_model_parameters import generate_model_parameters
from model.crnn import CRNNNetwork
from utils.collate_fn import collate_fn
from utils.update_model_progress import update_model_progress


def main():
    charset_path = "./config/charset.json"
    dataset_path = "../data-collection/collected-data"
    model_checkpoint_path = "./checkpoint/model"
    checkpoint_path = "./checkpoint"

    print("\n" + "=" * 80)
    print("[INFO] Starting OCR Training Pipeline")
    print("=" * 80)

    print(f"[INFO] Generating model parameters")
    checkpoint_progress = generate_model_parameters()
    
    remaining_models = []
    for remaining_model in checkpoint_progress["models"]:
        if remaining_model["trained"] is False:
            remaining_models.append(remaining_model)

    if len(remaining_models) <= 0:
        print(f"[INFO] No remaining models to train")
        return

    print(f"[INFO] Loading charset from: {charset_path}")
    with open(charset_path, "r") as f:
        charset = json.load(f)

    print(f"[INFO] Charset size: {len(charset)} tokens")
    print(f"[INFO] Loading dataset from: {dataset_path}")
    dataset = OCRDataset(dataset_path, charset)
    N = len(dataset)
    print(f"[INFO] Total samples found: {N}")

    generator = torch.Generator().manual_seed(42)
    indexes = torch.randperm(N, generator=generator).tolist()

    train_size = int(N * 0.8)
    test_size = int(N * 0.1)
    val_size = int(N - train_size - test_size)
    print(
        f"[INFO] Dataset split -> "
        f"Train: {train_size} | Val: {val_size} | Test: {test_size}"
    )
    train_idx = indexes[:train_size]
    val_idx = indexes[train_size:train_size + val_size]
    test_idx = indexes[train_size + val_size:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    mlflow.set_experiment("crnn_ocr_gridsearch")
    for model_configuration in remaining_models:

        with mlflow.start_run(run_name=f"model_{model_configuration['id']}"):

            configuration = model_configuration["parameters"]
            mlflow.log_params(configuration)

            print("[INFO] Initializing data pipelines")
            train_transforms = data_augmentation_pipeline(configuration["augmentation"])
            test_transforms = base_pipeline()
            val_transforms = base_pipeline()

            train_ds = OCRDataset(dataset_path, charset, transform=train_transforms, index=train_idx)
            test_ds = OCRDataset(dataset_path, charset, transform=test_transforms, index=test_idx)
            val_ds = OCRDataset(dataset_path, charset, transform=val_transforms, index=val_idx)

            BATCH_SIZE = configuration["data"]["batch_size"]

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

            print("[INFO] Initializing model with parameters: {parameters}".format(parameters=configuration["model"]))
            model = CRNNNetwork(
                num_classes=len(charset),
                cnn_out_channels=configuration["model"]["cnn"]["out_channels"],
                rnn_hidden_size=configuration["model"]["rnn"]["hidden_size"],
                rnn_num_layers=configuration["model"]["rnn"]["num_layers"]
            ).to(device)

            EPOCHS = configuration["training"]["epochs"]
            LR = configuration["optimizer"]["lr"]
            patience = configuration["training"]["early_stopping"]["patience"]
            early_stopping_enabled = configuration["training"]["early_stopping"]["enabled"]

            epochs_without_improvement = 0

            blank_idx = charset[configuration["ctc"]["blank_token"]]
            criterion = torch.nn.CTCLoss(blank=blank_idx, zero_infinity=configuration["ctc"]["zero_infinity"])
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LR,
                weight_decay=configuration["optimizer"]["weight_decay"]
            )

            best_val_loss = float("inf")
            best_model_path = None

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

                # ---------- Validation ----------
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

                print(
                    f"[EPOCH {epoch}/{EPOCHS}] "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

                # ---------- Save best model ----------
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    os.makedirs(model_checkpoint_path, exist_ok=True)
                    best_model_path = os.path.join(
                        model_checkpoint_path,
                        f"{model_configuration['id']}_best.pt"
                    )

                    torch.save(model.state_dict(), best_model_path)

                    print(
                        f"[INFO] New best model saved "
                        f"(val_loss={best_val_loss:.4f})"
                    )
                else:
                    epochs_without_improvement += 1
                    print(
                        f"[INFO] No improvement "
                        f"({epochs_without_improvement}/{patience})"
                    )

                if early_stopping_enabled and epochs_without_improvement >= patience:
                    print(
                        f"[EARLY STOPPING] "
                        f"Stopped at epoch {epoch} "
                        f"(best_val_loss={best_val_loss:.4f})"
                    )
                    break

            # ---------- Update progress ----------
            print("[INFO] Updating model progress")
            update_model_progress(
                f"{checkpoint_path}/model_progress_checkpoint.json",
                model_configuration["id"],
                best_val_loss,
                best_model_path
            )

            # ---------- Final Test ----------
            model.load_state_dict(torch.load(best_model_path))
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
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_artifact(best_model_path)

            

    print("\n" + "=" * 80)
    print("[INFO] Training pipeline finished successfully")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
