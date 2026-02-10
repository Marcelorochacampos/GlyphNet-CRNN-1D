import os
import json
import time
import torch
import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader, ConcatDataset, Subset
from jiwer import cer, wer

from data.dataset import OCRDataset
from data.data_augmentation import data_augmentation_pipeline, base_pipeline
from model.crnn import CRNNNetwork
from utils.collate_fn import collate_fn
from utils.update_model_progress import update_model_progress


# =========================================================
# Utils
# =========================================================

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model_config(checkpoint_path, target_id="model_0014"):
    log(f"Loading model progress checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "r") as f:
        data = json.load(f)

    for model in data["models"]:
        if model["id"] == target_id:
            log(f"Found model configuration: {target_id}")
            return model

    raise ValueError(f"Model {target_id} not found")


def save_training_checkpoint(path, epoch, model, optimizer, best_val_loss, epochs_no_improve):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve
    }, path)


def load_training_checkpoint(path, model, optimizer):
    log("Resuming training from checkpoint")
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"], ckpt["best_val_loss"], ckpt["epochs_no_improve"]


def decode_predictions(logits, charset, blank_idx):
    preds = logits.argmax(2).permute(1, 0)
    texts = []

    for pred in preds:
        prev = None
        chars = []
        for idx in pred:
            idx = idx.item()
            if idx != blank_idx and idx != prev:
                chars.append(charset[idx])
            prev = idx
        texts.append("".join(chars))
    return texts


# =========================================================
# Main
# =========================================================

def main():
    log("Starting OCR training pipeline")

    charset_path = "./config/charset.json"
    dataset_paths = [
        "../data-collection/collected-data-wikipedia",
        "../data-collection/collected-data-wikisource",
        "../data-collection/collected-data-synthetic"
    ]

    model_progress_checkpoint = "./checkpoint/model_progress_checkpoint.json"
    model_checkpoint_dir = "./checkpoint/model"
    training_resume_path = "./checkpoint/training_resume.pt"

    os.makedirs(model_checkpoint_dir, exist_ok=True)

    # ---------------- Charset ----------------
    log("Loading charset")
    with open(charset_path, "r") as f:
        charset = json.load(f)
    num_classes = len(charset)
    log(f"Charset loaded ({num_classes} tokens)")

    # ---------------- Model config ----------------
    model_info = load_model_config(model_progress_checkpoint, "model_0014")
    configuration = model_info["parameters"]

    # ---------------- Dataset loading ----------------
    log("Loading base datasets")
    base_datasets = []
    for p in dataset_paths:
        ds = OCRDataset(p, charset)
        base_datasets.append(ds)
        log(f"Loaded dataset {p} with {len(ds)} samples")

    full_dataset = ConcatDataset(base_datasets)
    N = len(full_dataset)
    log(f"Total samples (all datasets): {N}")

    # ---------------- Split ----------------
    log("Splitting dataset")
    generator = torch.Generator().manual_seed(42)
    idx = torch.randperm(N, generator=generator).tolist()

    train_size = int(N * 0.8)
    val_size = int(N * 0.1)

    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size + val_size]
    test_idx = idx[train_size + val_size:]

    log(f"Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")

    # ---------------- Transforms ----------------
    log("Building transformed datasets")

    train_ds = Subset(
        ConcatDataset([
            OCRDataset(p, charset, transform=data_augmentation_pipeline(configuration["augmentation"]))
            for p in dataset_paths
        ]),
        train_idx
    )

    val_ds = Subset(
        ConcatDataset([
            OCRDataset(p, charset, transform=base_pipeline())
            for p in dataset_paths
        ]),
        val_idx
    )

    test_ds = Subset(
        ConcatDataset([
            OCRDataset(p, charset, transform=base_pipeline())
            for p in dataset_paths
        ]),
        test_idx
    )

    # ---------------- DataLoaders ----------------
    log("Creating DataLoaders (this can take some time...)")

    train_loader = DataLoader(
        train_ds,
        batch_size=configuration["data"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=configuration["data"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=configuration["data"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    log("DataLoaders ready")

    # ---------------- Model ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    log("Initializing model")
    model = CRNNNetwork(
        num_classes=num_classes,
        cnn_out_channels=configuration["model"]["cnn"]["out_channels"],
        rnn_hidden_size=configuration["model"]["rnn"]["hidden_size"],
        rnn_num_layers=configuration["model"]["rnn"]["num_layers"]
    ).to(device)

    blank_idx = charset[configuration["ctc"]["blank_token"]]
    criterion = torch.nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configuration["optimizer"]["lr"],
        weight_decay=configuration["optimizer"]["weight_decay"]
    )

    # ---------------- Resume ----------------
    start_epoch = 1
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = None

    if os.path.exists(training_resume_path):
        start_epoch, best_val_loss, epochs_no_improve = load_training_checkpoint(
            training_resume_path, model, optimizer
        )

    # ---------------- MLflow ----------------
    mlflow.set_experiment("crnn_ocr_gridsearch")
    mlflow.start_run(run_name="model_0014_final")

    mlflow.log_params(configuration)
    mlflow.set_tag("phase", "final_training")
    mlflow.set_tag("dataset_sources", "wikipedia+wikisource+synthetic")

    try:
        log("Starting training loop")

        for epoch in range(start_epoch, configuration["training"]["epochs"] + 1):
            epoch_start = time.time()
            log(f"Epoch {epoch} started")

            model.train()
            train_loss = 0.0

            for batch_idx, (images, labels, target_lengths) in enumerate(train_loader, 1):
                images = images.to(device)
                labels = labels.to(device)
                target_lengths = target_lengths.to(device)

                optimizer.zero_grad()
                logits = model(images)
                log_probs = logits.log_softmax(2)

                T, B, _ = log_probs.shape
                input_lengths = torch.full((B,), T, device=device, dtype=torch.long)

                loss = criterion(log_probs, labels, input_lengths, target_lengths)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if batch_idx % 100 == 0:
                    avg_loss = train_loss / batch_idx
                    progress = (batch_idx / len(train_loader)) * 100
                    log(
                        f"[TRAIN] Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} "
                        f"({progress:6.2f}%) | "
                        f"Avg Loss: {avg_loss:.4f}"
                    )

            train_loss /= len(train_loader)

            # ---------------- Validation ----------------
            log(f"Epoch {epoch} validation started")
            model.eval()
            val_loss = 0.0
            all_preds, all_gt = [], []

            with torch.no_grad():
                for val_batch_idx, (images, labels, target_lengths) in enumerate(val_loader, 1):
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

                    avg_val_loss = val_loss / val_batch_idx
                    val_progress = (val_batch_idx / len(val_loader)) * 100
                    log(
                        f"[VAL] Batch {val_batch_idx}/{len(val_loader)} "
                        f"({val_progress:6.2f}%) | "
                        f"Avg Loss: {avg_val_loss:.4f}"
                    )

            val_loss /= len(val_loader)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            elapsed = time.time() - epoch_start
            log(
                f"Epoch {epoch} finished | ValLoss={val_loss:.4f} | Time={elapsed:.1f}s"
            )

            best_model_path = os.path.join(
                model_checkpoint_dir,
                "model_0014_best.pt"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(
                    model.state_dict(),
                    best_model_path
                )
                log("New best model saved")
            else:
                epochs_no_improve += 1
                log(f"No improvement ({epochs_no_improve})")

            save_training_checkpoint(
                training_resume_path,
                epoch + 1,
                model,
                optimizer,
                best_val_loss,
                epochs_no_improve
            )

            if epochs_no_improve >= configuration["training"]["early_stopping"]["patience"]:
                log("Early stopping triggered")
                break

        mlflow.end_run(status="FINISHED")

    except KeyboardInterrupt:
        log("Training interrupted by user")
        mlflow.set_tag("interrupted", True)
        mlflow.end_run(status="KILLED")
        return

    except Exception as e:
        mlflow.end_run(status="FAILED")
        raise e

    update_model_progress(
        model_progress_checkpoint,
        "model_0014",
        best_val_loss,
        os.path.join(model_checkpoint_dir, "model_0014_best.pt")
    )

     # ---------- Final Test ----------
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for images, labels, target_lengths in test_loader:
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

    test_loss /= len(test_loader)
    print(f"[TEST] Final Test Loss: {test_loss:.4f}")
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_artifact(best_model_path)

    log("Training completed successfully")


if __name__ == "__main__":
    main()
