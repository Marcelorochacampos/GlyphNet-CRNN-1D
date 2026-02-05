import json
from pathlib import Path

def update_model_progress(
    checkpoint_path: str,
    model_id: str,
    val_loss: float,
    model_path: str
):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )

    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)

    model_found = False

    for model in checkpoint["models"]:
        if model["id"] == model_id:
            model["trained"] = True
            model["val_loss"] = float(val_loss)
            model["model_path"] = model_path
            model_found = True
            break

    if not model_found:
        raise ValueError(
            f"Model with id '{model_id}' not found in checkpoint"
        )

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=4)

    print(
        f"[INFO] Model {model_id} updated | "
        f"val_loss={val_loss:.4f} | "
        f"path={model_path}"
    )
