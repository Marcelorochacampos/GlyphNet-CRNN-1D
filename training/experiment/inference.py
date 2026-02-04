import torch
import json
from PIL import Image
from torchvision import transforms

from model.crnn import CRNNNetwork
from data.data_augmentation import base_pipeline


# ---------- CTC Greedy Decoder ----------
def ctc_greedy_decode(log_probs, charset, blank_idx):
    """
    log_probs: [T, B, C]
    """
    idx2char = {v: k for k, v in charset.items()}

    preds = log_probs.argmax(dim=2)  # [T, B]
    preds = preds[:, 0].tolist()     # batch = 1

    decoded = []
    prev = None

    for p in preds:
        if p != prev and p != blank_idx:
            decoded.append(idx2char.get(p, ""))
        prev = p

    return "".join(decoded)


# ---------- Inference ----------
def run_inference(
    image_path,
    checkpoint_path,
    charset_path,
    device="cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load charset
    with open(charset_path, "r") as f:
        charset = json.load(f)

    blank_idx = charset["<blank>"]

    # Load model
    model = CRNNNetwork(
        num_classes=len(charset),
        img_height=32,
        cnn_out_channels=512,
        rnn_hidden_size=256,
        rnn_num_layers=2
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Image preprocessing
    transform = base_pipeline()
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)  # [1, 1, H, W]
    image = image.to(device)

    # Forward
    with torch.no_grad():
        logits = model(image)              # [T, B, C]
        log_probs = logits.log_softmax(2)

    # Decode
    text = ctc_greedy_decode(log_probs, charset, blank_idx)

    return text


# ---------- Run ----------
if __name__ == "__main__":
    image_path = "./experiment/samples/image.png"
    checkpoint_path = "./checkpoint/ocr_epoch_10.pt"
    charset_path = "./config/charset.json"

    text = run_inference(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        charset_path=charset_path
    )

    print("\n==============================")
    print(" OCR RESULT ")
    print("==============================")
    print(text)
