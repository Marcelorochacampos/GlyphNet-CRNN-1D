import os
import random
import json
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datetime import datetime

# ========================
# CONFIG
# ========================
OUTPUT_DIR = "./collected-data-synthetic"
FONTS_DIR = "./fonts"

IMG_HEIGHT = 64
MAX_WIDTH = 512
SAMPLES = 100_000

BACKGROUND_COLORS = [(255, 255, 255), (245, 245, 245), (230, 230, 230)]
TEXT_COLORS = [(0, 0, 0), (20, 20, 20), (50, 50, 50)]

FONT_SIZE_RANGE = (18, 48)
ROTATION_RANGE = (-3, 3)
BLUR_PROB = 0.3
NOISE_PROB = 0.3

# ========================
# UTILS
# ========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def random_text():
    length = random.randint(3, 40)
    chars = string.ascii_letters + string.digits + " áéíóúãõç.,-"
    return "".join(random.choice(chars) for _ in range(length))

def add_noise(img):
    pixels = img.load()
    for _ in range(random.randint(100, 300)):
        x = random.randint(0, img.width - 1)
        y = random.randint(0, img.height - 1)
        val = random.randint(0, 255)
        pixels[x, y] = (val, val, val)
    return img

def load_fonts():
    """
    Espera estrutura:
    fonts/
      ├── serif/
      │   └── EB_Garamond/
      │       ├── Regular.ttf
      │       └── Bold.ttf
      ├── sans/
      └── mono/
    """
    fonts = []

    for family in os.listdir(FONTS_DIR):
        family_path = os.path.join(FONTS_DIR, family)
        if not os.path.isdir(family_path):
            continue

        for font_name in os.listdir(family_path):
            font_dir = os.path.join(family_path, font_name)
            if not os.path.isdir(font_dir):
                continue

            for file in os.listdir(font_dir):
                if file.lower().endswith(".ttf"):
                    fonts.append({
                        "family": family,
                        "name": font_name,
                        "path": os.path.join(font_dir, file),
                        "file": file
                    })

    return fonts

def get_start_id(output_dir):
    """
    Encontra o próximo sample_id disponível
    """
    if not os.path.exists(output_dir):
        return 1

    existing = []
    for d in os.listdir(output_dir):
        if d.startswith("sample_"):
            try:
                existing.append(int(d.replace("sample_", "")))
            except ValueError:
                pass

    return max(existing) + 1 if existing else 1

# ========================
# GENERATION
# ========================
def generate_sample(sample_id, fonts):
    text = random_text()
    font_info = random.choice(fonts)

    font_size = random.randint(*FONT_SIZE_RANGE)
    font = ImageFont.truetype(font_info["path"], font_size)

    dummy_img = Image.new("RGB", (MAX_WIDTH, IMG_HEIGHT))
    draw = ImageDraw.Draw(dummy_img)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    img_w = min(text_w + 20, MAX_WIDTH)

    img = Image.new(
        "RGB",
        (img_w, IMG_HEIGHT),
        random.choice(BACKGROUND_COLORS)
    )

    draw = ImageDraw.Draw(img)
    draw.text(
        (10, (IMG_HEIGHT - text_h) // 2),
        text,
        fill=random.choice(TEXT_COLORS),
        font=font
    )

    # Rotation
    angle = random.uniform(*ROTATION_RANGE)
    img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    # Resize height back
    img = img.resize((img.width, IMG_HEIGHT))

    # Blur
    if random.random() < BLUR_PROB:
        img = img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2))
        )

    # Noise
    if random.random() < NOISE_PROB:
        img = add_noise(img)

    # Save
    sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_id}")
    ensure_dir(sample_dir)

    img.save(os.path.join(sample_dir, "image.png"))

    with open(os.path.join(sample_dir, "label.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "id": sample_id,
                "text": text,
                "font_family": font_info["family"],
                "font_name": font_info["name"],
                "font_file": font_info["file"],
                "timestamp": datetime.utcnow().isoformat(),
                "synthetic": True
            },
            f,
            indent=2,
            ensure_ascii=False
        )

# ========================
# MAIN
# ========================
def main():
    ensure_dir(OUTPUT_DIR)

    fonts = load_fonts()
    assert len(fonts) > 0, "Nenhuma fonte encontrada!"

    print(f"[INFO] Loaded {len(fonts)} font files")

    start_id = get_start_id(OUTPUT_DIR)
    print(f"[INFO] Starting from sample_{str(start_id).zfill(6)}")

    end_id = start_id + SAMPLES - 1

    for i in range(start_id, end_id + 1):
        sid = str(i).zfill(6)
        generate_sample(sid, fonts)

        if (i - start_id + 1) % 1000 == 0:
            print(f"[INFO] Generated {i - start_id + 1}/{SAMPLES}")

    print("[INFO] Synthetic data generation finished.")

if __name__ == "__main__":
    main()
