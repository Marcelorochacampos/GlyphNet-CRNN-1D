import os
import random
import json
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datetime import datetime

# ========================
# CONFIG
# ========================
OUTPUT_DIR = "./collected-data-synthetic-v2"
FONTS_DIR = "./fonts"

IMG_HEIGHT = 64
MAX_WIDTH = 512
SAMPLES = 100_000

MAX_LINES = 3
LINE_SPACING_RANGE = (2, 6)

BACKGROUND_COLORS = [(255, 255, 255), (245, 245, 245), (235, 235, 235)]
TEXT_COLORS = [(0, 0, 0), (30, 30, 30), (60, 60, 60)]

FONT_SIZE_RANGE = (18, 42)
ROTATION_RANGE = (-2.5, 2.5)

BLUR_PROB = 0.25
NOISE_PROB = 0.25

CHARS = string.ascii_letters + string.digits + " áéíóúãõç.,-"

# ========================
# UTILS
# ========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def random_word(min_len=2, max_len=10):
    return "".join(random.choice(CHARS) for _ in range(random.randint(min_len, max_len)))

def random_multiline_text():
    lines = random.randint(1, MAX_LINES)
    result = []

    for _ in range(lines):
        words = random.randint(2, 8)
        line = " ".join(random_word() for _ in range(words))
        result.append(line)

    return result

def add_noise(img):
    pixels = img.load()
    for _ in range(random.randint(150, 400)):
        x = random.randint(0, img.width - 1)
        y = random.randint(0, img.height - 1)
        v = random.randint(0, 255)
        pixels[x, y] = (v, v, v)
    return img

def load_fonts():
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
    if not os.path.exists(output_dir):
        return 1

    ids = []
    for d in os.listdir(output_dir):
        if d.startswith("sample_"):
            try:
                ids.append(int(d.replace("sample_", "")))
            except:
                pass

    return max(ids) + 1 if ids else 1

# ========================
# GENERATION
# ========================
def generate_sample(sample_id, fonts):
    lines = random_multiline_text()
    label_text = " ".join(lines)

    font_info = random.choice(fonts)
    font_size = random.randint(*FONT_SIZE_RANGE)
    font = ImageFont.truetype(font_info["path"], font_size)

    spacing = random.randint(*LINE_SPACING_RANGE)

    dummy = Image.new("RGB", (MAX_WIDTH, IMG_HEIGHT))
    draw_dummy = ImageDraw.Draw(dummy)

    widths = []
    heights = []

    for line in lines:
        bbox = draw_dummy.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])

    text_w = min(max(widths) + 20, MAX_WIDTH)
    total_text_h = sum(heights) + spacing * (len(lines) - 1)

    img = Image.new(
        "RGB",
        (text_w, IMG_HEIGHT),
        random.choice(BACKGROUND_COLORS)
    )

    draw = ImageDraw.Draw(img)

    y = (IMG_HEIGHT - total_text_h) // 2

    for line, h in zip(lines, heights):
        draw.text(
            (10, y),
            line,
            fill=random.choice(TEXT_COLORS),
            font=font
        )
        y += h + spacing

    # rotation
    angle = random.uniform(*ROTATION_RANGE)
    img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    # resize height back
    img = img.resize((img.width, IMG_HEIGHT))

    # blur
    if random.random() < BLUR_PROB:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.1)))

    # noise
    if random.random() < NOISE_PROB:
        img = add_noise(img)

    # save
    sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_id}")
    ensure_dir(sample_dir)

    img.save(os.path.join(sample_dir, "image.png"))

    with open(os.path.join(sample_dir, "label.txt"), "w", encoding="utf-8") as f:
        f.write(label_text)

    with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "id": sample_id,
                "text": label_text,
                "lines": lines,
                "font_family": font_info["family"],
                "font_name": font_info["name"],
                "font_file": font_info["file"],
                "timestamp": datetime.utcnow().isoformat(),
                "synthetic": True,
                "version": "v2-multiline"
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

    print("[INFO] Synthetic v2 generation finished.")

if __name__ == "__main__":
    main()
