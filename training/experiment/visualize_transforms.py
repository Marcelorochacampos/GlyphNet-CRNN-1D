import yaml
from PIL import Image
import matplotlib.pyplot as plt
from data.data_augmentation import data_augmentation_pipeline, base_pipeline

def show(image, title):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

image_path = "./experiment/samples/image_01.png"
image = Image.open(image_path).convert("L")

with open("./config/default_parameters.yaml", "r") as f:
    configuration = yaml.safe_load(f)

base = base_pipeline()
aug = data_augmentation_pipeline(configuration["augmentation"])

base_img = base(image)
aug_img = aug(image)

print("Original size:", image.size)
print("After base pipeline:", base_img.shape)
print("After aug pipeline:", aug_img.shape)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
show(image, "Original")

plt.subplot(1, 3, 2)
show(base_img.squeeze(), "Base Pipeline")

plt.subplot(1, 3, 3)
show(aug_img.squeeze(), "Augmented")

plt.show()