import requests
from io import BytesIO

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_remote_image(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


def show_histograms(image, title="Histogram"):
    img_arr = np.array(image)

    plt.figure(figsize=(12, 4))
    plt.suptitle(title)

    plt.subplot(1, 4, 1)
    plt.hist(img_arr.flatten(), bins=256)
    plt.title("Cały obraz")

    labels = ["Red", "Green", "Blue"]
    for i in range(3):
        plt.subplot(1, 4, i + 2)
        plt.hist(img_arr[:, :, i].flatten(), bins=256)
        plt.title(labels[i])

    plt.tight_layout()
    plt.show()

def estimate_quality(image):
    img_arr = np.array(image)

    spans = []
    for i in range(3):
        channel = img_arr[:, :, i]
        spans.append(channel.max() - channel.min())

    avg_span = sum(spans) / 3
    print("Średni zakres histogramu:", avg_span)

    if avg_span < 50:
        return "NISKA"
    elif avg_span < 120:
        return "ŚREDNIA"
    else:
        return "WYSOKA"

def improve_quality(image):
    img_arr = np.array(image).astype(np.float32)
    result = np.zeros_like(img_arr)

    for i in range(3):
        channel = img_arr[:, :, i]
        min_val = channel.min()
        max_val = channel.max()

        if max_val > min_val:
            result[:, :, i] = (channel - min_val) * 255 / (max_val - min_val)
        else:
            result[:, :, i] = channel

    return Image.fromarray(result.astype(np.uint8))

if __name__ == "__main__":

    url = "https://upload.wikimedia.org/wikipedia/commons/b/b1/024_Red-chested_cuckoo_at_Kibale_forest_National_Park_Photo_by_Giles_Laurent.jpg"

    print("➡ Wczytywanie obrazu...")
    image = load_remote_image(url)

    show_histograms(image, "Histogram - PRZED")

    quality = estimate_quality(image)
    print("Ocena jakości:", quality)

    if quality == "NISKA":
        print("⚠ Obraz wadliwy — poprawiam jakość...")
        improved = improve_quality(image)

        show_histograms(improved, "Histogram - PO POPRAWIE")

        new_quality = estimate_quality(improved)
        print("Nowa ocena jakości:", new_quality)

    else:
        print("✔ Obraz nie wymaga poprawy")

