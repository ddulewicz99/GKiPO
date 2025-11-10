import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("Obraz.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.title("Oryginalny obraz")

height, width = image.shape[:2]
new_width = width // 2
new_height = height // 2
resized_image = cv2.resize(image, (new_width, new_height))

gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray_image, cmap='gray')
plt.title("Obraz w skali szarości (50%)")

rotated_image = cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)

plt.imshow(rotated_image, cmap='gray')
plt.title("Obraz obrócony o 90° w prawo")
plt.show()

print("Rozmiar obrazu:", rotated_image.shape)
print("Macierz obrazu (fragment 10x10):")
print(rotated_image[:10, :10])
