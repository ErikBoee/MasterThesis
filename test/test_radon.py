import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.transform import radon, rescale

pixels = 1000

img = np.zeros((pixels, pixels), np.uint8)

# let's use "for" cycle to change colorspace of pixel in a random way
for x in range(pixels):
    for y in range(pixels):
        # We use "0" for black color (do nothing) and "1" for white color (change pixel value to [255,255,255])
        value = (x-500)**2 + (y-500)**2 < 40000
        if value == 1:
            img[x, y] = 1.0

cv2.imwrite(str(pixels) + "_x_" + str(pixels) + ".png", img)
cv2_image = cv2.imread(str(pixels) + "_x_" + str(pixels) + ".png", 1)


radon_transform = radon(img, theta=[0], circle=True)
plt.plot(np.linspace(0, len(radon_transform), len(radon_transform)), radon_transform)
plt.show()


