import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os.path as p

PATH = p.dirname(__file__)


def normalize(image):
    image = image - np.mean(image)
    std_dev = 3 * np.std(image)
    image = np.maximum(np.minimum(image, std_dev), -std_dev) / std_dev
    image = (image + 1) * 0.5
    return image


def displayNetwork(img):
    (n_pixels, n_images) = img.shape
    pixel_dim = int(np.ceil(np.sqrt(n_pixels)))
    n_row = int(np.ceil(np.sqrt(n_images)))
    n_col = int(np.ceil(n_images / n_row))
    buf = 1
    images = np.ones(shape=(buf + n_row * (pixel_dim + buf), buf + n_col * (pixel_dim + buf)))

    k = 0
    for i in range(n_row):
        for j in range(n_col):
            if k >= n_images:
                break
            x_i = buf + i * (pixel_dim + buf)
            x_j = buf + j * (pixel_dim + buf)
            y_i = x_i + pixel_dim
            y_j = x_j + pixel_dim
            imgData = normalize(img[:, k])
            images[x_i:y_i, x_j:y_j] = imgData.reshape(pixel_dim, pixel_dim)
            k += 1
    plt.imshow(images, cmap=cm.gray, interpolation='bicubic')
    plt.show()


def displayColorNetwork(img):
    (n_pixels, n_images) = img.shape
    n_pixels = int(n_pixels / 3)
    pixel_dim = int(np.ceil(np.sqrt(n_pixels)))
    n_row = int(np.ceil(np.sqrt(n_images)))
    n_col = int(np.ceil(n_images / n_row))
    buf = 1

    R = img[0:n_pixels, :]
    G = img[n_pixels:2 * n_pixels, :]
    B = img[2 * n_pixels:3 * n_pixels, :]

    images = np.ones(shape=(buf + n_row * (pixel_dim + buf), buf + n_col * (pixel_dim + buf), 3))

    k = 0
    for i in range(n_row):
        for j in range(n_col):
            if k >= n_images:
                break
            x_i = i * (pixel_dim + buf)
            y_i = x_i + pixel_dim
            x_j = j * (pixel_dim + buf)
            y_j = x_j + pixel_dim
            r_data = normalize(R[:, k])
            g_data = normalize(G[:, k])
            b_data = normalize(B[:, k])
            images[x_i:y_i, x_j:y_j, 0] = r_data.reshape(pixel_dim, pixel_dim)
            images[x_i:y_i, x_j:y_j, 1] = g_data.reshape(pixel_dim, pixel_dim)
            images[x_i:y_i, x_j:y_j, 2] = b_data.reshape(pixel_dim, pixel_dim)
            k += 1
    Fig, axes = plt.subplots(1, 1)
    axes.imshow(images)
    axes.set_frame_on(False)
    axes.set_axis_off()
    plt.show()
