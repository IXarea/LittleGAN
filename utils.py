import numpy as np
from PIL import Image
import tensorflow as tf


def save_image(image, path=None, shape=(None, None)):
    """
    :param EagerTensor image:
    :param string path:
    :param tuple shape:
    :return:
    """
    image = inverse_rescale(image).numpy().astype(np.uint8)
    if image.shape.__len__() is 4:
        width, height = shape

        if width is None and height is None:
            height = int(np.ceil(np.sqrt(image.shape[0])))
        if width is None:
            width = int(np.ceil(np.divide(image.shape[0], height)))
        if height is None:
            height = int(np.ceil(np.divide(image.shape[0], width)))

        img_width, img_height, img_channel = image.shape[1:4]
        combined_image = np.zeros((width * img_width, height * img_height, img_channel), np.uint8)

        for index, img in enumerate(image):
            y = index // width
            x = index % width
            for layer in range(img_channel):
                combined_image[x * img_width:(x + 1) * img_width, y * img_height:(y + 1) * img_height, layer] = img[:, :, layer]
        image = combined_image

    if image.shape[2] == 1:
        image = image.reshape(image.shape[0:2])
        mode = "L"
    else:
        mode = "RGB"
    image = Image.fromarray(image, mode)

    if path is None:
        image.show()
    else:
        image.save(path)


def soft(x):
    return 0.96 * x + 0.02


def data_rescale(x):
    return tf.subtract(tf.divide(x, 127.5), 1)


def inverse_rescale(y):
    return tf.round(tf.multiply(tf.add(y, 1), 127.5))
