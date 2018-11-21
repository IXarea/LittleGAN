import numpy as np
from PIL import Image
from tensorflow.python.keras.layers import Add, Conv2D, LeakyReLU
from instance import InstanceNormalization
from dataset import CelebA


def add_sequential_layer(layer_in, layers_add, trainable=None):
    layer_out = layer_in
    for layer in layers_add:
        if trainable is not None:
            layer.trainable = trainable
        layer_out = layer(layer_out)
    return layer_out


def residual_block(layer, n_conv, kernel, leaky_alpha):
    x = Conv2D(n_conv, kernel_size=kernel, strides=1, padding='same')(layer)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = Conv2D(int(layer.shape[-1]), kernel_size=kernel, strides=1, padding='same')(x)
    x = Add()([layer, x])
    return x


def save_image(image, path=None, shape=(None, None)):
    """
    :param EagerTensor image:
    :param string path:
    :param tuple shape:
    :return:
    """
    image = CelebA.inverse_rescale(image).numpy().astype(np.uint8)
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
            i = index // width
            j = index % width
            for l in range(img_channel):
                combined_image[j * img_width:(j + 1) * img_width, i * img_height:(i + 1) * img_height, l] = img[:, :, l]
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
