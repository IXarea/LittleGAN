from keras.layers import Add, Conv2D, LeakyReLU
from keras_contrib.layers import InstanceNormalization


def add_sequential_layer(layer_in, layers_add):
    layer_out = layer_in
    for layer in layers_add:
        layer_out = layer(layer_out)
    return layer_out


def residual_block(layer, n_conv, kernel):
    x = Conv2D(n_conv, kernel_size=kernel, strides=1, padding='same')(layer)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(int(layer.shape[-1]), kernel_size=kernel, strides=1, padding='same')(x)
    x = Add()([layer, x])
    return x
