import math
import os
import random
from glob import glob

import numpy as np
from PIL import Image
from keras.layers import Add, Conv2D, LeakyReLU
from keras_contrib.layers import InstanceNormalization


def add_sequential_layer(layer_in, layers_add, trainable=None):
    layer_out = layer_in
    for layer in layers_add:
        if trainable is not None:
            layer.trainable = trainable
        layer_out = layer(layer_out)
    return layer_out


def residual_block(layer, n_conv, kernel):
    x = Conv2D(n_conv, kernel_size=kernel, strides=1, padding='same')(layer)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(int(layer.shape[-1]), kernel_size=kernel, strides=1, padding='same')(x)
    x = Add()([layer, x])
    return x


def data_rescale(x):
    return x / 127.5 - 1


def inverse_rescale(y):
    return (y + 1) * 127.5


def save_weights(models, path):
    for item in models:
        weights_path = os.path.join(path, '%s_weight.h5' % item)
        models[item].save_weights(weights_path, overwrite=True)


def save_img(img, path=None):
    img = inverse_rescale(np.array(img)).astype(np.uint8)
    if img.shape[2] == 1:
        img = img.reshape(img.shape[0:2])
        mode = "L"
    else:
        mode = "RGB"
    img = Image.fromarray(img, mode)
    if path is None:
        img.show()
    else:
        img.save(path)


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        for l in range(shape[2]):
            image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], l] = img[:, :, l]
    return image


def combine_images2(generated_images, height, width):
    shape = generated_images.shape[1:4]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        for l in range(shape[2]):
            image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], l] = img[:, :, l]
    return image


class CelebA:
    def __init__(self, args):
        self._img_list = []
        for dir_name in args.img_path:
            self._img_list += glob(dir_name + "/*" + args.img_ext)
        self._attributes_list = self._get_attr_list(args.attr_path, args.attr)
        self.shape = (args.img_size, args.img_size, args.img_channel)
        self.batch_size = args.batch_size
        self.batches = int(len(self._img_list) / args.batch_size)
        self.all_label = ["有短髭", "柳叶眉", "有魅力", "有眼袋", "秃头", "有刘海", "大嘴唇", "大鼻子", "黑发", "金发", "睡眼惺松", "棕发", "浓眉",
                          "丰满", "双下巴", "眼镜", "山羊胡", "白发", "浓妆", "高颧骨", "男性", "嘴轻微张开", "八字胡", "眯缝眼", "完全没有胡子", "鹅蛋脸",
                          "白皮肤", "尖鼻子", "发际线高的", "脸红的", "有鬓脚", "微笑", "直发", "卷发", "戴耳环", "戴帽子", "涂口红", "戴项链", "戴领带",
                          "年轻人"]
        self.label = [self.all_label[x] for x in args.attr]

    def get_generator(self):
        while True:
            random.shuffle(self._img_list)
            x, y = [], []
            i = 0
            for item in self._img_list:
                img = self._get_img_array(item)
                img_id = int(os.path.splitext(os.path.basename(item))[0]) - 1
                x.append(img)
                attr = self._attributes_list[img_id]
                y.append(attr)
                i += 1
                if i == self.batch_size:
                    yield (np.array(x).astype(float), np.array(y).astype(float))
                    x, y = [], []
                    i = 0

    def _get_img_array(self, img_path):
        im = Image.open(img_path)
        # im.thumbnail((self.shape[0], self.shape[1]))
        if self.shape[2] == 1:
            im.convert("L")
        img = data_rescale(np.array(im).reshape(self.shape))
        return img

    @staticmethod
    def _get_attr_list(attr_file, attr_filter):
        attributes_list_raw = open(attr_file).read().splitlines()
        attributes_list = []
        for item in attributes_list_raw:
            attr_raw = item.split()[1:]
            if attr_filter is None:
                attributes_list.append(attr_raw)
            else:
                attributes_list.append([attr_raw[x] for x in attr_filter])
        return attributes_list
