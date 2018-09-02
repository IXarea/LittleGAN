import math
import os
import random
from glob import glob

import numpy as np
from PIL import Image


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


class CelebA:

    def __init__(self, path, attr_file, batch_size, shape, img_ext=".jpg", attr_filter=None):
        self.img_list = []
        for dir_name in path:
            self.img_list += glob(dir_name + "/*" + img_ext)
        self.attributes_list = self.get_attr_list(attr_file, attr_filter)
        self.shape = shape
        self.batch_size = batch_size
        self.batches = int(len(self.img_list) / batch_size)
        self.all_label = ["有短髭", "柳叶眉", "有魅力的", "有眼袋", "秃头的", "有刘海", "大嘴唇", "大鼻子", "黑发", "金发", "睡眼惺松的", "棕发", "浓眉",
                          "丰满的", "双下巴", "眼镜", "山羊胡", "白发", "浓妆", "高颧骨", "男性", "嘴轻微张开", "八字胡", "眯缝眼", "完全没有胡子", "鹅蛋脸",
                          "白皮肤", "尖鼻子", "发际线高的", "脸红的", "有鬓脚", "微笑", "直发", "卷发", "戴耳环", "戴帽子", "涂口红", "戴项链", "戴领带",
                          "年轻人"]
        self.label = [self.all_label[x] for x in attr_filter]

    def get_generator(self):
        while True:
            random.shuffle(self.img_list)
            x, y = [], []
            i = 0
            for item in self.img_list:
                img = self.get_img_array(item)
                img_id = int(os.path.splitext(os.path.basename(item))[0]) - 1
                x.append(img)
                attr = self.attributes_list[img_id]
                y.append(attr)
                i += 1
                if i == self.batch_size:
                    yield (np.array(x), np.array(y).astype(int))
                    x, y = [], []
                    i = 0

    def get_img_array(self, img_path):
        im = Image.open(img_path)
        im.thumbnail((self.shape[0], self.shape[1]))
        if self.shape[2] == 1:
            im.convert("L")
        img = data_rescale(np.array(im).reshape(self.shape))
        return img

    @staticmethod
    def get_attr_list(attr_file, attr_filter):
        attributes_list_raw = open(attr_file).read().splitlines()
        attributes_list = []
        for item in attributes_list_raw:
            attr_raw = item.split()[1:]
            if attr_filter is None:
                attributes_list.append(attr_raw)
            else:
                attributes_list.append([attr_raw[x] for x in attr_filter])
        return attributes_list
