import tensorflow as tf
from glob import glob
from utils import soft, data_rescale
from os import path


class CelebA:
    def __init__(self, args):
        print(" - Initializing Dataset...")
        self.args = args
        self._image_list = glob(path.join(args.image_path, "*." + args.image_ext))
        self._attributes_list = self._get_attr_list(args.attr_path, args.attr)
        self.batches = len(self._image_list) // args.batch_size
        self.all_label = ["有短髭", "柳叶眉", "有魅力", "有眼袋", "秃头", "有刘海", "大嘴唇", "大鼻子", "黑发", "金发",
                          "睡眼惺松", "棕发", "浓眉", "丰满", "双下巴", "眼镜", "山羊胡", "白发", "浓妆", "高颧骨",
                          "男性", "嘴轻微张开", "八字胡", "眯缝眼", "完全没有胡子", "鹅蛋脸", "白皮肤", "尖鼻子", "发际线高的", "脸红的",
                          "有鬓脚", "微笑", "直发", "卷发", "戴耳环", "戴帽子", "涂口红", "戴项链", "戴领带", "年轻人"]
        self.label = [self.all_label[x] for x in args.attr]
        dataset = tf.data.Dataset.from_tensor_slices((self._image_list, self._attributes_list))
        dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=self._parse, batch_size=args.batch_size, num_parallel_calls=args.threads))
        dataset = dataset.shuffle(buffer_size=self.args.prefetch)
        self.dataset = dataset.prefetch(buffer_size=self.args.prefetch)

    def _parse(self, filename, label):
        image = tf.read_file(filename)
        image = tf.image.decode_image(image, self.args.image_channel)

        image.set_shape([self.args.image_dim, self.args.image_dim, self.args.image_channel])
        image = tf.cast(image, tf.float32)
        image = data_rescale(image)

        return image, soft(tf.string_to_number(label))

    @staticmethod
    def _get_attr_list(attr_file, attr_filter):
        with open(attr_file) as f:
            attributes_list_raw = f.read().splitlines()
        attributes_list = []
        for item in attributes_list_raw:
            attr_raw = item.split()[1:]
            if attr_filter is None:
                attributes_list.append(attr_raw)
            else:
                attributes_list.append([attr_raw[x] for x in attr_filter])
        return attributes_list

    def get_new_iterator(self):
        return self.dataset.make_one_shot_iterator()
