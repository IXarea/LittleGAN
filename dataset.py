import tensorflow as tf
from glob import glob


class CelebA:
    def __init__(self, args):
        self.args = args
        self._img_list = []
        for dir_name in args.img_path:
            print(dir_name + "/*." + args.img_ext)
            self._img_list += glob(dir_name + "/*." + args.img_ext)
        self._attributes_list = self._get_attr_list(args.attr_path, args.attr)
        self.batches = len(self._img_list) // args.batch_size
        self.all_label = ["有短髭", "柳叶眉", "有魅力", "有眼袋", "秃头", "有刘海", "大嘴唇", "大鼻子", "黑发", "金发", "睡眼惺松", "棕发", "浓眉",
                          "丰满", "双下巴", "眼镜", "山羊胡", "白发", "浓妆", "高颧骨", "男性", "嘴轻微张开", "八字胡", "眯缝眼", "完全没有胡子", "鹅蛋脸",
                          "白皮肤", "尖鼻子", "发际线高的", "脸红的", "有鬓脚", "微笑", "直发", "卷发", "戴耳环", "戴帽子", "涂口红", "戴项链", "戴领带",
                          "年轻人"]
        self.label = [self.all_label[x] for x in args.attr]
        dataset = tf.data.Dataset.from_tensor_slices((self._img_list, self._attributes_list))
        dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=self._parse, batch_size=args.batch_size, num_parallel_calls=4))
        dataset = dataset.shuffle(buffer_size=2048)
        self.dataset = dataset.prefetch(buffer_size=2048)
        self.iterator = dataset.make_one_shot_iterator()

    def _parse(self, filename, label):
        image = tf.read_file(filename)
        image = tf.image.decode_image(image, self.args.img_channel)
        image.set_shape([self.args.img_dim, self.args.img_dim, self.args.img_channel])
        image = tf.cast(image, tf.float32)
        image = self.data_rescale(image)
        return image, tf.string_to_number(label)

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

    def get_new_iterator(self):
        self.iterator = self.dataset.make_one_shot_iterator()
        return self.iterator

    @staticmethod
    def data_rescale(x):
        return tf.subtract(tf.divide(x, 127.5), 1)

    @staticmethod
    def inverse_rescale(y):
        return tf.round(tf.multiply(tf.add(y, 1), 127.5))
