from os import path
import json
import time


class FakeArg:
    def __init__(self):
        self.image_channel = 3
        self.batch_size = 2
        self.attr_path = "E:/ds/modify/list_attr_celeba.txt"
        self.image_path = ["E:/ds/data_crop_128_jpg"]
        self.image_path = ["X:\\dataset\\test"]
        self.attr_path = "X:\\dataset\\test.txt"
        self.image_ext = "jpg"
        self.attr = "8,15,20,22,26,36,39"
        _attr = self.attr.split(",")
        self.attr = [int(item) for item in _attr if item.isnumeric() and int(item) >= 0]
        self.cond_dim = len(self.attr)
        self.norm = "instance"
        self.image_dim = 128
        self.min_filter = 24
        self.conv_filter = [self.min_filter * 2 ** (4 - x) for x in range(5)]
        self.kernel_size = 5
        self.leaky_alpha = 0.2
        self.dropout_rate = 0.25
        self.noise_dim = 128
        self.init_dim = 8
        self.l1_lambda = 0.02
        self.lr = 1e-4

        self.gp_weight = 5.0
        self.use_gp = False
        self.use_clip = True
        self.clip_range = 0.5

        self.epoch = 100
        self.freq_gen = 1
        self.freq_test = 5
        self.all_result_dir = "../LittleGAN-result"
        self.exp_name = str(int(time.time())) + "test"
        self.all_result_dir = path.abspath(self.all_result_dir)
        self.result_dir = path.join(self.all_result_dir, self.exp_name)
        self.restore = True
        self.train_adj = True
        self.prefetch = self.batch_size * 5
        self.use_partition = True
        self.partition_interval = 2
        # todo: recover the status from interrupt


if __name__ == "main":
    import tensorflow as tf

    tf.enable_eager_execution()
    from dataset import CelebA
    from littlegan import Trainer, Adjuster, Discriminator, Decoder, Encoder, Generator

    # from ixtest import FakeArg
    arg = FakeArg()
    dec = Decoder(arg)
    enc = Encoder(arg)
    gen = Generator(arg, dec)
    disc = Discriminator(arg, enc)
    adj = Adjuster(arg, enc, dec)
    data = CelebA(arg)
    train = Trainer(arg, gen, disc, adj, data)
    train.train()
