import math
import os

import keras
import keras.backend as k
import numpy as np
import tensorflow as tf
from git import Repo
from keras.callbacks import TensorBoard
from keras.layers import Add, Input, Dense, Reshape, Conv2D, Flatten, LeakyReLU, Dropout, Conv2DTranspose
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import Progbar
from keras_contrib.layers import InstanceNormalization

import utils


class OurGAN:
    def __init__(self, noise_dim, cond_dim, img_dim, channels, path):
        """
        训练器 初始化
        """
        self.result_path = os.path.abspath(path)
        dirs = [".", "ev_img", "gen_img", "model"]
        for item in dirs:
            if not os.path.exists(os.path.join(self.result_path, item)):
                os.makedirs(os.path.join(self.result_path, item))
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.img_dim = img_dim
        self.channels = channels

        self._setup()

        self.a_noise = np.random.normal(size=[64, self.noise_dim])
        self.a_cond = np.random.uniform(-1., 1., size=[64, self.cond_dim])
        self.tb = TensorBoard(log_dir=self.result_path, write_images=True)
        self.tb.set_model(self.all_net)

    @staticmethod
    def name():
        return "OurGAN"

    def _setup(self):
        self.g_opt = Adam(2e-4, 0.8)
        self.g_l1_opt = Adam(4e-5, 0.8)
        self.d_opt = Adam(2e-4, 0.8)

        self.init_dim = 8
        self.kernel_size = 5
        self.residual_kernel_size = 5
        self.conv_layers = int(math.log2(self.img_dim / self.init_dim))

        self.noise_input = Input(shape=(self.noise_dim,))
        self.cond_input = Input(shape=(self.cond_dim,))
        self.img_input = Input(shape=(self.img_dim, self.img_dim, self.channels,))

        self._setup_layers()
        self._setup_g("G")
        self._setup_d("D")
        self._setup_u_net()
        self._setup_gan("GAN")

        self.discriminator.compile(self.d_opt, 'binary_crossentropy', metrics=['accuracy'])
        self.generator.compile(self.g_l1_opt, 'mae')
        self.u_net.compile(self.g_l1_opt, "mae")

        self.discriminator.trainable = False
        self.gan.compile(self.g_opt, ['binary_crossentropy', "binary_crossentropy"])

    def _setup_layers(self):
        self.layers = {}
        self.conv_filter = [512, 512, 256, 128, 64]
        # Out:16*16*512
        self.layers["g_16"] = [
            Conv2DTranspose(self.conv_filter[1], kernel_size=self.kernel_size, strides=2, padding='same'),
            InstanceNormalization(),
            LeakyReLU(alpha=0.2)
        ]
        self.layers["g_32"] = [
            Conv2DTranspose(self.conv_filter[2], kernel_size=self.kernel_size, strides=2, padding='same'),
            InstanceNormalization(),
            LeakyReLU(alpha=0.2)
        ]
        self.layers["g_64"] = [
            Conv2DTranspose(self.conv_filter[3], kernel_size=self.kernel_size, strides=2, padding='same'),
            InstanceNormalization(),
            LeakyReLU(alpha=0.2)
        ]
        # Out：128*128*64
        self.layers["g_128"] = [
            Conv2DTranspose(self.conv_filter[4], kernel_size=self.kernel_size, strides=2, padding='same'),
            InstanceNormalization(),
            LeakyReLU(alpha=0.2)
        ]

        self.layers["u_g_16"] = [
            Conv2DTranspose(self.conv_filter[1], kernel_size=self.kernel_size, strides=2, padding='same'),
            InstanceNormalization(),
            LeakyReLU(alpha=0.2)
        ]
        self.layers["u_g_32"] = [
            Conv2DTranspose(self.conv_filter[2], kernel_size=self.kernel_size, strides=2, padding='same'),
            InstanceNormalization(),
            LeakyReLU(alpha=0.2)
        ]
        self.layers["u_g_64"] = [
            Conv2DTranspose(self.conv_filter[3], kernel_size=self.kernel_size, strides=2, padding='same'),
            InstanceNormalization(),
            LeakyReLU(alpha=0.2)
        ]
        # Out：128*128*64
        self.layers["u_g_128"] = [
            Conv2DTranspose(self.conv_filter[4], kernel_size=self.kernel_size, strides=2, padding='same'),
            InstanceNormalization(),
            LeakyReLU(alpha=0.2)
        ]

        # Out:64*64*128
        self.layers["d_128"] = [
            Conv2D(self.conv_filter[3], kernel_size=self.kernel_size, strides=2, padding='same'),
            InstanceNormalization(),
            LeakyReLU(alpha=0.2),
            Dropout(0.25)
        ]
        # Out:32*32*256
        self.layers["d_64"] = [
            Conv2D(self.conv_filter[2], kernel_size=self.kernel_size, strides=2, padding='same'),
            InstanceNormalization(),
            LeakyReLU(alpha=0.2),
            Dropout(0.25)
        ]
        # Out:16*16*512
        self.layers["d_32"] = [
            Conv2D(self.conv_filter[1], kernel_size=self.kernel_size, strides=2, padding='same'),
            InstanceNormalization(),
            LeakyReLU(alpha=0.2),
            Dropout(0.25)
        ]
        # Out:8*8*512
        self.layers["d_16"] = [
            Conv2D(self.conv_filter[0], kernel_size=self.kernel_size, strides=2, padding='same'),
            InstanceNormalization(),
            LeakyReLU(alpha=0.2),
            Dropout(0.25)
        ]

        self.layers["c_8"] = [Dense(8 ** 2), Reshape([8, 8, 1])]
        self.layers["c_16"] = [Dense(16 ** 2), Reshape([16, 16, 1])]
        self.layers["c_32"] = [Dense(32 ** 2), Reshape([32, 32, 1])]
        self.layers["c_64"] = [Dense(64 ** 2), Reshape([64, 64, 1])]

    def _setup_g(self, name):

        # 8x8
        x = Concatenate()([self.noise_input, self.cond_input])
        x = Dense(self.init_dim ** 2 * self.conv_filter[0], activation=LeakyReLU(0.2))(x)  # 不可使用两次全连接
        x = Reshape([self.init_dim, self.init_dim, self.conv_filter[0]])(x)
        x = InstanceNormalization()(x)

        c = OurGAN.add_sequential_layer(self.cond_input, self.layers["c_8"])
        x = Concatenate()([x, c])
        x = OurGAN.add_sequential_layer(x, self.layers["g_16"])

        c = OurGAN.add_sequential_layer(self.cond_input, self.layers["c_16"])
        x = Concatenate()([x, c])
        x = OurGAN.add_sequential_layer(x, self.layers["g_32"])
        # 64x64
        c = OurGAN.add_sequential_layer(self.cond_input, self.layers["c_32"])
        x = Concatenate()([x, c])
        x = OurGAN.add_sequential_layer(x, self.layers["g_64"])
        # 128x128
        c = OurGAN.add_sequential_layer(self.cond_input, self.layers["c_64"])
        x = Concatenate()([x, c])
        x = OurGAN.add_sequential_layer(x, self.layers["g_128"])

        self.g_output = Conv2D(self.channels, kernel_size=self.kernel_size, padding='same', activation='tanh')(x)
        self.generator = Model(inputs=[self.noise_input, self.cond_input], outputs=[self.g_output], name=name)

    def _setup_d(self, d_name):
        x = self.img_input
        x = OurGAN.add_sequential_layer(x, self.layers["d_128"])
        x = OurGAN.add_sequential_layer(x, self.layers["d_64"])
        x = OurGAN.add_sequential_layer(x, self.layers["d_32"])
        x = OurGAN.add_sequential_layer(x, self.layers["d_16"])

        x = Flatten()(x)
        self.d_output = Dense(1, activation="sigmoid")(x)

        x = Dense(128)(x)
        x = Concatenate()([x, self.cond_input])
        self.dc_output = Dense(1, activation="sigmoid")(x)

        self.discriminator = Model(inputs=[self.img_input, self.cond_input], outputs=[self.d_output, self.dc_output],
                                   name=d_name)

    @staticmethod
    def add_sequential_layer(layer_in, layers_add):
        layer_out = layer_in
        for layer in layers_add:
            layer_out = layer(layer_out)
        return layer_out

    @staticmethod
    def _residual_block(layer, n_conv, kernel):
        x = Conv2D(n_conv, kernel_size=kernel, strides=1, padding='same')(layer)
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(int(layer.shape[-1]), kernel_size=kernel, strides=1, padding='same')(x)
        x = Add()([layer, x])
        return x

    def _setup_u_net(self):

        x = self.img_input
        d_128 = OurGAN.add_sequential_layer(x, self.layers["d_128"])
        d_64 = OurGAN.add_sequential_layer(d_128, self.layers["d_64"])
        d_32 = OurGAN.add_sequential_layer(d_64, self.layers["d_32"])
        x = OurGAN.add_sequential_layer(d_32, self.layers["d_16"])

        x = OurGAN._residual_block(x, self.conv_filter[0], self.residual_kernel_size)
        c = OurGAN.add_sequential_layer(self.cond_input, self.layers["c_8"])
        x = Concatenate()([x, c])
        x = OurGAN.add_sequential_layer(x, self.layers["u_g_16"])

        c = OurGAN.add_sequential_layer(self.cond_input, self.layers["c_16"])
        x = Concatenate()([x, d_32, c])
        x = OurGAN.add_sequential_layer(x, self.layers["u_g_32"])

        c = OurGAN.add_sequential_layer(self.cond_input, self.layers["c_32"])
        x = Concatenate()([x, d_64, c])
        x = OurGAN.add_sequential_layer(x, self.layers["u_g_64"])

        c = OurGAN.add_sequential_layer(self.cond_input, self.layers["c_64"])
        x = Concatenate()([x, d_128, c])
        x = OurGAN.add_sequential_layer(x, self.layers["u_g_128"])

        x = Conv2D(self.channels, kernel_size=self.kernel_size, padding='same', activation='tanh')(x)

        self.u_net = Model([self.img_input, self.cond_input], x)

    def _setup_gan(self, name):
        g_output = self.generator([self.noise_input, self.cond_input])
        d_output = self.discriminator([g_output, self.cond_input])
        self.gan = Model(inputs=[self.noise_input, self.cond_input], outputs=d_output, name=name)
        u_output = self.u_net([g_output, self.cond_input])
        d_output_2 = self.discriminator([u_output, self.cond_input])
        self.all_net = Model([self.noise_input, self.cond_input], d_output_2)

    def _train(self, batch_size, data_generator):
        # Disc Data
        d_noise = np.random.normal(size=(batch_size, self.noise_dim))
        cond_fake = np.random.uniform(-1., 1., size=[batch_size, self.cond_dim])
        img_fake = self.generator.predict([d_noise, cond_fake])
        img_true, cond_true = data_generator.__next__()
        reverse_cond = 1 - cond_true

        # Gen Data
        gan_noise = np.random.normal(size=(batch_size, self.noise_dim))
        gan2_noise = np.random.normal(size=(batch_size, self.noise_dim))
        gan3_noise = np.random.normal(size=(batch_size, self.noise_dim))
        g_noise = np.random.normal(size=(batch_size, self.noise_dim))
        # Targets
        fake_target = np.zeros(batch_size).astype(float)
        true_target = np.ones(batch_size).astype(float)
        g_target = np.ones((batch_size, 1)).astype(float)

        # Train

        d_loss_fake = self.discriminator.train_on_batch([img_fake, cond_fake], [fake_target, fake_target])
        gan_loss_1 = self.gan.train_on_batch([gan_noise, cond_true], [g_target, g_target])
        # u_loss_1 = self.u_net.train_on_batch([np.clip(img_true - img_true, -1, 1), cond_true], img_true)

        d_loss_true = self.discriminator.train_on_batch([img_true, cond_true], [true_target, true_target])
        gan_loss_2 = self.gan.train_on_batch([gan2_noise, cond_true], [g_target, g_target])
        # g_loss = self.generator.train_on_batch([g_noise, cond_true], img_true)

        d_loss_cfake = self.discriminator.train_on_batch([img_true, reverse_cond], [true_target, fake_target])
        gan_loss_3 = self.gan.train_on_batch([gan3_noise, cond_true], [g_target, g_target])
        # u_loss_2 = self.u_net.train_on_batch([np.clip(img_fake - img_true, -1, 1), cond_true], img_true)

        # Calculate
        d_loss = (d_loss_true[0] + d_loss_fake[0] + d_loss_cfake[0]) / 3
        d_acc = (d_loss_true[1] + d_loss_fake[1] + d_loss_cfake[1]) / 3
        gan_loss_d = (gan_loss_1[0] + gan_loss_2[0] + gan_loss_3[0]) / 2
        gan_loss_c = (gan_loss_1[1] + gan_loss_2[0] + gan_loss_3[0]) / 2
        gan_loss = (gan_loss_c + gan_loss_d) / 2
        # u_loss = (u_loss_1 + u_loss_2) / 2
        u_loss, g_loss = 0, 0
        rate = gan_loss / d_loss
        if rate > 2:
            k.set_value(self.gan.optimizer.lr, 3e-4)
            k.set_value(self.discriminator.optimizer.lr, 2e-4)
        elif rate < 0.5:
            k.set_value(self.gan.optimizer.lr, 2e-4)
            k.set_value(self.discriminator.optimizer.lr, 3e-4)
        else:
            k.set_value(self.gan.optimizer.lr, 2e-4)
            k.set_value(self.discriminator.optimizer.lr, 2e-4)
        return g_loss, gan_loss_d, gan_loss_c, d_loss, d_acc, u_loss, img_true, img_fake

    def plot(self):
        models = {"G": self.generator, "D": self.discriminator, "U-NET": self.u_net, "GAN": self.gan}
        with open(self.result_path + "/models.txt", "w") as f:
            def print_fn(content):
                print(content + "\n", file=f)

            for item in models:
                pad_len = int(0.5 * (53 - item.__len__()))
                f.write("\r\n\r\n" + "=" * pad_len + "   Model: " + item + "  " + "=" * pad_len + "\r")
                models[item].summary(print_fn=print_fn)
                keras.utils.plot_model(
                    models[item], to_file=self.result_path + "/%s.png" % item, show_shapes=True)

    def fit(self, batch_size, epoch, data, model_freq_batch, model_freq_epoch, img_freq, start_epoch):
        """
        训练方法
        """

        data_generator = data.get_generator()
        batches = data.batches
        repo = Repo(os.path.dirname(os.path.realpath(__file__)))
        repo.archive(open(self.result_path + "/program.tar", "wb"))
        title = ["LossG", "LossGd", "LossGc", "LossD", "AccD", "LossU"]
        for e in range(start_epoch, 1 + epoch):
            progress_bar = Progbar(batches * batch_size)
            for b in range(1, 1 + batches):
                result = self._train(batch_size, data_generator)
                log = result[:6]

                img_true, img_fake = result[6], result[7]
                progress_bar.add(batch_size, values=[x for x in zip(title, log)])
                OurGAN.write_log(self.tb, [str(e) + "/" + x for x in title], log, b)
                if b % img_freq == 0:
                    utils.save_img(utils.combine_images(img_true),
                                   os.path.join(self.result_path, "real.png"))
                    utils.save_img(utils.combine_images(img_fake),
                                   os.path.join(self.result_path, "gen_img/{}-{}.png".format(e, b)))
                    utils.save_img(utils.combine_images(self.generator.predict([self.a_noise, self.a_cond])),
                                   os.path.join(self.result_path, "ev_img/{}-{}.png").format(e, b))
                    utils.save_img(utils.combine_images(self.u_net.predict([img_fake, self.a_cond])),
                                   os.path.join(self.result_path, "ev_img/u-{}-{}.png").format(e, b))
                if b % model_freq_batch == 0:
                    utils.save_weights({"G": self.generator, "D": self.discriminator, "U-Net": self.u_net},
                                       os.path.join(self.result_path, "model"))
            if e % model_freq_epoch == 0:
                utils.save_weights(
                    {"G-" + str(e): self.generator, "D-" + str(e): self.discriminator, "U-Net-" + str(e): self.u_net},
                    os.path.join(self.result_path, "model"))

    def predict(self, condition, noise=None, labels=None):
        batch_size = condition.shape[0]
        if noise is None:
            noise = np.random.normal(size=[batch_size, self.noise_dim])
        np.set_printoptions(threshold=batch_size * self.noise_dim)
        img = utils.combine_images(self.generator.predict([noise, condition]))
        with open(os.path.join(self.result_path, "generate.log"), "w")as f:
            f.write("Generate Image Condition\r\n\r")
            if labels is not None:
                print(labels, "\r\n", file=f)
            lid = 0
            for item in condition:
                lid += 1
                print(lid, item, file=f)
        utils.save_img(img, os.path.join(self.result_path, "generate.png"))
        utils.save_img(img)

    @staticmethod
    def write_log(callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
