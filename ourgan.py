import math
import os

import keras.backend as k
import numpy as np
import tensorflow as tf
from git import Repo
from keras.layers import Input, Dense, Reshape, Conv2D, Flatten, LeakyReLU, Dropout, Conv2DTranspose
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import Progbar, multi_gpu_model, plot_model
from keras_contrib.layers import InstanceNormalization

import utils
from helper import add_sequential_layer


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
        self.sess = k.get_session()
        self._setup()
        self.writer = None
        self.train_generator = self.generator
        self.train_discriminator = self.discriminator
        self.train_u_net = self.u_net
        self.train_setup = False

        self.current_generator_train = None
        self.current_discriminator_train = None
        self.current_u_net_train = None

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
        self._setup_d()
        self._setup_u_net()
        self._setup_gan("GAN")

    def _setup_train(self):
        self.p_real_noise = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
        self.p_fake_noise = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
        self.p_real_cond = tf.placeholder(tf.float32, shape=[None, self.cond_dim])
        self.p_fake_cond = tf.placeholder(tf.float32, shape=[None, self.cond_dim])

        self.p_real_img = tf.placeholder(tf.float32, shape=[None, self.img_dim, self.img_dim, self.channels])

        self.fake_img = self.train_generator([self.p_fake_noise, self.p_fake_cond])
        self.fake_img_real = self.train_generator([self.p_real_noise, self.p_real_cond])

        self.dis_fake = self.train_discriminator([self.fake_img])
        self.dis_real = self.train_discriminator([self.p_real_img])

        self.u_img = self.train_u_net([self.fake_img, self.p_real_cond])
        self.dis_u = self.train_discriminator([self.u_img])

        self.gen_loss = k.mean(k.abs(1 - self.dis_fake[0])) + k.mean(
            k.abs(self.p_real_cond - self.dis_fake[1])) + 0.2 * k.mean(k.abs(self.fake_img_real - self.p_real_img))

        self.dis_loss_ori = k.mean(k.abs(self.dis_fake[0])) + k.mean(k.abs(1 - self.dis_real[0])) + k.mean(
            k.abs(self.p_real_cond - self.dis_real[1])) + k.mean(k.abs(self.p_fake_cond - self.dis_fake[1]))

        self.u_loss = k.mean(k.abs(1 - self.dis_u[0])) + k.mean(k.abs(self.p_real_cond - self.dis_u[1])) + 0.2 * (
            k.mean(k.abs(self.u_img - self.p_real_img)))

        alpha = k.random_uniform(shape=[k.shape(self.p_real_noise)[0], 1, 1, 1])

        interp = alpha * self.p_real_img + (1 - alpha) * self.fake_img

        gradients = k.gradients(self.train_discriminator([interp]), [interp])[0]
        gp = tf.sqrt(tf.reduce_mean(tf.square(gradients), axis=1))
        gp = tf.reduce_mean((gp - 1.0) * 2)
        self.dis_loss = gp + self.dis_loss_ori

        tf.summary.scalar("loss/g_loss", self.gen_loss)
        tf.summary.scalar("loss/d_loss", self.dis_loss)
        tf.summary.scalar("loss/u_loss", self.u_loss)

        tf.summary.scalar("loss/d_loss_origin", self.dis_loss_ori)

        tf.summary.scalar("misc/gp", gp)

        def sum_dis_result(dis_result, name):
            tf.summary.histogram("aux/%s_d" % name, dis_result[0])
            tf.summary.histogram("aux/%s_c" % name, dis_result[1])
            tf.summary.scalar("acc/%s_d" % name, tf.reduce_sum(dis_result[0]))
            tf.summary.scalar("acc/%s_c" % name, tf.reduce_sum(dis_result[1]))

        sum_dis_result(self.dis_real, "real")
        sum_dis_result(self.dis_fake, "fake")
        sum_dis_result(self.dis_u, "u")

        self.merge_summary = tf.summary.merge_all()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # Todo:confirm here
            self.dis_updater = tf.train.AdamOptimizer(1e-4, 0.5, 0.9) \
                .minimize(self.dis_loss, var_list=self.current_discriminator_train)
            self.gen_updater = tf.train.AdamOptimizer(1e-4, 0.5, 0.9) \
                .minimize(self.gen_loss, var_list=self.current_generator_train)
            self.u_updater = tf.train.AdamOptimizer(2e-4, 0.5, 0.9) \
                .minimize(self.u_loss, var_list=self.current_u_net_train)

        self.sess.run(tf.global_variables_initializer())
        self.train_setup = True

    def _setup_layers(self):
        self.layers = {}
        self.conv_filter = [384, 192, 96, 96, 48]
        # Out:16*16*512
        self.layers["g_8_16"] = [
            Conv2DTranspose(self.conv_filter[1], kernel_size=self.kernel_size, strides=2, padding='same',
                            name="g_8_16_conv"),
            InstanceNormalization(name="g_8_16_norm"),
            LeakyReLU(alpha=0.2, name="g_8_16_relu")
        ]
        self.layers["g_16_32"] = [
            Conv2DTranspose(self.conv_filter[2], kernel_size=self.kernel_size, strides=2, padding='same',
                            name="g_16_32_conv"),
            InstanceNormalization(name="g_16_32_norm"),
            LeakyReLU(alpha=0.2, name="g_16_32_relu")
        ]
        self.layers["g_32_64"] = [
            Conv2DTranspose(self.conv_filter[3], kernel_size=self.kernel_size, strides=2, padding='same',
                            name="g_32_64_conv"),
            InstanceNormalization(name="g_32_64_norm"),
            LeakyReLU(alpha=0.2, name="g_32_64_relu")
        ]
        # Out：128*128*64
        self.layers["g_64_128"] = [
            Conv2DTranspose(self.conv_filter[4], kernel_size=self.kernel_size, strides=2, padding='same',
                            name="g_64_128_conv"),
            InstanceNormalization(name="g_64_128_norm"),
            LeakyReLU(alpha=0.2, name="g_64_128_relu")
        ]

        self.layers["ug_8_16"] = [
            Conv2DTranspose(self.conv_filter[1], kernel_size=self.kernel_size, strides=2, padding='same',
                            name="ug_8_16_conv"),
            InstanceNormalization(name="ug_8_16_norm"),
            LeakyReLU(alpha=0.2, name="ug_8_16_relu")
        ]
        self.layers["ug_16_32"] = [
            Conv2DTranspose(self.conv_filter[2], kernel_size=self.kernel_size, strides=2, padding='same',
                            name="ug_16_32_conv"),
            InstanceNormalization(name="ug_16_32_norm"),
            LeakyReLU(alpha=0.2, name="ug_16_32_relu")
        ]
        self.layers["ug_32_64"] = [
            Conv2DTranspose(self.conv_filter[3], kernel_size=self.kernel_size, strides=2, padding='same',
                            name="ug_32_64_conv"),
            InstanceNormalization(name="ug_32_64_norm"),
            LeakyReLU(alpha=0.2, name="ug_32_64_relu")
        ]
        # Out：128*128*64
        self.layers["ug_64_128"] = [
            Conv2DTranspose(self.conv_filter[4], kernel_size=self.kernel_size, strides=2, padding='same',
                            name="ug_64_128_conv"),
            InstanceNormalization(name="ug_64_128_norm"),
            LeakyReLU(alpha=0.2, name="ug_64_128_relu")
        ]

        # Out:64*64*128
        self.layers["d_128_64"] = [
            Conv2D(self.conv_filter[3], kernel_size=self.kernel_size, strides=2, padding='same', name="d_128_64_conv"),
            InstanceNormalization(name="d_128_64_norm"),
            LeakyReLU(alpha=0.2, name="d_128_64_relu"),
            Dropout(0.25, name="d_128_64_dropout")
        ]
        # Out:32*32*256
        self.layers["d_64_32"] = [
            Conv2D(self.conv_filter[2], kernel_size=self.kernel_size, strides=2, padding='same', name="d_64_32_conv"),
            InstanceNormalization(name="d_64_32_norm"),
            LeakyReLU(alpha=0.2, name="d_64_32_relu"),
            Dropout(0.25, name="d_64_32_dropout")
        ]
        # Out:16*16*512
        self.layers["d_32_16"] = [
            Conv2D(self.conv_filter[1], kernel_size=self.kernel_size, strides=2, padding='same', name="d_32_16_conv"),
            InstanceNormalization(name="d_32_16_norm"),
            LeakyReLU(alpha=0.2, name="d_32_16_relu"),
            Dropout(0.25, name="d_32_16_dropout")
        ]
        # Out:8*8*512
        self.layers["d_16_8"] = [
            Conv2D(self.conv_filter[0], kernel_size=self.kernel_size, strides=2, padding='same', name="d_16_8_conv"),
            InstanceNormalization(name="d_16_8_norm"),
            LeakyReLU(alpha=0.2, name="d_16_8_relu"),
            Dropout(0.25, name="d_16_8_dropout")
        ]
        self.layers["c_8"] = [Dense(8 ** 2 * 64, name="c_8_dense"), Reshape([8, 8, 64], name="c_8_reshape")]
        self.layers["c_16"] = [Dense(16 ** 2 * 32, name="c_16_dense"), Reshape([16, 16, 32], name="c_16_reshape")]
        self.layers["c_32"] = [Dense(32 ** 2 * 16, name="c_32_dense"), Reshape([32, 32, 16], name="c_32_reshape")]
        self.layers["c_64"] = [Dense(64 ** 2 * 8, name="c_64_dense"), Reshape([64, 64, 8], name="c_64_reshape")]

    def _setup_g(self, name):

        # 8x8
        x = Concatenate()([self.noise_input, self.cond_input])
        x = Dense(self.init_dim ** 2 * self.conv_filter[0])(x)  # 不可使用两次全连接
        x = LeakyReLU(0.2)(x)
        x = Reshape([self.init_dim, self.init_dim, self.conv_filter[0]])(x)
        x = InstanceNormalization()(x)

        # c = add_sequential_layer(self.cond_input, self.layers["c_8"])
        # x = Concatenate()([x, c])
        x = add_sequential_layer(x, self.layers["g_8_16"])

        c = add_sequential_layer(self.cond_input, self.layers["c_16"])
        x = Concatenate()([x, c])
        x = add_sequential_layer(x, self.layers["g_16_32"])
        # 64x64
        # c = add_sequential_layer(self.cond_input, self.layers["c_32"])
        # x = Concatenate()([x, c])
        x = add_sequential_layer(x, self.layers["g_32_64"])
        # 128x128
        c = add_sequential_layer(self.cond_input, self.layers["c_64"])
        x = Concatenate()([x, c])
        x = add_sequential_layer(x, self.layers["g_64_128"])

        self.g_output = Conv2D(self.channels, kernel_size=self.kernel_size, padding='same', activation='tanh')(x)
        self.generator = Model(inputs=[self.noise_input, self.cond_input], outputs=[self.g_output], name=name)
        # Todo:Change here when modified the model
        self.generator_train_list = [
            [0, 1, 2, 3],  # Input Image Dense
            [4, 5, 6, 7],  # 8->16
            [8, 9, 18, 19],  # Input Condition Dense
            [10, 11, 12, 13],  # 16->32
            [14, 15, 16, 17],  # 32->64
            [20, 21, 22, 23],  # 64->128
            [24, 25]  # Output Conv
        ]

    def _setup_d(self):
        x = self.img_input
        x = add_sequential_layer(x, self.layers["d_128_64"])
        x = add_sequential_layer(x, self.layers["d_64_32"])
        x = add_sequential_layer(x, self.layers["d_32_16"])
        x = add_sequential_layer(x, self.layers["d_16_8"])

        x = Flatten()(x)
        self.d_output = Dense(1, name="d_img_output")(x)  # Output whether the image is generated by program
        # x = Concatenate()([x, self.cond_input])
        self.dc_output = Dense(self.cond_dim, name="d_cond_output")(x)  # Output if the image's condition

        self.discriminator = Model(inputs=[self.img_input], outputs=[self.d_output, self.dc_output])
        self.discriminator_train_list = [
            [0, 1, 2, 3],  # 128->64
            [4, 5, 6, 7],  # 64->32
            [8, 9, 10, 11],  # 32->16
            [12, 13, 14, 15],  # 16->8
            [16, 17, 18, 19],  # Output Dense
        ]

    def _setup_u_net(self):

        x = self.img_input
        d_64 = add_sequential_layer(x, self.layers["d_128_64"], False)
        d_32 = add_sequential_layer(d_64, self.layers["d_64_32"], False)
        d_16 = add_sequential_layer(d_32, self.layers["d_32_16"], False)
        # x = OurGAN.add_sequential_layer(d_16, self.layers["d_16_8"])

        # x = OurGAN._residual_block(x, self.conv_filter[0], self.residual_kernel_size)
        # c = OurGAN.add_sequential_layer(self.cond_input, self.layers["c_8"])
        # x = Concatenate()([x, c])
        # x = OurGAN.add_sequential_layer(x, self.layers["ug_8_16"])

        c = add_sequential_layer(self.cond_input, self.layers["c_16"])
        x = Concatenate()([d_16, c])
        x = add_sequential_layer(x, self.layers["ug_16_32"])
        # out_32 = Conv2D(self.channels, kernel_size=self.kernel_size, padding='same', activation='tanh')(x)

        # c = add_sequential_layer(self.cond_input, self.layers["c_32"])
        # x = Concatenate()([x, d_32, c])
        x = add_sequential_layer(x, self.layers["ug_32_64"])
        # out_64 = Conv2D(self.channels, kernel_size=self.kernel_size, padding='same', activation='tanh')(x)

        c = add_sequential_layer(self.cond_input, self.layers["c_64"])
        x = Concatenate()([x, d_64, c])
        x = add_sequential_layer(x, self.layers["ug_64_128"])

        x = Conv2D(self.channels, kernel_size=self.kernel_size, padding='same', activation='tanh')(x)

        self.u_net = Model([self.img_input, self.cond_input], [x])
        self.u_net_train_list = [
            [0, 1, 10, 11],  # Input Cond
            [2, 3, 4, 5],  # 16->32
            [6, 7, 8, 9],  # 32->64
            [12, 13, 14, 15],  # 32->16
            [16, 17],  # Output Dense
        ]

    def _setup_gan(self, name):
        pass
        # g_output = self.generator([self.noise_input, self.cond_input])
        # d_output = self.discriminator([g_output, self.cond_input])
        # self.gan = Model(inputs=[self.noise_input, self.cond_input], outputs=d_output, name=name)
        # u_output = self.u_net([g_output, self.cond_input])
        # d_output_2 = self.discriminator([u_output, self.cond_input])
        # self.all_net = Model([self.noise_input, self.cond_input], d_output_2)

    def _train(self, batch_size, data_generator, step):
        # Disc Data
        img_true, cond_true = data_generator.__next__()
        d_noise = np.random.normal(size=(batch_size, self.noise_dim))
        cond_fake = np.random.uniform(-1., 1., size=[batch_size, self.cond_dim]).round(1)
        g_noise = np.random.normal(size=(batch_size, self.noise_dim))

        _, _, _, d_loss, d_loss_ori, g_loss, u_loss, summary, fake_img = self.sess.run(
            [self.dis_updater, self.gen_updater, self.u_updater, self.dis_loss, self.dis_loss_ori, self.gen_loss,
             self.u_loss, self.merge_summary, self.fake_img_real],
            {self.p_real_img: img_true, self.p_real_cond: cond_true, self.p_real_noise: d_noise,
             self.p_fake_noise: g_noise, self.p_fake_cond: cond_fake})
        self.writer.add_summary(summary, step)
        return g_loss, d_loss, d_loss_ori, u_loss, img_true, fake_img, cond_true

    def plot(self):
        models = {"G": self.generator, "D": self.discriminator, "U-NET": self.u_net}
        with open(self.result_path + "/models.txt", "w") as f:
            def print_fn(content):
                print(content, file=f)

            for item in models:
                pad_len = int(0.5 * (53 - item.__len__()))
                print_fn("=" * pad_len + "   Model: " + item + "  " + "=" * pad_len)
                models[item].summary(print_fn=print_fn)
                print_fn("\n")
                plot_model(models[item], to_file=self.result_path + "/%s.png" % item, show_shapes=True)

    def fit(self, batch_size, epoch, data, model_freq_batch, model_freq_epoch, img_freq, start_epoch, gpu_list):
        """
        训练方法
        """
        if gpu_list.__len__() > 1:
            self.train_generator = multi_gpu_model(self.generator)
            self.train_discriminator = multi_gpu_model(self.discriminator)
            self.train_u_net = multi_gpu_model(self.u_net)
        if not self.train_setup:
            self._setup_train()
        data_generator = data.get_generator()
        batches = data.batches
        repo = Repo(os.path.dirname(os.path.realpath(__file__)))
        repo.archive(open(self.result_path + "/program.tar", "wb"))
        title = ["LossG", "LossD", "LossDo", "LossU"]
        self.writer = tf.summary.FileWriter(session=self.sess, logdir=self.result_path, graph=self.sess.graph)
        for e in range(start_epoch, 1 + epoch):
            progress_bar = Progbar(batches * batch_size)
            a_noise = np.random.normal(size=[64, self.noise_dim])
            a_cond = np.random.uniform(-1., 1., size=[64, self.cond_dim]).round(1)
            with open(os.path.join(self.result_path, "ev.log"), "a") as f:
                print("\r\nCondition Label:\r\n", data.label, "\r\nEpoch %d Condition:\r\n" % e, a_cond,
                      "\r\n", file=f)
            for b in range(1, 1 + batches):
                if b % 2 is 1:
                    self.current_u_net_train = self.u_net.trainable_weights
                    self.current_discriminator_train = self.discriminator.trainable_weights
                    self.current_generator_train = self.generator.trainable_weights
                else:
                    self.current_generator_train = \
                        [self.generator.trainable_weights[x] for x in
                         self.generator_train_list[b // 2 % len(self.generator_train_list)]]
                    self.current_discriminator_train = \
                        [self.discriminator.trainable_weights[x] for x in
                         self.discriminator_train_list[b // 2 % len(self.discriminator_train_list)]]
                    self.current_u_net_train = [self.u_net.trainable_weights[x] for x in
                                                self.u_net_train_list[b // 2 % len(self.u_net_train_list)]]
                result = self._train(batch_size, data_generator, e * batches + b)
                log = result[:4]

                img_true, img_fake, cond_true = result[4], result[5], result[6]
                progress_bar.add(batch_size, values=[x for x in zip(title, log)])
                if b % img_freq == 0:
                    utils.save_img(utils.combine_images(img_true),
                                   os.path.join(self.result_path, "real.png"))
                    utils.save_img(utils.combine_images(img_fake),
                                   os.path.join(self.result_path, "gen_img/{}-{}.png".format(e, b)))
                    utils.save_img(utils.combine_images(self.generator.predict([a_noise, a_cond])),
                                   os.path.join(self.result_path, "ev_img/{}-{}.png").format(e, b))
                    utils.save_img(utils.combine_images(self.u_net.predict([img_fake, a_cond])),
                                   os.path.join(self.result_path, "ev_img/u-{}-{}.png").format(e, b))
                    with open(os.path.join(self.result_path, "train_cond.log"), "a") as f:
                        print("\r\nCondition Label:\r\n", data.label, "\r\nEpoch %d Batch %d Condition:\r\n" % (e, b),
                              a_cond,
                              "\r\n", file=f)
                if b % model_freq_batch == 0:
                    utils.save_weights({"G": self.generator, "D": self.discriminator, "U-Net": self.u_net},
                                       os.path.join(self.result_path, "model"))
            if e % model_freq_epoch == 0:
                utils.save_weights(
                    {"G-" + str(e): self.generator, "D-" + str(e): self.discriminator, "U-Net-" + str(e): self.u_net},
                    os.path.join(self.result_path, "model")
                )

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
