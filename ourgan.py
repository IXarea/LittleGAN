import math
import os

import keras
import keras.backend as k
import numpy as np
from git import Repo
from keras.layers import Input, Dense, Reshape, Conv2D, Flatten, Activation, LeakyReLU, Dropout, Conv2DTranspose
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD
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
        self.a_cond = np.random.randint(0, 2, size=[64, self.cond_dim])

    @staticmethod
    def name():
        return "OurGAN"

    def _setup(self):
        self.g_opt = Adam(0.0002, 0.5)
        self.g_l1_opt = SGD(0.0001, 0.5)
        self.d_opt = Adam(0.0003, 0.5)

        self.max_conv = 256
        self.init_dim = 8
        self.kernel_size = 5
        self.conv_layers = int(math.log2(self.img_dim / self.init_dim))

        self.noise_input = Input(shape=(self.noise_dim,))
        self.cond_input = Input(shape=(self.cond_dim,))
        self.img_input = Input(shape=(self.img_dim, self.img_dim, self.channels,))
        self._setup_g_1("G")
        self._setup_d_q("D")
        self._setup_gan("GAN")

        self.discriminator.compile(self.d_opt, 'binary_crossentropy', metrics=['accuracy'])
        self.generator.compile(self.g_l1_opt, 'mae')
        # self.q_net.compile(self.d_q_opt, self.mutual_info_loss, metrics=['accuracy'])

        self.discriminator.trainable = False
        self.gan.compile(self.g_opt, ['binary_crossentropy', "binary_crossentropy"])

    def _setup_g_1(self, name):

        n_conv = self.max_conv

        x = Concatenate()([self.noise_input, self.cond_input])
        x = Dense(self.init_dim ** 2 * self.max_conv * 2, activation="relu")(x)  # 不可使用两次全连接
        x = Reshape((self.init_dim, self.init_dim, self.max_conv * 2))(x)
        x = InstanceNormalization()(x)

        for _ in range(1, 1 + self.conv_layers):
            print("G Conv Filters:", n_conv)
            x = Conv2DTranspose(n_conv, kernel_size=self.kernel_size, strides=2, padding='same')(x)
            x = InstanceNormalization()(x)
            x = Activation('relu')(x)
            n_conv = int(n_conv / 2)

        self.g_output = Conv2D(self.channels, kernel_size=self.kernel_size, padding='same', activation='tanh')(x)
        self.generator = Model(inputs=[self.noise_input, self.cond_input], outputs=[self.g_output], name=name)

    def _setup_d_q(self, d_name):
        n_conv = self.max_conv // 2 ** (self.conv_layers - 1)
        x = self.img_input
        for _ in range(1, 1 + self.conv_layers):
            print("D Conv Filters:", n_conv)
            x = Conv2D(n_conv, kernel_size=self.kernel_size, strides=2, padding='same')(x)
            x = InstanceNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.25)(x)
            n_conv = n_conv * 2

        x = Flatten()(x)
        self.d_output = Dense(1, activation="sigmoid")(x)

        x = Dense(128)(x)
        x = Concatenate()([x, self.cond_input])
        self.d2_output = Dense(1, activation="sigmoid")(x)

        self.discriminator = Model(inputs=[self.img_input, self.cond_input], outputs=[self.d_output, self.d2_output],
                                   name=d_name)

    def _setup_gan(self, name):

        d_output = self.discriminator([self.g_output, self.cond_input])
        self.gan = Model(inputs=[self.noise_input, self.cond_input], outputs=d_output, name=name)

    def _train(self, batch_size, data_generator):
        # Disc Data
        d_noise = np.random.normal(size=(batch_size, self.noise_dim))
        cond_fake = np.random.randint(0, 2, size=[batch_size, self.cond_dim])
        img_fake = self.generator.predict([d_noise, cond_fake])
        img_true, cond_true = data_generator.__next__()

        # Gen Data
        gan_noise = np.random.normal(size=(batch_size, self.noise_dim))
        gan_cond = np.random.randint(0, 2, size=[batch_size, self.cond_dim])
        gan2_noise = np.random.normal(size=(batch_size, self.noise_dim))
        gan2_cond = np.random.randint(0, 2, size=[batch_size, self.cond_dim])

        # Targets
        fake_target = np.zeros(batch_size)
        true_target = np.ones(batch_size)
        g_target = np.ones((batch_size, 1))

        # Train
        d_loss_fake = self.discriminator.train_on_batch([img_fake, cond_fake], [fake_target, fake_target])
        gan_loss_1 = self.gan.train_on_batch([gan_noise, gan_cond], [g_target, g_target])
        d_loss_true = self.discriminator.train_on_batch([img_true, cond_true], [true_target, true_target])
        gan_loss_2 = self.gan.train_on_batch([gan2_noise, gan2_cond], [g_target, g_target])

        # Calculate
        d_loss = (d_loss_true[0] + d_loss_fake[0]) / 2
        d_acc = (d_loss_true[1] + d_loss_fake[1]) / 2
        gan_loss_d = (gan_loss_1[0] + gan_loss_2[0]) / 2
        gan_loss_c = (gan_loss_1[1] + gan_loss_2[1]) / 2

        if (gan_loss_d + gan_loss_c) / d_loss > 2:
            g_noise = np.random.normal(size=(batch_size, self.noise_dim))
            g_loss = self.generator.train_on_batch([g_noise, cond_true], img_true)
        else:
            g_loss = 0

        return g_loss, gan_loss_d, gan_loss_c, d_loss, d_acc, img_true, img_fake

    @staticmethod
    def mutual_info_loss(c, c_given_x):
        eps = 1e-8
        conditional_entropy = k.mean(- k.sum(k.log(c_given_x + eps) * c, axis=1))
        entropy = k.mean(- k.sum(k.log(c + eps) * c, axis=1))
        return conditional_entropy + entropy

    def plot(self):
        models = {"G": self.generator, "D": self.discriminator, "GAN": self.gan}
        with open(self.result_path + "/models.txt", "w") as f:
            def print_fn(content):
                f.write(content + "\n")

            for item in models:
                pad_len = int(0.5 * (53 - item.__len__()))
                f.write("\r\n\r\n" + "=" * pad_len + "   Model: " + item + "  " + "=" * pad_len + "\r")
                models[item].summary(print_fn=print_fn)
                keras.utils.plot_model(
                    models[item], to_file=self.result_path + "/%s.png" % item, show_shapes=True)

    def fit(self, batch_size, epoch, data, model_freq_batch, model_freq_epoch, img_freq):
        """
        训练方法
        """

        data_generator = data.get_generator()
        batches = data.batches
        repo = Repo(os.path.dirname(os.path.realpath(__file__)))
        with open(self.result_path + "/train.log", "w") as f:
            f.write("==================  Train Start  ==================")
            f.write("\r\nGit Repo Version: " + repo.head.commit.name_rev)
        repo.archive(open(self.result_path + "/program.tar", "wb"))

        for e in range(1, 1 + epoch):
            progress_bar = Progbar(batches * batch_size)
            for b in range(1, 1 + batches):
                g_loss, gan_loss_d, gan_loss_c, d_loss, d_acc, img_true, img_fake = self._train(batch_size,
                                                                                                data_generator)

                progress_bar.add(batch_size, values=[("LG", g_loss), ("LGd", gan_loss_d), ("LGc", gan_loss_c),
                                                     ("Loss_D", d_loss), ("Acc_D", d_acc)])
                if b % (img_freq // 2) == 0:
                    with open(self.result_path + "/train.log", "a") as f:
                        f.write("\r\n({},{}) G_L:{}, G_Ld:{}, G_Lc:{}, D_L: {}, D_Acc: {}"
                                .format(e, b, g_loss, gan_loss_d, gan_loss_c, d_loss, d_acc))
                if b % img_freq == 0:
                    utils.save_img(utils.combine_images(img_true),
                                   os.path.join(self.result_path, "real.png"))
                    utils.save_img(utils.combine_images(img_fake),
                                   os.path.join(self.result_path, "gen_img/{}-{}.png".format(e, b)))
                    utils.save_img(utils.combine_images(self.generator.predict([self.a_noise, self.a_cond])),
                                   os.path.join(self.result_path, "ev_img/{}-{}.png").format(e, b))
                if b % model_freq_batch == 0:
                    utils.save_weights({"G": self.generator, "D": self.discriminator},
                                       os.path.join(self.result_path, "model"))
            if e % model_freq_epoch == 0:
                utils.save_weights({"G-" + str(e): self.generator, "D-" + str(e): self.discriminator},
                                   os.path.join(self.result_path, "model"))

    def predict(self, batch_size):
        cond = keras.utils.to_categorical(
            ([i for i in range(self.cond_dim)] * (int(batch_size / self.cond_dim) + 1))[0:batch_size])
        noise = np.random.uniform(-1, 1, size=(batch_size, self.noise_dim))
        img = utils.combine_images(self.generator.predict([noise, cond]))
        utils.save_img(img, self.result_path + "/generate.png")
        utils.save_img(img)
