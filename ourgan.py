import math
import os

import keras.backend as k
import numpy as np
import tensorflow as tf
from git import Repo
from keras.layers import Input, Dense, Reshape, Conv2D, Flatten, LeakyReLU, Dropout, Conv2DTranspose, BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.utils import Progbar, multi_gpu_model, plot_model

from utils import add_sequential_layer, save_img, combine_images, save_weights


class OurGAN:
    def __init__(self, path, arg):
        """
        模型初始化
        """
        print("Build Model Start")

        self.noise_dim = arg.noise
        self.cond_dim = len(arg.attr)
        self.img_dim = arg.img_size
        self.channels = arg.img_channel
        self.conv_filter = [arg.min_filter * 2 ** (4 - x) for x in range(5)]
        self.sess = k.get_session()

        self.result_path = os.path.abspath(path)
        dirs = [".", "ev_img", "gen_img", "model", "events"]
        for item in dirs:
            if not os.path.exists(os.path.join(self.result_path, item)):
                os.makedirs(os.path.join(self.result_path, item))

        self._setup()

        self._train_discriminator = self.discriminator
        self._train_generator = self.generator
        self._train_u_net = self.u_net

        self.current_d_opt = None
        self.current_g_opt = None
        self.current_u_opt = None

        self.d_part_updater = []
        self.g_part_updater = []
        self.u_part_updater = []

        self.writer = None
        self.train_setup = False
        print("Initialize Model OK")

    def _setup(self):

        self.init_dim = 8
        self.kernel_size = 5
        self.residual_kernel_size = 5
        self.conv_layers = int(math.log2(self.img_dim / self.init_dim))

        self.noise_input = Input(shape=(self.noise_dim,))
        self.cond_input = Input(shape=(self.cond_dim,))
        self.img_input = Input(shape=(self.img_dim, self.img_dim, self.channels,))
        # Public Layers
        self.layers = {
            "d_128_64": [
                Conv2D(self.conv_filter[3], kernel_size=self.kernel_size, strides=2, padding='same', name="d_128_64_conv"),
                BatchNormalization(name="d_128_64_norm"), LeakyReLU(alpha=0.2, name="d_128_64_relu"), Dropout(0.25, name="d_128_64_dropout")],
            "d_64_32": [
                Conv2D(self.conv_filter[2], kernel_size=self.kernel_size, strides=2, padding='same', name="d_64_32_conv"),
                BatchNormalization(name="d_64_32_norm"), LeakyReLU(alpha=0.2, name="d_64_32_relu"), Dropout(0.25, name="d_64_32_dropout")],
            "d_32_16": [
                Conv2D(self.conv_filter[1], kernel_size=self.kernel_size, strides=2, padding='same', name="d_32_16_conv"),
                BatchNormalization(name="d_32_16_norm"), LeakyReLU(alpha=0.2, name="d_32_16_relu"), Dropout(0.25, name="d_32_16_dropout")],
            "d_16_8": [
                Conv2D(self.conv_filter[0], kernel_size=self.kernel_size, strides=2, padding='same', name="d_16_8_conv"),
                BatchNormalization(name="d_16_8_norm"), LeakyReLU(alpha=0.2, name="d_16_8_relu"), Dropout(0.25, name="d_16_8_dropout")],

            "g_8_16": [
                Conv2DTranspose(self.conv_filter[1], kernel_size=self.kernel_size, strides=2, padding='same', name="g_8_16_conv"),
                BatchNormalization(name="g_8_16_norm"), LeakyReLU(alpha=0.2, name="g_8_16_relu")],
            "g_16_32": [
                Conv2DTranspose(self.conv_filter[2], kernel_size=self.kernel_size, strides=2, padding='same', name="g_16_32_conv"),
                BatchNormalization(name="g_16_32_norm"), LeakyReLU(alpha=0.2, name="g_16_32_relu")],
            "g_32_64": [
                Conv2DTranspose(self.conv_filter[3], kernel_size=self.kernel_size, strides=2, padding='same', name="g_32_64_conv"),
                BatchNormalization(name="g_32_64_norm"), LeakyReLU(alpha=0.2, name="g_32_64_relu")],
            "g_64_128": [
                Conv2DTranspose(self.conv_filter[4], kernel_size=self.kernel_size, strides=2, padding='same', name="g_64_128_conv"),
                BatchNormalization(name="g_64_128_norm"), LeakyReLU(alpha=0.2, name="g_64_128_relu")],

            "ug_8_16": [
                Conv2DTranspose(self.conv_filter[1], kernel_size=self.kernel_size, strides=2, padding='same', name="ug_8_16_conv"),
                BatchNormalization(name="ug_8_16_norm"), LeakyReLU(alpha=0.2, name="ug_8_16_relu")],
            "ug_16_32": [
                Conv2DTranspose(self.conv_filter[2], kernel_size=self.kernel_size, strides=2, padding='same', name="ug_16_32_conv"),
                BatchNormalization(name="ug_16_32_norm"), LeakyReLU(alpha=0.2, name="ug_16_32_relu")],
            "ug_32_64": [
                Conv2DTranspose(self.conv_filter[3], kernel_size=self.kernel_size, strides=2, padding='same', name="ug_32_64_conv"),
                BatchNormalization(name="ug_32_64_norm"), LeakyReLU(alpha=0.2, name="ug_32_64_relu")],
            "ug_64_128": [
                Conv2DTranspose(self.conv_filter[4], kernel_size=self.kernel_size, strides=2, padding='same', name="ug_64_128_conv"),
                BatchNormalization(name="ug_64_128_norm"), LeakyReLU(alpha=0.2, name="ug_64_128_relu")],

            "c_8": [Dense(8 ** 2 * 64, name="c_8_dense"), Reshape([8, 8, 64], name="c_8_reshape")],
            "c_16": [Dense(16 ** 2 * 32, name="c_16_dense"), Reshape([16, 16, 32], name="c_16_reshape")],
            "c_32": [Dense(32 ** 2 * 16, name="c_32_dense"), Reshape([32, 32, 16], name="c_32_reshape")],
            "c_64": [Dense(64 ** 2 * 8, name="c_64_dense"), Reshape([64, 64, 8], name="c_64_reshape")]
        }
        self._setup_model_d()
        self._setup_model_g()
        self._setup_model_u()

    def _setup_train(self, part_optimizer=False, debug_loss=False):
        print("Initialize Training Start")
        # 构建计算图(正向传播)
        with tf.name_scope("graph"):
            # 输入占位符
            self.p_real_noise = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
            self.p_fake_noise = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
            self.p_real_cond = tf.placeholder(tf.float32, shape=[None, self.cond_dim])
            self.p_fake_cond = tf.placeholder(tf.float32, shape=[None, self.cond_dim])

            self.p_real_img = tf.placeholder(tf.float32, shape=[None, self.img_dim, self.img_dim, self.channels])
            # 生成图像
            self.fake_img = self._train_generator([self.p_fake_noise, self.p_fake_cond])
            self.fake_img_real = self._train_generator([self.p_real_noise, self.p_real_cond])
            # 判别图像
            self.dis_fake = self._train_discriminator([self.fake_img])
            self.dis_real = self._train_discriminator([self.p_real_img])
            # 调整图像
            self.u_img = self._train_u_net([self.fake_img, self.p_real_cond])
            self.dis_u = self._train_discriminator([self.u_img])
            self.u_img_2 = self._train_u_net([self.p_real_img, self.p_real_cond])
            self.dis_u_2 = self._train_discriminator([self.u_img_2])
        # 定义损失函数
        with tf.name_scope("loss"):
            # 生成器损失函数
            gen_loss_dis_d = k.mean(k.square(0.98 - self.dis_fake[0]))
            gen_loss_dis_c = k.mean(k.square(self.p_real_cond - self.dis_fake[1]))
            gen_loss_l1 = k.mean(k.abs(self.p_real_img - self.fake_img_real))
            self.gen_loss = gen_loss_dis_c + gen_loss_dis_d + 0.05 * gen_loss_l1
            # 判别器损失函数
            dis_loss_real_d = k.mean(k.square(0.98 - self.dis_real[0]))
            dis_loss_real_c = k.mean(k.square(self.p_real_cond - self.dis_real[1]))
            dis_loss_fake_d = k.mean(k.square(self.dis_fake[0] - 0.02))
            dis_loss_fake_c = k.mean(k.square(self.p_fake_cond - self.dis_fake[1]))
            self.dis_loss_ori = dis_loss_fake_c + dis_loss_fake_d + dis_loss_real_c + dis_loss_real_d
            # 自编码网络损失函数
            u_loss_dis_d = k.mean(k.square(0.98 - self.dis_u[0]))
            u_loss_dis_c = k.mean(k.square(self.p_real_cond - self.dis_u[1]))
            u_loss_dis_d2 = k.mean(k.square(0.98 - self.dis_u_2[0]))
            u_loss_dis_c2 = k.mean(k.square(self.p_real_cond - self.dis_u_2[1]))
            u_loss_l1 = k.mean(k.abs(self.p_real_img - self.u_img))
            self.u_loss = u_loss_dis_c + u_loss_dis_d + 0.05 * u_loss_l1 + u_loss_dis_d2 + u_loss_dis_c2
            # 梯度惩罚
            alpha = k.random_uniform(shape=[k.shape(self.p_real_noise)[0], 1, 1, 1])
            interp = alpha * self.p_real_img + (1 - alpha) * self.fake_img
            gradients = k.gradients(self._train_discriminator([interp]), [interp])[0]
            gp = tf.sqrt(tf.reduce_mean(tf.square(gradients), axis=1))
            gp = tf.reduce_mean((gp - 1.0) * 2)
            self.dis_loss = gp + self.dis_loss_ori
            print("Initialize Training: Build Graph OK")
        # 训练过程可视化
        with tf.name_scope("summary"):
            tf.summary.scalar("loss/g_loss", self.gen_loss)
            tf.summary.scalar("loss/d_loss", self.dis_loss)
            tf.summary.scalar("loss/u_loss", self.u_loss)
            if debug_loss:
                tf.summary.scalar("loss-dev/d_loss_origin", self.dis_loss_ori)
                tf.summary.scalar("loss-dev/gp", gp)
                tf.summary.scalar("loss-dev/gen_loss_dis_d", gen_loss_dis_d)
                tf.summary.scalar("loss-dev/gen_loss_dis_c", gen_loss_dis_c)
                tf.summary.scalar("loss-dev/gen_loss_l1", gen_loss_l1)
                tf.summary.scalar("loss-dev/dis_loss_real_d", dis_loss_real_d)
                tf.summary.scalar("loss-dev/dis_loss_real_c", dis_loss_real_c)
                tf.summary.scalar("loss-dev/dis_loss_fake_d", dis_loss_fake_d)
                tf.summary.scalar("loss-dev/dis_loss_fake_c", dis_loss_fake_c)
                tf.summary.scalar("loss-dev/u_loss_dis_d", u_loss_dis_d)
                tf.summary.scalar("loss-dev/u_loss_dis_d2", u_loss_dis_d2)
                tf.summary.scalar("loss-dev/u_loss_dis_c", u_loss_dis_c)
                tf.summary.scalar("loss-dev/u_loss_dis_c2", u_loss_dis_c2)
                tf.summary.scalar("loss-dev/u_loss_l1", u_loss_l1)

                def sum_dis_result(dis_result, name):
                    tf.summary.histogram("aux/%s_d" % name, dis_result[0])
                    tf.summary.histogram("aux/%s_c" % name, dis_result[1])
                    tf.summary.scalar("acc/%s_d" % name, tf.reduce_sum(dis_result[0]))
                    tf.summary.scalar("acc/%s_c" % name, tf.reduce_sum(dis_result[1]))

                sum_dis_result(self.dis_real, "real")
                sum_dis_result(self.dis_fake, "fake")
                sum_dis_result(self.dis_u, "u")
            self.merge_summary = tf.summary.merge_all()
            print("Initialize Training: Prepare Visualize OK")
        # 构建优化操作 最小化损失函数(反向传播)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # 全局优化器
            d_updater = tf.train.AdamOptimizer(15e-5, 0.5, 0.9)
            self.d_full_updater = d_updater.minimize(self.dis_loss, var_list=self.discriminator.trainable_weights)
            print("Initialize Training: Build Full Optimizer OK: Discriminator")
            g_updater = tf.train.AdamOptimizer(10e-5, 0.5, 0.9)
            self.g_full_updater = g_updater.minimize(self.gen_loss, var_list=self.generator.trainable_weights)
            print("Initialize Training: Build Full Optimizer OK: Generator")
            u_updater = tf.train.AdamOptimizer(5e-5, 0.5, 0.9)
            self.u_full_updater = u_updater.minimize(self.u_loss, var_list=self.u_net.trainable_weights[12:])
            print("Initialize Training: Build Full Optimizer OK: U-Net")
            # 分块优化器
            if part_optimizer:
                d_train_part = [[self.discriminator.trainable_weights[x] for x in item] for item in self.discriminator_train_list]
                self.d_part_updater = [d_updater.minimize(self.dis_loss, var_list=x) for x in d_train_part]
                print("Initialize Training: Build Part Optimizer OK: Discriminator")
                g_train_part = [[self.generator.trainable_weights[x] for x in item] for item in self.generator_train_list]
                self.g_part_updater = [g_updater.minimize(self.gen_loss, var_list=x) for x in g_train_part]
                print("Initialize Training: Build Part Optimizer OK: Generator")
                u_train_part = [[self.u_net.trainable_weights[x] for x in item] for item in self.u_net_train_list]
                self.u_part_updater = [u_updater.minimize(self.u_loss, var_list=x) for x in u_train_part]
                print("Initialize Training: Build Part Optimizer OK: U-Net")
        self.sess.run(tf.global_variables_initializer())
        self.train_setup = True
        self.current_d_opt = self.d_full_updater
        self.current_g_opt = self.g_full_updater
        self.current_u_opt = self.u_full_updater
        print("Initialize Training OK")

    def _setup_model_d(self):
        x = self.img_input
        x = add_sequential_layer(x, self.layers["d_128_64"])
        x = add_sequential_layer(x, self.layers["d_64_32"])
        x = add_sequential_layer(x, self.layers["d_32_16"])
        x = add_sequential_layer(x, self.layers["d_16_8"])

        x = Flatten()(x)
        # Output whether the image is generated by program
        self.d_output = Dense(1, name="d_img_output", activation='sigmoid')(x)
        # Output if the image's condition
        self.dc_output = Dense(self.cond_dim, name="d_cond_output", activation="sigmoid")(x)

        self.discriminator = Model(inputs=[self.img_input], outputs=[self.d_output, self.dc_output], name="D")
        # Todo: This config is only for no residual and add 2 condition
        self.discriminator_train_list = [
            [16, 17, 18, 19],  # Output Dense
            [12, 13, 14, 15],  # 16->8
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 128->64->32->16
        ]

    def _setup_model_g(self):
        x = Concatenate()([self.noise_input, self.cond_input])
        x = Dense(self.init_dim ** 2 * self.conv_filter[0])(x)  # 不可使用两次全连接
        x = LeakyReLU(0.2)(x)
        x = Reshape([self.init_dim, self.init_dim, self.conv_filter[0]])(x)
        x = BatchNormalization()(x)

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
        self.generator = Model(inputs=[self.noise_input, self.cond_input], outputs=[self.g_output], name="G")
        # Todo: This config is only for no residual and add 2 condition
        self.generator_train_list = [
            [10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23],  # 16->32->64->128 32.6
            [4, 5, 6, 7],  # 8->16 82
            [0, 1, 2, 3],  # Input Image Dense 176
        ]

    def _setup_model_u(self):

        x = self.img_input
        d_64 = add_sequential_layer(x, self.layers["d_128_64"])
        d_32 = add_sequential_layer(d_64, self.layers["d_64_32"])
        d_16 = add_sequential_layer(d_32, self.layers["d_32_16"])
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

        self.u_net = Model([self.img_input, self.cond_input], [x], name="U-Net")
        # Todo: This config is only for no residual and add 2 condition
        self.u_net_train_list = [
            [18, 19, 20, 21, 24, 25, 26, 27],  # 32->64->128
            [14, 15, 16, 17],  # 16->32
            [12, 13, 22, 23],  # Input Cond
        ]

    def _train_step(self, batch_size, data_generator, step):
        # Prepare Data
        img_true, cond_true = data_generator.__next__()
        d_noise = np.random.normal(size=(batch_size, self.noise_dim))
        cond_fake = np.random.uniform(-1., 1., size=[batch_size, self.cond_dim]).round(1)
        g_noise = np.random.normal(size=(batch_size, self.noise_dim))
        # Run Train Operation
        _, _, _, d_loss, g_loss, u_loss, summary, fake_img = self.sess.run(
            [self.current_d_opt, self.current_g_opt, self.current_u_opt, self.dis_loss, self.gen_loss, self.u_loss, self.merge_summary,
             self.fake_img_real],
            {self.p_real_img: img_true, self.p_real_cond: cond_true, self.p_real_noise: d_noise, self.p_fake_noise: g_noise, self.p_fake_cond: cond_fake})
        # Write to Tensorboard
        self.writer.add_summary(summary, step)
        return d_loss, g_loss, u_loss, img_true, fake_img, cond_true

    def plot(self):
        models = {"D": self.discriminator, "G": self.generator, "U": self.u_net}
        with open(self.result_path + "/models.txt", "w") as f:
            def print_fn(content):
                print(content, file=f)

            for item in models:
                pad_len = int(0.5 * (53 - item.__len__()))
                print_fn("=" * pad_len + "   Model: " + item + "  " + "=" * pad_len)
                models[item].summary(print_fn=print_fn)
                print_fn("\n")
            for item in models:
                plot_model(models[item], to_file=self.result_path + "/%s.png" % item, show_shapes=True)

    def fit(self, data, arg):
        """
        公开的训练方法
        """
        # 初始化训练模型和数据
        gpu_num = len(arg.gpu)
        if gpu_num > 1:
            self._train_generator = multi_gpu_model(self.generator, gpu_num)
            self._train_discriminator = multi_gpu_model(self.discriminator, gpu_num)
            self._train_u_net = multi_gpu_model(self.u_net, gpu_num)
        if not self.train_setup:
            self._setup_train(arg.part is 1, arg.debug is 1)
        data_generator = data.get_generator()
        batches = data.batches
        repo = Repo(os.path.dirname(os.path.realpath(__file__)))
        repo.archive(open(self.result_path + "/program.tar", "wb"))
        # 可视化准备
        title = ["LossD", "LossG", "LossU"]
        g_parts = len(self.g_part_updater)
        d_parts = len(self.d_part_updater)
        u_parts = len(self.u_part_updater)
        for e in range(arg.start, 1 + arg.epoch):
            e_log_path = self.result_path + "/events/e-" + str(e)
            if os.path.isdir(e_log_path):
                continue
            os.makedirs(e_log_path)
            self.writer = tf.summary.FileWriter(session=self.sess, logdir=e_log_path)
            if e is arg.start:
                self.writer.add_graph(self.sess.graph)
            print("Epoch " + str(e) + ":\n")
            progress_bar = Progbar(batches * arg.batch_size)
            a_noise = np.random.normal(size=[64, self.noise_dim])
            a_cond = np.random.uniform(-1., 1., size=[64, self.cond_dim]).round(1)
            with open(os.path.join(self.result_path, "ev.log"), "a") as f:
                print("\r\nCondition Label:\r\n", data.label, "\r\nEpoch %d Condition:\r\n" % e, a_cond, "\r\n", file=f)
            for b in range(1, 1 + batches):
                # 切换优化器
                if arg.part is 1:
                    if b % 5 is 0:
                        self.current_g_opt = self.g_part_updater[b // 5 % g_parts]
                        self.current_d_opt = self.d_part_updater[b // 5 % d_parts]
                        self.current_u_opt = self.u_part_updater[b // 5 % u_parts]
                    else:
                        self.current_u_opt = self.u_full_updater
                        self.current_d_opt = self.d_full_updater
                        self.current_g_opt = self.g_full_updater
                # 训练
                result = self._train_step(arg.batch_size, data_generator, (e - 1) * batches + b)
                log = result[:3]

                img_true, img_fake, cond_true = result[3], result[4], result[5]
                progress_bar.add(arg.batch_size, values=[x for x in zip(title, log)])
                # 图片和模型保存
                if b % arg.img_freq == 0:
                    save_img(combine_images(img_true), os.path.join(self.result_path, "real.png"))
                    save_img(combine_images(img_fake), os.path.join(self.result_path, "gen_img/{}-{}.png".format(e, b)))
                    save_img(combine_images(self.generator.predict([a_noise, a_cond])),
                             os.path.join(self.result_path, "ev_img/{}-{}.png").format(e, b))
                    save_img(combine_images(self.u_net.predict([img_fake, a_cond])),
                             os.path.join(self.result_path, "ev_img/u-{}-{}.png").format(e, b))
                    with open(os.path.join(self.result_path, "train_cond.log"), "a") as f:
                        print("\r\nCondition Label:\r\n", data.label, "\r\nEpoch %d Batch %d Condition:\r\n" % (e, b), a_cond, "\r\n", file=f)
                if b % arg.model_freq_batch == 0:
                    save_weights({"G": self.generator, "D": self.discriminator, "U-Net": self.u_net}, os.path.join(self.result_path, "model"))
            if e % arg.model_freq_epoch == 0:
                save_weights({"G-" + str(e): self.generator, "D-" + str(e): self.discriminator, "U-Net-" + str(e): self.u_net},
                             os.path.join(self.result_path, "model"))

    def predict(self, condition, noise=None, labels=None):
        batch_size = condition.shape[0]
        if noise is None:
            noise = np.random.normal(size=[batch_size, self.noise_dim])
        np.set_printoptions(threshold=batch_size * self.noise_dim)
        img = combine_images(self.generator.predict([noise, condition]))
        with open(os.path.join(self.result_path, "generate.log"), "w")as f:
            f.write("Generate Image Condition\r\n\r")
            if labels is not None:
                print(labels, "\r\n", file=f)
            lid = 0
            for item in condition:
                lid += 1
                print(lid, item, file=f)
        save_img(img, os.path.join(self.result_path, "generate.png"))
        save_img(img)
