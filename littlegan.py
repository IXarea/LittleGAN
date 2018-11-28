import tensorflow as tf
from instance import InstanceNormalization
from os import path, makedirs
from utils import save_image
import json
import shutil
from git import Repo


class Encoder(tf.keras.Model):
    def __init__(self, args):
        """
        :param Arg args:
        """
        super(Encoder, self).__init__()
        self.args = args
        for i in range(1, 5):
            self.__setattr__("conv" + str(i), tf.layers.Conv2D(self.args.conv_filter[4 - i], self.args.kernel_size, 2, "same"))
            self.__setattr__("norm" + str(i), InstanceNormalization())

    def call(self, inputs, training=None, mask=None):
        x = inputs
        outputs = []
        for i in range(1, 5):
            x = self.__getattribute__("conv" + str(i))(x)
            x = self.__getattribute__("norm" + str(i))(x)
            x = tf.nn.leaky_relu(x, self.args.leaky_alpha)
            x = tf.layers.dropout(x, self.args.dropout_rate)
            outputs.append(x)
        return outputs


class Decoder(tf.keras.Model):
    def __init__(self, args):
        """
        :param Arg args:
        """
        super(Decoder, self).__init__()
        self.args = args
        for i in range(1, 5):
            self.__setattr__("conv" + str(i), tf.layers.Conv2DTranspose(self.args.conv_filter[i], self.args.kernel_size, (2, 2), "same"))
            self.__setattr__("norm" + str(i), InstanceNormalization())

    def call(self, inputs, training=None, mask=None):
        x, add = inputs
        for i in range(1, 5):
            if add[i - 1] is not None:
                x = tf.add(x, add[i - 1])
            x = self.__getattribute__("conv" + str(i))(x)
            x = self.__getattribute__("norm" + str(i))(x)
            x = tf.nn.leaky_relu(x, self.args.leaky_alpha)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self, args, encoder):
        """
        :param Arg args:
        """
        super(Discriminator, self).__init__()
        self.args = args
        self.encoder = encoder
        self.dense_pr = tf.layers.Dense(1, "sigmoid")
        self.dense_cond = tf.layers.Dense(self.args.cond_dim, "sigmoid")

    @tf.contrib.eager.defun
    def call(self, inputs, training=None, mask=None):
        x = inputs
        encoder_layers = self.encoder(x)
        x = tf.layers.flatten(encoder_layers.pop())
        output_pr = self.dense_pr(x)
        output_cond = self.dense_cond(x)
        return output_pr, output_cond


class Generator(tf.keras.Model):
    def __init__(self, args, decoder):
        """
        :param Arg args:
        """
        super(Generator, self).__init__()
        self.args = args
        self.dense = tf.layers.Dense(self.args.init_dim ** 2 * self.args.conv_filter[0])
        self.norm = InstanceNormalization()
        self.decoder = decoder
        self.conv = tf.layers.Conv2DTranspose(self.args.image_channel, self.args.kernel_size, strides=(1, 1), padding="same", activation="tanh")

    @tf.contrib.eager.defun
    def call(self, inputs, training=None, mask=None):
        x = tf.concat(inputs, -1)
        x = self.dense(x)
        x = tf.nn.leaky_relu(x, self.args.leaky_alpha)
        x = tf.reshape(x, [-1, self.args.init_dim, self.args.init_dim, self.args.conv_filter[0]])
        x = self.norm(x)
        x = self.decoder([x, [None] * 4])
        output_image = self.conv(x)
        return output_image


class Adjuster(tf.keras.Model):
    def __init__(self, args, encoder, decoder):
        """
        :param Arg args:
        """
        super(Adjuster, self).__init__()
        self.args = args

        self.encoder = encoder
        self.dense = tf.layers.Dense(self.args.init_dim ** 2 * self.args.conv_filter[0])
        self.norm = InstanceNormalization()
        self.decoder = decoder
        self.conv = tf.layers.Conv2DTranspose(self.args.image_channel, self.args.kernel_size, strides=(1, 1), padding="same", activation="tanh")

    @tf.contrib.eager.defun
    def call(self, inputs, training=None, mask=None):
        image, cond = inputs
        encoder_layers = self.encoder(image)
        c = self.dense(cond)
        c = tf.nn.leaky_relu(c, alpha=self.args.leaky_alpha)
        c = self.norm(c)
        c = tf.reshape(c, [-1, self.args.init_dim, self.args.init_dim, self.args.conv_filter[0]])
        encoder_layers.reverse()
        x = self.decoder([c, encoder_layers])
        output_adj = self.conv(x)
        return output_adj


class Trainer:
    def __init__(self, args, generator, discriminator, adjuster, dataset):
        """
        :param Arg args:
        :param Generator generator:
        :param Discriminator discriminator:
        :param Adjuster adjuster:
        :param CelebA dataset:
        """
        self.args = args

        self.dataset = dataset
        self.adjuster = adjuster
        self.discriminator = discriminator
        self.generator = generator
        self.models = [self.discriminator, self.generator, self.adjuster]
        self._init_graph()
        self.generator_optimizer = tf.train.AdamOptimizer(args.lr)
        self.discriminator_optimizer = tf.train.AdamOptimizer(args.lr)
        self.adjuster_optimizer = tf.train.AdamOptimizer(args.lr)
        self.checkpoint = tf.train.Checkpoint(discriminator=self.discriminator, generator=self.generator, adjuster=self.adjuster,
                                              discriminator_optimizer=self.discriminator_optimizer, generator_optimizer=self.generator_optimizer,
                                              adjuster_optimizer=self.adjuster_optimizer)
        if path.isdir(path.join(self.args.result_dir, "checkpoint")) and self.args.restore:
            self.checkpoint.restore(tf.train.latest_checkpoint(path.join(self.args.result_dir, "checkpoint")))
        self.init_result_dir()
        self.writer = tf.contrib.summary.create_file_writer(path.join(self.args.result_dir, "log"))
        self.writer.set_as_default()

        self.part_groups = {
            "Generator": [range(0, 4), range(4, 8), range(8, 22)],
            "Discriminator": [range(0, 12), range(12, 16), range(16, 20)],
            "Adjuster": [range(16, 20), range(36, 38)]
        }
        self.part_weights = {}
        for model in self.models:
            name = model.__class__.__name__
            self.part_weights[name] = [[model.weights[layer] for layer in group] for group in self.part_groups[name]]

        self.all_weights = {
            "Generator": self.generator.weights,
            "Discriminator": self.discriminator.weights,
            "Adjuster": [self.adjuster.weights[w] for w in [16, 17, 18, 19, 36, 37]]
        }

        self.test_image, self.test_cond = self.dataset.iterator.get_next()
        self.test_noise = tf.random_normal([self.test_cond.shape[0], self.args.noise_dim])

    def _init_graph(self):

        real_image, real_cond = self.dataset.iterator.get_next()
        noise = tf.random_uniform([real_cond.shape[0], self.args.noise_dim])
        self.discriminator(real_image)
        self.generator([noise, real_cond])
        self.adjuster([real_image, real_cond])

    @staticmethod
    def discriminator_loss(real_true_c, real_predict_c, real_predict_pr, fake_predict_pr):
        return (tf.reduce_mean(tf.abs(real_true_c - real_predict_c)) * 2
                + tf.reduce_mean(tf.abs(0.98 - real_predict_pr))
                + tf.reduce_mean(tf.abs(0.02 - fake_predict_pr)))

    def generator_loss(self, cond_ori, cond_disc, pr_disc, image_ori, image_gen):
        return (tf.reduce_mean(tf.abs(0.98 - pr_disc))
                + tf.reduce_mean(tf.abs(cond_ori - cond_disc))
                + tf.reduce_mean(self.args.l1_lambda * tf.abs(image_ori - image_gen)))

    def adjuster_loss(self, cond_ori, cond_disc_real, pr_disc_real, cond_disc_fake, pr_disc_fake, image_ori, image_adj_real, image_adj_fake):
        fake = (tf.reduce_mean(tf.abs(0.98 - pr_disc_fake))
                + tf.reduce_mean(tf.abs(cond_ori - cond_disc_fake))
                + self.args.l1_lambda * tf.reduce_mean(tf.abs(image_ori - image_adj_fake)))
        real = (tf.reduce_mean(tf.abs(0.98 - pr_disc_real))
                + tf.reduce_mean(tf.abs(cond_ori - cond_disc_real))
                + self.args.l1_lambda * tf.reduce_mean(tf.abs(image_ori - image_adj_real)))
        return real + fake

    @staticmethod
    def gradient_penalty(real, fake, f):
        """
            with tf.GradientTape() as gp_tape:
                alpha = tf.random_uniform(shape=[real.shape[0], 1, 1, 1])
                inter = real + alpha * (fake - real)
                predict = f(inter)
            gradients = gp_tape.gradient(predict, f.weights)[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.abs(gradients), reduction_indices=range(1, inter.shape.ndims)))
            gp = tf.reduce_mean((slopes - 1.) ** 2)
            return gp
        """
        raise NotImplementedError("GP didn't implemented on eager mode")

    def _get_train_weight(self, model, batch_no):
        name = model.__class__.__name__
        if self.args.use_partition and batch_no % (self.args.partition_interval + 1) is 0:
            weights = self.part_weights[name]
            return weights[(batch_no // (self.args.partition_interval + 1)) % weights.__len__()]
        else:
            return self.all_weights[name]

    def _train_step(self, batch_no):
        real_image, real_cond = self.dataset.iterator.get_next()
        noise = tf.random_normal([real_cond.shape[0], self.args.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_image = self.generator([noise, real_cond])
            real_pr, real_c = self.discriminator(real_image)
            fake_pr, fake_c = self.discriminator(fake_image)

            disc_loss = self.discriminator_loss(real_cond, real_c, real_pr, fake_pr)
            gen_loss = self.generator_loss(real_cond, fake_c, fake_pr, real_image, fake_image)
            if self.args.use_gp:
                # todo: Is gp must use on cross-entropy
                # todo: explore how to gp on eager mode
                disc_gp = self.gradient_penalty(real_image, fake_image, self.discriminator)
                disc_loss = -disc_loss + disc_gp * self.args.gp_weight

        gradients_of_gen = gen_tape.gradient(gen_loss, self._get_train_weight(self.generator, batch_no))
        gradients_of_disc = disc_tape.gradient(disc_loss, self._get_train_weight(self.discriminator, batch_no))

        if self.args.use_clip:
            gradients_of_disc = [tf.clip_by_value(x, -self.args.clip_range, self.args.clip_range) for x in gradients_of_disc]

        self.generator_optimizer.apply_gradients(zip(gradients_of_gen, self._get_train_weight(self.generator, batch_no)))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_disc, self._get_train_weight(self.discriminator, batch_no)))
        if self.args.train_adj:
            adj_weight = self._get_train_weight(self.adjuster, batch_no)

            with tf.GradientTape() as adj_tape:
                adj_real_image = self.adjuster([real_image, real_cond])
                adj_fake_image = self.adjuster([fake_image, real_cond])
                adj_fake_pr, adj_fake_c = self.discriminator(adj_fake_image)
                adj_real_pr, adj_real_c = self.discriminator(adj_real_image)

                adj_loss = self.adjuster_loss(real_cond, adj_real_c, adj_real_pr, adj_fake_c, adj_fake_pr, real_image, adj_real_image, adj_fake_image)
            gradients_of_adj = adj_tape.gradient(adj_loss, adj_weight)
            self.adjuster_optimizer.apply_gradients(zip(gradients_of_adj, adj_weight))

            return fake_image, adj_real_image, adj_fake_image, gen_loss, disc_loss, adj_loss

        return fake_image, None, None, gen_loss, disc_loss, None

    def interrupted(self, signum, f_name):
        self.checkpoint.save(path.join(self.args.result_dir, "checkpoint", "interrupt"))
        print("\n Checkpoint has been saved")
        print(signum, f_name)
        import sys
        sys.exit(1)

    def train(self):
        loss_label = ["LossG", "LossD", "LossA"]
        import signal
        signal.signal(signal.SIGINT, self.interrupted)

        global_step = tf.train.get_or_create_global_step()

        for e in range(1, self.args.epoch):
            progress_bar = tf.keras.utils.Progbar(self.dataset.batches * self.args.batch_size)
            self.dataset.get_new_iterator()
            for b in range(1, self.dataset.batches + 1):

                try:
                    result = self._train_step(b)
                except tf.errors.OutOfRangeError:
                    break
                losses = result[3:6]
                global_step.assign_add(1)
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('loss/gen', losses[0])
                    tf.contrib.summary.scalar('loss/disc', losses[1])
                    if losses[2] is not None:
                        tf.contrib.summary.scalar('loss/adj', losses[2])
                # Terminal平均输出
                progress_add = []
                for label, loss in zip(loss_label, losses):
                    if loss is not None:
                        progress_add.append((label, loss))
                progress_bar.add(self.args.batch_size, progress_add)

                # 输出训练生成图像
                if b % self.args.freq_gen is 0:
                    gen_image = result[0]
                    save_image(gen_image, path.join(self.args.result_dir, "train", "gen", "%d-%d.jpg" % (e, b)))

                    if result[1] is not None and result[2] is not None:
                        adj_image = tf.concat(result[1:3], 0)
                        save_image(adj_image, path.join(self.args.result_dir, "train", "adj", "%d-%d.jpg" % (e, b)))
                if b % self.args.freq_test is 0:
                    gen_image = self.generator([self.test_noise, self.test_cond])
                    save_image(gen_image, path.join(self.args.result_dir, "test", "gen", "%d-%d.jpg" % (e, b)))
                    with open(path.join(self.args.result_dir, "test", "disc", "%d-%d.json" % (e, b)), "w") as f:
                        save = dict()
                        save["real_cond"] = self.test_cond
                        save["real_pr"], save["real_c"] = self.discriminator(self.test_image)
                        save["fake_pr"], save["fake_c"] = self.discriminator(gen_image)
                        for x in save:
                            save[x] = (tf.round(save[x] * 10)).numpy().astype(int).tolist()
                        json.dump(save, f)

                    if self.args.train_adj:
                        adj_real_image = self.adjuster([self.test_image, self.test_cond])
                        adj_fake_image = self.adjuster([gen_image, self.test_cond])
                        adj_image = tf.concat([adj_real_image, adj_fake_image], 0)
                        save_image(adj_image, path.join(self.args.result_dir, "test", "adj", "%d-%d.jpg" % (e, b)))

            self.checkpoint.save(path.join(self.args.result_dir, "checkpoint", str(e)))

    def init_result_dir(self):
        dirs = [".", "train/gen", "train/adj", "test/adj", "test/gen", "test/disc", "checkpoint", "log"]
        for item in dirs:
            if not path.exists(path.join(self.args.result_dir, item)):
                makedirs(path.join(self.args.result_dir, item))
        shutil.copyfile(self.args.env_file, path.join(self.args.result_dir, "config.json"))
        if not self.args.debug:
            repo = Repo(".")
            with open(path.join(self.args.result_dir, "code.tar"), "wb") as f:
                repo.archive(f)

    def plot(self):
        # todo: the shapes and the graph doesn't ok
        with open(self.args.result_dir + "/models.txt", "w") as f:
            def print_fn(content):
                print(content, file=f)

            model = [self.discriminator.encoder, self.generator.decoder, self.discriminator, self.generator]
            if self.args.train_adj:
                model.append(self.adjuster)
            for item in model:
                name = item.__class__.__name__
                pad_len = int(0.5 * (53 - name.__len__()))
                print_fn("=" * pad_len + "   Model: " + name + "  " + "=" * pad_len)
                item.summary(print_fn=print_fn)
                print_fn("\n")
                tf.keras.utils.plot_model(item, to_file=path.join(self.args.result_dir, "image", "%s.png" % name), show_shapes=True)
