import tensorflow as tf
from instance import InstanceNormalization

tfe = tf.contrib.eager


class Encoder(tf.keras.Model):
    def __init__(self, args):
        """
        :param FakeArg args:
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
        :param FakeArg args:
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
        :param FakeArg args:
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
        :param FakeArg args:
        """
        super(Generator, self).__init__()
        self.args = args
        self.dense = tf.layers.Dense(self.args.init_dim ** 2 * self.args.conv_filter[0])
        self.norm = InstanceNormalization()
        self.decoder = decoder
        self.conv = tf.layers.Conv2DTranspose(self.args.img_channel, self.args.kernel_size, strides=(1, 1), padding="same", activation="tanh")

    @tf.contrib.eager.defun
    def call(self, inputs, training=None, mask=None):
        x = tf.concat(inputs, -1)
        x = self.dense(x)
        x = tf.nn.leaky_relu(x, self.args.leaky_alpha)
        x = tf.reshape(x, [-1, self.args.init_dim, self.args.init_dim, self.args.conv_filter[0]])
        x = self.norm(x)
        x = self.decoder([x, [None] * 4])
        output_img = self.conv(x)
        return output_img


class Adjuster(tf.keras.Model):
    def __init__(self, args, encoder, decoder):
        """
        :param FakeArg args:
        """
        super(Adjuster, self).__init__()
        self.args = args

        self.encoder = encoder
        self.dense = tf.layers.Dense(self.args.init_dim ** 2 * self.args.conv_filter[0])
        self.norm = InstanceNormalization()
        self.decoder = decoder
        self.conv = tf.layers.Conv2DTranspose(self.args.img_channel, self.args.kernel_size, strides=(1, 1), padding="same", activation="tanh")

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

        :param FakeArg args:
        :param Generator generator:
        :param Discriminator discriminator:
        :param Adjuster adjuster:
        :param CelebA dataset:
        """
        self.adjuster = adjuster
        self.dataset = dataset
        self.discriminator = discriminator
        self.generator = generator
        self.args = args
        self.generator_optimizer = tf.train.AdamOptimizer(args.lr * 2)
        self.discriminator_optimizer = tf.train.AdamOptimizer(args.lr * 3)
        self.adjuster_optimizer = tf.train.AdamOptimizer(args.lr)
        self.checkpoint = tfe.Checkpoint(discriminator=self.discriminator, generator=self.generator, adjuster=self.adjuster,
                                         discriminator_optimizer=self.discriminator_optimizer, generator_optimizer=self.generator_optimizer,
                                         adjuster_optimizer=self.adjuster_optimizer)

    @staticmethod
    def discriminator_loss(real_true_c, real_predict_c, real_predict_pr, fake_predict_pr):
        return (tf.reduce_mean(tf.square(real_true_c - real_predict_c)) * 2
                + tf.reduce_mean(tf.square(0.98 - real_predict_pr))
                + tf.reduce_mean(tf.square(0.02 - fake_predict_pr)))

    def generator_loss(self, cond_ori, cond_disc, pr_disc, img_ori, img_gen):
        return (tf.reduce_mean(tf.square(0.98 - pr_disc))
                + tf.reduce_mean(tf.square(cond_ori - cond_disc))
                + tf.reduce_mean(self.args.l1_lambda * tf.abs(img_ori - img_gen)))

    def adjuster_loss(self, cond_ori, cond_disc_real, pr_disc_real, cond_disc_fake, pr_disc_fake, img_ori, img_adj_real, img_adj_fake):
        fake = (tf.reduce_mean(tf.square(0.98 - pr_disc_fake))
                + tf.reduce_mean(tf.square(cond_ori - cond_disc_fake))
                + self.args.l1_lambda * tf.reduce_mean(tf.abs(img_ori - img_adj_fake)))
        real = (tf.reduce_mean(tf.square(0.98 - pr_disc_real))
                + tf.reduce_mean(tf.square(cond_ori - cond_disc_real))
                + self.args.l1_lambda * tf.reduce_mean(tf.abs(img_ori - img_adj_real)))
        return real + fake

    def _train_step(self):
        real_img, real_cond = self.dataset.iterator.get_next()
        noise = tf.random_normal([self.args.batch_size, self.args.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as adj_tape:
            fake_img = self.generator([noise, real_cond])
            real_pr, real_c = self.discriminator(real_img)
            fake_pr, fake_c = self.discriminator(fake_img)

            # adj_real = self.adjuster([real_img, real_cond])
            # adj_fake = self.adjuster([fake_img, real_cond])
            # adj_fake_pr, adj_fake_c = self.discriminator(adj_fake)
            # adj_real_pr, adj_real_c = self.discriminator(adj_real)

            disc_loss = self.discriminator_loss(real_cond, real_c, real_pr, fake_pr)
            gen_loss = self.generator_loss(real_cond, fake_c, fake_pr, real_img, fake_img)
            # adj_loss = self.adjuster_loss(real_cond, adj_real_c, adj_real_pr, adj_fake_c, adj_fake_pr, real_img, adj_real, adj_fake)

        gradients_of_gen = gen_tape.gradient(gen_loss, self.generator.weights)
        gradients_of_disc = disc_tape.gradient(disc_loss, self.discriminator.weights)
        # gradients_of_adj = adj_tape.gradient(adj_loss, self.adjuster.weights)
        self.generator_optimizer.apply_gradients(zip(gradients_of_gen, self.generator.weights))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_disc, self.discriminator.weights))
