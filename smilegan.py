import tensorflow as tf

from tensorflow.contrib.layers import instance_norm


class SmileGAN:
    def __init__(self, args):
        self.args = args
        if self.args.norm == "instance":
            self.norm = instance_norm
        elif self.args.norm == "batch":
            self.norm = tf.layers.batch_normalization
        else:
            raise NotImplementedError("Normalization Layer Not Implemented")

    def encoder(self, image):
        with tf.name_scope("Encoder"):
            layers = [image]
            for i in range(1, 5):
                x = tf.layers.conv2d(layers[i - 1], self.args.conv_filter[4 - i], self.args.kernel_size, strides=(2, 2), padding="same")
                x = self.norm(x)
                x = tf.nn.leaky_relu(x, alpha=self.args.leaky_alpha)
                layers.append(tf.layers.dropout(x, self.args.dropout_rate))
            return layers

    def decoder(self, x, add):
        with tf.name_scope("Decoder"):
            for i in range(1, 5):
                if add[i - 1] is not None:
                    x = tf.add(x, add[i - 1])
                x = tf.layers.conv2d_transpose(x, self.args.conv_filter[i], self.args.kernel_size, strides=(2, 2), padding="same")
                x = self.norm(x)
                x = tf.nn.leaky_relu(x, alpha=self.args.leaky_alpha)
        return x

    def discriminator(self, image):
        with tf.name_scope("Discriminator"):
            encoder_layers = self.encoder(image)

            x = tf.layers.flatten(encoder_layers[4])
            output_d = tf.layers.dense(x, 1, activation="sigmoid")
            output_c = tf.layers.dense(x, self.args.cond_dim, activation="sigmoid")
        return output_d, output_c

    @staticmethod
    def discriminator_loss(real_true_c, real_predict_c, real_predict_pr, fake_predict_pr):
        return (tf.reduce_mean(tf.square(real_true_c - real_predict_c)) * 2
                + tf.reduce_mean(tf.square(0.98 - real_predict_pr))
                + tf.reduce_mean(tf.square(0.02 - fake_predict_pr)))

    def generator(self, noise, cond):
        with tf.name_scope("Generator"):
            x = tf.concat([noise, cond], 1)
            x = tf.layers.dense(x, self.args.init_dim ** 2 * self.args.conv_filter[0])
            x = tf.nn.leaky_relu(x, alpha=self.args.leaky_alpha)

            x = tf.reshape(x, [-1, self.args.init_dim, self.args.init_dim, self.args.conv_filter[0]])
            x = self.norm(x)

            x = self.decoder(x, [None] * 4)

            output_g = tf.layers.conv2d_transpose(x, self.args.img_channel, self.args.kernel_size, strides=(1, 1), padding="same", activation="tanh")
        return output_g

    def generator_loss(self, cond_ori, cond_disc, pr_disc, img_ori, img_gen):
        return (tf.reduce_mean(tf.square(0.98 - pr_disc))
                + tf.reduce_mean(tf.square(cond_ori - cond_disc))
                + tf.reduce_mean(self.args.l1_lambda * tf.abs(img_ori - img_gen)))

    def adjuster(self, image, cond):
        with tf.name_scope("Adjuster"):
            encoder_layers = self.encoder(image)

            c = tf.layers.dense(cond, self.args.init_dim ** 2 * self.args.conv_filter[0])
            c = tf.nn.leaky_relu(c, alpha=self.args.leaky_alpha)
            c = self.norm(c)
            c = tf.reshape(c, [-1, self.args.init_dim, self.args.init_dim, self.args.conv_filter[0]])
            encoder_layers.reverse()
            encoder_layers.pop()
            x = self.decoder(c, encoder_layers)
            output_adj = tf.layers.conv2d_transpose(x, self.args.img_channel, self.args.kernel_size, strides=(1, 1), padding="same", activation="tanh")
        return output_adj

    def adjuster_loss(self, cond_ori, cond_disc_real, pr_disc_real, cond_disc_fake, pr_disc_fake, img_ori, img_adj_real, img_adj_fake):
        fake = (tf.reduce_mean(tf.square(0.98 - pr_disc_fake))
                + tf.reduce_mean(tf.square(cond_ori - cond_disc_fake))
                + self.args.l1_lambda * tf.reduce_mean(tf.abs(img_ori - img_adj_fake)))
        real = (tf.reduce_mean(tf.square(0.98 - pr_disc_real))
                + tf.reduce_mean(tf.square(cond_ori - cond_disc_real))
                + self.args.l1_lambda * tf.reduce_mean(tf.abs(img_ori - img_adj_real)))
        return real + fake


class Encoder(tf.keras.Model):
    def __init__(self, args):
        """
        :param FakeArg args:
        """
        super(Encoder, self).__init__()
        self.args = args
        for i in range(1, 5):
            self.__setattr__("conv" + str(i), tf.layers.Conv2D(self.args.conv_filter[4 - i], self.args.kernel_size, 2, "same"))
            # change to instance
            self.__setattr__("norm" + str(i), tf.layers.BatchNormalization())
            self.__setattr__("relu" + str(i), tf.keras.layers.LeakyReLU(self.args.leaky_alpha))
            self.__setattr__("drop" + str(i), tf.layers.Dropout(self.args.dropout_rate))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        outputs = []
        for i in range(1, 5):
            x = self.__getattribute__("conv" + str(i))(x)
            x = self.__getattribute__("norm" + str(i))(x)
            x = self.__getattribute__("relu" + str(i))(x)
            x = self.__getattribute__("drop" + str(i))(x)
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
            # change to instance
            self.__setattr__("norm" + str(i), tf.layers.BatchNormalization())
            self.__setattr__("relu" + str(i), tf.keras.layers.LeakyReLU(self.args.leaky_alpha))

    def call(self, inputs, training=None, mask=None):
        x, add = inputs
        for i in range(1, 5):
            if add[i - 1] is not None:
                x = tf.add(x, add[i - 1])
            x = self.__getattribute__("conv" + str(i))(x)
            x = self.__getattribute__("norm" + str(i))(x)
            x = self.__getattribute__("relu" + str(i))(x)
        return x
