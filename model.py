import tensorflow as tf
from instance import InstanceNormalization
from tensorflow import keras


class Encoder(keras.Model):
    def __init__(self, args):
        """
        :param Arg args:
        """
        super(Encoder, self).__init__()
        self.args = args
        for i in range(1, 5):
            self.__setattr__("conv" + str(i),
                             tf.compat.v1.layers.Conv2D(self.args.conv_filter[4 - i], self.args.kernel_size, 2, "same"))
            self.__setattr__("norm" + str(i), InstanceNormalization())

    def call(self, inputs, training=None, mask=None):
        x = inputs
        outputs = []
        for i in range(1, 5):
            x = self.__getattribute__("conv" + str(i))(x)
            x = self.__getattribute__("norm" + str(i))(x)
            x = tf.nn.leaky_relu(x, self.args.leaky_alpha)
            x = tf.compat.v1.layers.dropout(x, self.args.dropout_rate)
            outputs.append(x)
        return outputs


class Decoder(keras.Model):
    def __init__(self, args):
        """
        :param Arg args:
        """
        super(Decoder, self).__init__()
        self.args = args
        for i in range(1, 5):
            self.__setattr__("conv" + str(i),
                             tf.compat.v1.layers.Conv2DTranspose(self.args.conv_filter[i], self.args.kernel_size,
                                                                 (2, 2), "same"))
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


class Discriminator(keras.Model):
    def __init__(self, args, encoder):
        """
        :param Arg args:
        """
        super(Discriminator, self).__init__()
        self.args = args
        self.encoder = encoder
        self.dense_pr = tf.compat.v1.layers.Dense(1, "sigmoid")
        self.dense_cond = tf.compat.v1.layers.Dense(self.args.cond_dim, "sigmoid")

    @tf.contrib.eager.defun
    def call(self, inputs, training=None, mask=None):
        # Todo: try to discriminate the feature map
        x = inputs
        encoder_layers = self.encoder(x)
        x = tf.compat.v1.layers.flatten(encoder_layers.pop())
        output_pr = self.dense_pr(x)
        output_cond = self.dense_cond(x)
        return output_pr, output_cond


class Generator(keras.Model):
    def __init__(self, args, decoder):
        """
        :param Arg args:
        """
        super(Generator, self).__init__()
        self.args = args
        self.dense = tf.compat.v1.layers.Dense(self.args.init_dim ** 2 * self.args.conv_filter[0])
        self.norm = InstanceNormalization()
        self.decoder = decoder
        self.conv = tf.compat.v1.layers.Conv2DTranspose(self.args.image_channel, self.args.kernel_size, strides=(1, 1),
                                                        padding="same", activation="tanh")

    @tf.contrib.eager.defun
    def call(self, inputs, training=None, mask=None):
        """
        生成器
        :param inputs: [noise, real_cond]
        :param training:
        :param mask:
        :return:
        """
        x = tf.concat(inputs, -1)
        x = self.dense(x)
        x = tf.nn.leaky_relu(x, self.args.leaky_alpha)
        x = tf.reshape(x, [-1, self.args.init_dim, self.args.init_dim, self.args.conv_filter[0]])
        x = self.norm(x)
        x = self.decoder([x, [None] * 4])
        output_image = self.conv(x)
        return output_image


class Adjuster(keras.Model):
    def __init__(self, args, discriminator, generator):
        """

        :param Arg args:
        :param Discriminator discriminator:
        :param Generator generator:
        """
        super(Adjuster, self).__init__()
        self.args = args

        self.encoder = discriminator.encoder
        self.dense = tf.compat.v1.layers.Dense(self.args.init_dim ** 2 * self.args.conv_filter[0])
        self.norm = InstanceNormalization()
        self.decoder = generator.decoder
        self.conv = generator.conv

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
