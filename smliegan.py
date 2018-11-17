import tensorflow as tf

from tensorflow.contrib.layers import instance_norm


class SmileGAN:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args
        self.pic, self.real_cond = dataset.iterator.get_next()
        self.p_z = tf.placeholder(tf.float32, [None, args.noise_dim])
        self.p_c = tf.placeholder(tf.float32, [None, args.cond_dim])

        self._setup_discriminator()
        self._setup_generator()

    def _setup_discriminator(self):
        x = self.pic

        for i in range(4):
            x = tf.layers.conv2d(x, self.args.conv_filter[3 - i], self.args.kernel_size, strides=(2, 2), padding="same")
            if self.args.norm == "instance":
                x = instance_norm(x)
            elif self.args.norm == "batch":
                x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x, alpha=self.args.leaky_alpha)
            x = tf.layers.dropout(x, self.args.dropout_rate)

        x = tf.layers.flatten(x)
        self.output_d = tf.layers.dense(x, 1, activation="sigmoid")
        self.output_c = tf.layers.dense(x, self.args.cond_dim, activation="sigmoid")

    def _setup_generator(self):
        x = tf.concat([self.p_z, self.p_c], 1)
        x = tf.layers.dense(x, self.args.init_dim ** 2 * self.args.conv_filter[0])
        x = tf.nn.leaky_relu(x, alpha=self.args.leaky_alpha)
        x = tf.reshape(x, [-1, self.args.init_dim, self.args.init_dim, self.args.conv_filter[0]])
        if self.args.norm == "instance":
            x = instance_norm(x)
        elif self.args.norm == "batch":
            x = tf.layers.batch_normalization(x)

        for i in range(1, 5):
            x = tf.layers.conv2d_transpose(x, self.args.conv_filter[i], self.args.kernel_size, strides=(2, 2), padding="same")
            if self.args.norm == "instance":
                x = instance_norm(x)
            elif self.args.norm == "batch":
                x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x, alpha=self.args.leaky_alpha)
        self.output_g = tf.layers.conv2d_transpose(x, self.args.img_channel, self.args.kernel_size, strides=(1, 1), padding="same", activation="tanh")

    def _setup_adjuster(self):
        pass
