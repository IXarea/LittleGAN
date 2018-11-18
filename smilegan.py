import tensorflow as tf

from tensorflow.contrib.layers import instance_norm


class SmileGAN:
    def __init__(self, args):
        self.args = args
        print(self.args.norm)
        if self.args.norm == "instance":
            self.norm = instance_norm
        elif self.args.norm == "batch":
            self.norm = tf.layers.batch_normalization
        else:
            raise NotImplementedError("Normalization Layer Not Implemented")

    def encoder(self, image):
        with tf.name_scope("Encoder"):
            layers = [image]
            for i in range(4):
                x = tf.layers.conv2d(layers[i], self.args.conv_filter[3 - i], self.args.kernel_size, strides=(2, 2), padding="same")
                x = self.norm(x)
                x = tf.nn.leaky_relu(x, alpha=self.args.leaky_alpha)
                layers[i + 1] = tf.layers.dropout(x, self.args.dropout_rate)
        return layers

    def decoder(self, x):
        with tf.name_scope("Decoder"):
            for i in range(1, 5):
                x = tf.layers.conv2d_transpose(x, self.args.conv_filter[i], self.args.kernel_size, strides=(2, 2), padding="same")
                x = self.norm(x)
                x = tf.nn.leaky_relu(x, alpha=self.args.leaky_alpha)
        return x

    def discriminator(self, image):
        with tf.name_scope("Discriminator"):
            x = self.encoder(image)

            x = tf.layers.flatten(x)
            output_d = tf.layers.dense(x, 1, activation="sigmoid")
            output_c = tf.layers.dense(x, self.args.cond_dim, activation="sigmoid")
        return output_d, output_c

    def generator(self, noise, cond):
        with tf.name_scope("Generator"):
            x = tf.concat([noise, cond], 1)
            x = tf.layers.dense(x, self.args.init_dim ** 2 * self.args.conv_filter[0])
            x = tf.nn.leaky_relu(x, alpha=self.args.leaky_alpha)

            x = tf.reshape(x, [-1, self.args.init_dim, self.args.init_dim, self.args.conv_filter[0]])
            x = self.norm(x)

            x = self.decoder(x)

            output_g = tf.layers.conv2d_transpose(x, self.args.img_channel, self.args.kernel_size, strides=(1, 1), padding="same", activation="tanh")
        return output_g

    def adjuster(self, image, cond):
        with tf.name_scope("Adjuster"):
            x = self.encoder(image)

            c = tf.layers.dense(cond, self.args.init_dim ** 2 * self.args.conv_filter[0])
            c = tf.nn.leaky_relu(c, alpha=self.args.leaky_alpha)
            c = self.norm(c)
            c = tf.reshape(c, [-1, self.args.init_dim, self.args.init_dim, self.args.conv_filter[0]])
            x = tf.add(c, x)
            x = self.decoder(x)
        return x
