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
    def discriminator_loss(real_true_c, real_predict_c, real_predict_d, fake_true_c, fake_predict_c, fake_predict_d):
        return (tf.reduce_mean(tf.square(real_true_c - real_predict_c))
                + tf.reduce_mean(tf.square(fake_true_c - fake_predict_c))
                + tf.reduce_mean(tf.square(0.98 - real_predict_d))
                + tf.reduce_mean(tf.square(0.02 - fake_predict_d)))

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

    def adjuster_loss(self, cond_ori, cond_disc, pr_disc, img_ori, img_adj):
            return (tf.reduce_mean(tf.square(0.98 - pr_disc))
                    + tf.reduce_mean(tf.square(cond_ori - cond_disc))
                    + tf.reduce_mean(self.args.l1_lambda * tf.abs(img_ori - img_adj)))
