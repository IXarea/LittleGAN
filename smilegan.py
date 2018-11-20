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
            output_pr = tf.layers.dense(x, 1, activation="sigmoid")
            output_c = tf.layers.dense(x, self.args.cond_dim, activation="sigmoid")
        return output_pr, output_c

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


class Trainer:
    def __init__(self, args, model, dataset):
        """
        :param FakeArg args:
        :param SmileGAN model:
        :param CelebA dataset:
        """
        self.generator_optimizer = tf.train.AdamOptimizer(args.lr * 2)
        self.discriminator_optimizer = tf.train.AdamOptimizer(args.lr * 3)
        self.adjuster_optimizer = tf.train.AdamOptimizer(args.lr)
        self.model = model
        self.dataset = dataset
        self.args = args

    def calc_loss(self):
        real_img, real_cond = self.dataset.iterator.get_next()
        noise = tf.random_normal([self.args.batch_size, self.args.noise_dim])
        fake_img = self.model.generator(noise, real_cond)
        real_pr, real_c = self.model.discriminator(real_img)
        fake_pr, fake_c = self.model.discriminator(fake_img)

        adj_real = self.model.adjuster(real_img, real_cond)
        adj_fake = self.model.adjuster(fake_img, real_cond)
        adj_fake_pr, adj_fake_c = self.model.discriminator(adj_fake)
        adj_real_pr, adj_real_c = self.model.discriminator(adj_real)

        disc_loss = tf.contrib.eager.gradients_function(
            self.model.discriminator_loss
        )(real_cond, real_c, real_pr, fake_pr)
        gen_loss = tf.contrib.eager.gradients_function(
            self.model.generator_loss
        )(real_cond, fake_c, fake_pr, real_img, fake_img)
        adj_loss = tf.contrib.eager.gradients_function(
            self.model.adjuster_loss
        )(real_cond, adj_real_c, adj_real_pr, adj_fake_c, adj_fake_pr, real_img, adj_real, adj_fake)
        return disc_loss, gen_loss, adj_loss




