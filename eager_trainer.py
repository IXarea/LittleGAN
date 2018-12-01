from os import path, makedirs
from utils import save_image, soft
import json
import shutil
import numpy as np
from git import Repo
import tensorflow as tf
import time


class EagerTrainer:
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
        self.generator_optimizer = tf.train.AdamOptimizer(args.lr, args.beta_1, args.beta_2)
        self.discriminator_optimizer = tf.train.AdamOptimizer(args.lr, args.beta_1, args.beta_2)
        self.adjuster_optimizer = tf.train.AdamOptimizer(args.lr)
        self.checkpoint = tf.train.Checkpoint(discriminator=self.discriminator, generator=self.generator, adjuster=self.adjuster,
                                              discriminator_optimizer=self.discriminator_optimizer, generator_optimizer=self.generator_optimizer,
                                              adjuster_optimizer=self.adjuster_optimizer)
        if path.isdir(path.join(self.args.result_dir, "checkpoint")) and self.args.restore:
            print("Loading Checkpoint...")
            self.checkpoint.restore(tf.train.latest_checkpoint(path.join(self.args.result_dir, "checkpoint")))
        self.init_dir()
        self.writer = tf.contrib.summary.create_file_writer(path.join(self.args.result_dir, "log"))
        self.writer.set_as_default()

        self.part_groups = {
            "Generator": [range(0, 4), range(4, 8), range(8, 22)],
            "Discriminator": [range(0, 12), range(12, 16), range(16, 20)],
            "Adjuster": [range(16, 20), range(36, 38)]
        }
        # weights可指kernel size/k/b等参数
        self.part_weights = {}
        for model in self.models:
            name = model.__class__.__name__
            self.part_weights[name] = [[model.weights[layer] for layer in group] for group in self.part_groups[name]]

        self.all_weights = {
            "Generator": self.generator.weights,
            "Discriminator": self.discriminator.weights,
            "Adjuster": [self.adjuster.weights[w] for w in [16, 17, 18, 19, 36, 37]]
        }

    def _init_graph(self):
        iterator, self.test_noise, self.test_cond, self.test_image = None, None, None, None
        npz_file = path.join(self.args.test_data_dir, "test_data_" + str(self.args.env) + ".npz")
        if path.isfile(npz_file) and self.args.reuse:
            data = np.load(npz_file)
            self.test_noise, self.test_cond, self.test_image = data["n"], data["c"], data["i"]
        while True:
            try:
                self.discriminator(self.test_image)
                self.generator([self.test_noise, self.test_cond])
                self.adjuster([self.test_image, self.test_cond])
                break
            except (tf.errors.InvalidArgumentError, AttributeError):
                print("No reuse test data, generating")
                if None is iterator:
                    iterator = self.dataset.get_new_iterator()
                self.test_image, self.test_cond = iterator.get_next()
                self.test_noise = tf.random_normal([self.test_cond.shape[0], self.args.noise_dim])
                np.savez_compressed(npz_file, n=self.test_noise, c=self.test_cond, i=self.test_image)

    @staticmethod
    def discriminator_loss(real_true_c, real_predict_c, real_predict_pr, fake_predict_pr):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_true_c, real_predict_c)) * 2 \
               + tf.reduce_mean(tf.keras.losses.binary_crossentropy(soft(tf.ones(real_predict_pr.shape)), real_predict_pr)) \
               + tf.reduce_mean(tf.keras.losses.binary_crossentropy(soft(tf.zeros(fake_predict_pr.shape)), fake_predict_pr))

    def generator_loss(self, cond_ori, cond_disc, pr_disc, image_ori, image_gen):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(soft(tf.ones(pr_disc.shape)), pr_disc)) \
               + tf.reduce_mean(tf.keras.losses.binary_crossentropy(cond_ori, cond_disc)) \
               + self.args.l1_lambda * tf.reduce_mean(tf.abs(image_ori - image_gen))

    def adjuster_loss(self, cond_ori, cond_disc, pr_disc, image_ori, image_adj):
        cond2 = tf.concat([cond_ori, cond_ori], axis=0)
        image2 = tf.concat([image_ori, image_ori], axis=0)
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(soft(tf.ones(pr_disc.shape)), pr_disc)) \
               + tf.reduce_mean(tf.keras.losses.binary_crossentropy(cond2, cond_disc)) \
               + self.args.l1_lambda * tf.reduce_mean(tf.abs(image2 - image_adj))

    def _get_train_weight(self, model, batch_no):
        name = model.__class__.__name__
        # 如果用到partition and 批次数除以（partition间隔加一）的值为0
        if self.args.use_partition and batch_no % (self.args.partition_interval + 1) is 0:
            weights = self.part_weights[name]
            # 【批次数整除（向下）（partition间隔加一）在除以每组的组数】=进行partition的组别编号
            # 这里即第几组的权重
            return weights[(batch_no // (self.args.partition_interval + 1)) % weights.__len__()]
        else:
            return self.all_weights[name]

    def _train_step(self, batch_no, real_image, real_cond):
        # Todo: why use uniform distribution as noise will cause mode collapse
        noise = tf.random_normal([self.args.batch_size, self.args.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_image = self.generator([noise, real_cond])
            real_pr, real_c = self.discriminator(real_image)
            fake_pr, fake_c = self.discriminator(fake_image)

            disc_loss = self.discriminator_loss(real_cond, real_c, real_pr, fake_pr)
            gen_loss = self.generator_loss(real_cond, fake_c, fake_pr, real_image, fake_image)
            if self.args.use_gp:
                # todo: Is gp must use on cross-entropy
                # todo: explore how to gp on eager mode
                raise NotImplementedError("GP didn't implemented on eager mode")

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
                adj_image = tf.concat([adj_real_image, adj_fake_image], axis=0)
                adj_pr, adj_c = self.discriminator(adj_image)
                adj_loss = self.adjuster_loss(real_cond, adj_c, adj_pr, real_image, adj_image)
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

        for e in range(1, self.args.epoch + 1):
            print("Exp:", self.args.exp_name, "Epoch:", e, "Starting...")
            iterator = self.dataset.get_new_iterator()
            progress_bar = tf.keras.utils.Progbar(self.dataset.batches * self.args.batch_size)
            start_time = time.time()
            for b in range(1, self.dataset.batches + 1):
                try:
                    real_image, real_cond = iterator.get_next()
                except tf.errors.OutOfRangeError:
                    break
                if real_cond.shape[0] != self.args.batch_size:
                    print("Skip one batch")
                    continue
                result = self._train_step(b, real_image, real_cond)
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
                    self.predict(self.test_noise, self.test_cond, self.test_image,
                                 path.join(self.args.result_dir, "test", "gen", "%d-%d.jpg" % (e, b)),
                                 path.join(self.args.result_dir, "test", "disc", "%d-%d.json" % (e, b)),
                                 path.join(self.args.result_dir, "test", "adj", "%d-%d.jpg" % (e, b))
                                 )
            end_time = time.time()
            print("Time usage:", start_time - end_time, "s")
            self.checkpoint.save(path.join(self.args.result_dir, "checkpoint", str(e)))

    def init_dir(self):
        if not path.exists(self.args.test_data_dir):
            makedirs(self.args.result_dir)
        dirs = [".", "train/gen", "train/adj", "test/adj", "test/gen", "test/disc", "checkpoint", "log", "sample", "evaluate/gen", "evaluate/adj",
                "evaluate/disc"]
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

    def predict(self, noise, cond, image, gen_image_save_path=None, json_save_path=None, adj_image_save_path=None):
        start_time = time.time()
        gen_image = self.generator([noise, cond])
        end_time = time.time()
        print("Generate Time", end_time - start_time, "s")
        if None is not gen_image_save_path:
            save_image(gen_image, gen_image_save_path)

        save = dict()
        save["real_cond"] = cond
        save["real_pr"], save["real_c"] = self.discriminator(image)
        save["fake_pr"], save["fake_c"] = self.discriminator(gen_image)
        save["real_pr_mse"] = tf.reduce_mean(tf.keras.metrics.mean_squared_error(soft(1), save["real_pr"]), axis=0).numpy().astype(float)
        save["real_c_mse"] = tf.reduce_mean(tf.keras.metrics.mean_squared_error(cond, save["real_c"]), axis=0).numpy().astype(float)
        save["fake_pr_mse"] = tf.reduce_mean(tf.keras.metrics.mean_squared_error(soft(0), save["fake_pr"]), axis=0).numpy().astype(float)
        save["fake_c_mse"] = tf.reduce_mean(tf.keras.metrics.mean_squared_error(cond, save["fake_c"]), axis=0).numpy().astype(float)
        for x in ["real_cond", "real_pr", "real_c", "fake_c", "fake_pr"]:
            save[x] = (tf.round(save[x] * 100)).numpy().astype(int).tolist()
        if None is not json_save_path:
            with open(json_save_path, "w") as f:
                json.dump(save, f)
        adj_fake_image, adj_real_image = None, None
        if self.args.train_adj:
            adj_real_image = self.adjuster([image, cond])
            adj_fake_image = self.adjuster([gen_image, cond])
            adj_image = tf.concat([adj_real_image, adj_fake_image], 0)
            if None is not adj_image_save_path:
                save_image(adj_image, adj_image_save_path)

        return gen_image, save, adj_real_image, adj_fake_image
