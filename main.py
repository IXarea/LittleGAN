from config import Arg

args = Arg()
import tensorflow as tf

tf.enable_eager_execution()

from os import path, system
from dataset import CelebA
from utils import save_image
from littlegan import Trainer, Adjuster, Discriminator, Decoder, Encoder, Generator
from git import Repo
import time
import numpy as np

decoder = Decoder(args)
encoder = Encoder(args)
generator = Generator(args, decoder)
discriminator = Discriminator(args, encoder)
adjuster = Adjuster(args, encoder, decoder)
# data = CelebA(args)
# model = Trainer(args, gen, disc, adj, data)

print("Application Params: ", args, "\r\n")
cond_dim = len(args.attr)

print("Using GPUs: ", args.gpu)

data = CelebA(args)
print("\r\nImage Flows From: ", args.image_path, "   Image Count: ", args.batch_size * data.batches)
print("\r\nUsing Attribute: ", data.label)

model = Trainer(args, generator, discriminator, adjuster, data)

if args.mode == "train":
    repo = Repo(".")
    if repo.is_dirty() and not args.debug:  # 程序被修改且不是测试模式
        raise EnvironmentError("Git repo is Dirty! Please train after committed.")
    model.train()
elif args.mode == "visual":  # loss etc的可视化
    print("The result path is ", path.join(args.result_dir, "log"))
    system("tensorboard --host 0.0.0.0 --logdir " + path.join(args.result_dir, "log"))
elif args.mode == "plot":
    model.plot()  # 输出模型结构图
elif args.mode == "random-sample":
    iterator = data.get_new_iterator()
    now_time = int(time.time())
    for b in range(args.random_sample_batch):
        image, cond = iterator.get_next()
        noise = tf.random_uniform([cond.shape[0], args.noise_dim])

        model.predict(noise, cond, image,
                      path.join(args.result_dir, "sample", "generator-%s-%d.jpg" % (now_time, b)),
                      path.join(args.result_dir, "sample", "discriminator-%s-%d.json" % (now_time, b)),
                      path.join(args.result_dir, "sample", "adjuster-%s-%d.jpg" % (now_time, b))
                      )
        np.savez_compressed(path.join(args.result_dir, "sample", "input_data-%s-%d.npz" % (now_time, b)), n=noise, c=cond, i=image)
elif args.mode == "evaluate":
    iterator = data.get_new_iterator()
    progress = tf.keras.utils.Progbar(args.evaluate_sample_batch * args.batch_size)
    for b in range(args.evaluate_sample_batch):
        base_index = b * args.batch_size + 1
        image, cond = iterator.get_next()
        noise = tf.random_uniform([cond.shape[0], args.noise_dim])
        gen_image, save, adj_real_image, adj_fake_image = model.predict(noise, cond, image,
                                                                        None, path.join(args.result_dir, "evaluate", "discriminator.json"), None)
        for i in range(args.batch_size):
            save_image(gen_image[i], path.join(args.result_dir, "evaluate", "gen", str(base_index + i) + ".jpg"))
            if adj_real_image is not None and adj_fake_image is not None:
                save_image(adj_real_image[i], path.join(args.result_dir, "evaluate", "adj", "real_" + str(base_index + i) + ".jpg"))
                save_image(adj_fake_image[i], path.join(args.result_dir, "evaluate", "adj", "fake_" + str(base_index + i) + ".jpg"))
        progress.add(args.batch_size)

    if not args.gpu:
        args.gpu = [-1]

    gen_cmd = "python evaluate.py calc %s %s %s %s --gpu %s" % (
        path.join(args.result_dir, "evaluate", "gen"),
        path.join(args.test_data_dir, args.evaluate_pre_calculated),
        args.test_data_dir,
        ",".join(map(str, args.gpu)),
        path.join(args.result_dir, "evaluate", "fid-gen.log")
    )

    print("Running: \"", gen_cmd, "\"")
    system(gen_cmd)
    if args.train_adj:
        adj_cmd = "python evaluate.py calc %s %s %s %s --gpu %s" % (
            path.join(args.result_dir, "evaluate", "adj"),
            path.join(args.test_data_dir, args.evaluate_pre_calculated),
            args.test_data_dir,
            ",".join(map(str, args.gpu)),
            path.join(args.result_dir, "evaluate", "fid-adj.log")
        )
        print("Running: \"", adj_cmd, "\"")
        system(adj_cmd)


else:
    print("没有此模式：", args.mode)

"""
elif args.mode == "predict":  # elif=else if
    cond = to_categorical(range(cond_dim), cond_dim) * 1.65 - 0.7  # 规则生成属性，为防止属性过大或过小*1.65-0.7
    cond = np.tile(cond, (cond_dim, 1))
    noise = np.random.normal(size=[cond_dim, args.noise])  # 随机噪音
    noise = np.repeat(noise, cond_dim, 0)
    save_img(model.predict(cond, noise))  # 根据属性与随机噪音生成图像
elif args.mode == "manual-predict":  # 手动输入属性
    print("请输入 ", cond_dim, " 个属性(用空格隔开): ")

    cond = [float(x) for x in input().split(" ")]
    if len(cond) == cond_dim:
        save_img(model.predict(np.array([cond])))
        # new（now used）
elif args.mode == "predict2":
    bar = Progbar(100)
    for i in range(1, 101):  # 循环100次
        cond = to_categorical(range(cond_dim), cond_dim) * 0.88 + 0.02
        noise = np.random.normal(size=[1, args.noise])
        noise = np.repeat(noise, cond_dim, 0)
        img = model.predict(cond, noise)
        img2 = img[[x for x in range(7) if x % 7 in [0, 3, 4, 5]]]
        save_img(combine_images2(img2, 1, 4), path.join(base_path, "g-%d.png" % i))
        bar.add(1)

"""
