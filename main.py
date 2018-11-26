from config import Arg

args = Arg()
import tensorflow as tf

tf.enable_eager_execution()

from os import path, system

from dataset import CelebA
from littlegan import Trainer, Adjuster, Discriminator, Decoder, Encoder, Generator

from git import Repo

dec = Decoder(args)
enc = Encoder(args)
gen = Generator(args, dec)
disc = Discriminator(args, enc)
adj = Adjuster(args, enc, dec)
data = CelebA(args)
model = Trainer(args, gen, disc, adj, data)

print("Application Params: ", args, "\r\n")
cond_dim = len(args.attr)

print("Using GPUs: ", args.gpu)

if args.mode == "train":
    if args.attr_path is None or args.image_path.__len__() == 0:
        raise ValueError("params error!")
    data = CelebA(args)
    print("\r\nImage Flows From: ", args.image_path, "   Image Count: ", args.batch_size * data.batches)
    print("\r\nUsing Attribute: ", data.label)
    repo = Repo(".")
    if repo.is_dirty() and not args.debug:  # 程序被修改且不是测试模式
        raise EnvironmentError("Git repo is Dirty! Please train after committed.")
    model.train()
elif args.mode == "visual":  # loss etc的可视化
    print("The result path is ", path.join(args.result_dir, "log"))
    system("tensorboard --host 0.0.0.0 --logdir " + path.join(args.result_dir, "log"))
elif args.mode == "plot":
    model.plot()  # 输出模型结构图
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
