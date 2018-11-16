from os import path, system

import numpy as np
from git import Repo
from tensorflow.python.keras.utils import to_categorical, Progbar

# 引用
from config import args
from ourgan import OurGAN
from utils import CelebA, save_img, combine_images2

print("Application Params: ", args, "\r\n")
cond_dim = len(args.attr)
base_path = path.abspath("../SmileGAN-result/" + args.name)  # 路径（来源）
model = OurGAN(base_path, args)
# d、g、u的判断（是否存在）
d_file = path.abspath(base_path + "/model/D_weight.h5")
if path.isfile(d_file):
    print("Loading Exist Discriminator Weight")
    model.discriminator.load_weights(d_file)

g_file = path.abspath(base_path + "/model/G_weight.h5")
if path.isfile(g_file):
    print("Loading Exist Generator Weight")
    model.generator.load_weights(g_file)

u_file = path.abspath(base_path + "/model/U-Net_weight.h5")
if path.isfile(u_file):
    print("Loading Exist U-Net Weight")
    model.u_net.load_weights(u_file)
print("Using GPUs: ", args.gpu)

# old model
if args.mode == "plot":
    model.plot()  # 输出模型结构图
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
elif args.mode == "train":
    if args.attr_path is None or args.img_path.__len__() == 0:
        raise ValueError("params error!")
    data = CelebA(args)
    print("\r\nImage Flows From: ", args.img_path, "   Image Count: ", data.batch_size * data.batches)
    print("\r\nUsing Attribute: ", data.label)
    repo = Repo(".")
    if repo.is_dirty() and args.test == 0:  # 程序被修改且不是测试模式
        raise EnvironmentError("Git repo is Dirty! Please train after committed.")
    model.fit(data, args)
elif args.mode is "visual":  # loss etc的可视化
    print("The result path is " + path.abspath("../result/" + args.name))
    system("start tensorboard --host 0.0.0.0 --logdir " + path.abspath("../result/" + args.name + "/events"))
else:
    print("没有此模式：", args.mode)
