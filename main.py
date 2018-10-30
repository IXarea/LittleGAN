from config import args
import os
import keras
import numpy as np
from git import Repo
from ourgan import OurGAN
from utils import CelebA

attr = [8, 15, 20, 22, 26, 36, 39]
cond_dim = len(attr)
channels = 3
print("\r\nApplication Params: ", args, "\r\n")
model = OurGAN(args.noise, cond_dim, args.img_size, channels, os.path.abspath("../result/" + args.name))
if args.plot == 1:
    model.plot()
g_file = os.path.abspath("../result/" + args.name + "/model/G_weight.h5")
if os.path.isfile(g_file):
    print("\r\n Loading Exist Generator Weight")
    model.generator.load_weights(g_file)
d_file = os.path.abspath("../result/" + args.name + "/model/D_weight.h5")
if os.path.isfile(d_file):
    print("\r\n Loading Exist Discriminator Weight")
    model.discriminator.load_weights(d_file)
u_file = os.path.abspath("../result/" + args.name + "/model/U-Net_weight.h5")
if os.path.isfile(u_file):
    print("\r\n Loading Exist U-Net Weight")
    model.u_net.load_weights(u_file)
print("\r\nUsing GPUs: ", args.gpu, "\r\n")
if args.attr_path is None or args.img_path.__len__() is 0:
    if args.mode == "train" or args.mode == "predict":
        raise ValueError("params error!")
    elif args.mode == "manual-predict":
        print("请输入 ", cond_dim, " 个属性(用空格隔开): ")
        cond = [float(x) for x in input().split(" ")]
        if len(cond) == cond_dim:
            model.predict(np.array([cond]))
else:
    data = CelebA(args.img_path, args.attr_path, args.batch_size, (args.img_size, args.img_size, channels),
                  args.img_ext, attr)
    print("\r\nImage Flows From: ", args.img_path, "   Image Count: ", data.batch_size * data.batches, "\r\n")
    print("\r\nUsing Attribute: ", data.label, "\r\n")
    if args.mode == "train":

        repo = Repo(".")
        if repo.is_dirty() and args.test == 0:
            raise EnvironmentError("Git repo is Dirty! Please train after committed.")

        model.fit(args.batch_size, args.epoch, data, args.model_freq_batch, args.model_freq_epoch,
                  args.img_freq, args.start, args.gpu)
    elif args.mode == "predict":
        cond = keras.utils.to_categorical(range(cond_dim), cond_dim) * 1.65 - 0.7
        cond = np.tile(cond, (cond_dim, 1))
        noise = np.random.normal(size=[cond_dim, args.noise])
        noise = np.repeat(noise, cond_dim, 0)
        model.predict(cond, noise)
