from config import args
from os import path
from keras.utils import to_categorical
import numpy as np
from git import Repo
from ourgan import OurGAN
from utils import CelebA

cond_dim = len(args.attr)
print("Application Params: ", args, "\r\n")
base_path = path.abspath("../result/" + args.name)
model = OurGAN(base_path, args)
g_file = path.abspath(base_path + "/model/G_weight.h5")
if path.isfile(g_file):
    print("Loading Exist Generator Weight")
    model.generator.load_weights(g_file)
d_file = path.abspath(base_path + "/model/D_weight.h5")
if path.isfile(d_file):
    print("Loading Exist Discriminator Weight")
    model.discriminator.load_weights(d_file)
u_file = path.abspath(base_path + "/model/U-Net_weight.h5")
if path.isfile(u_file):
    print("Loading Exist U-Net Weight")
    model.u_net.load_weights(u_file)
print("Using GPUs: ", args.gpu)

if args.mode is "plot":
    model.plot()
elif args.mode is "predict":
    cond = to_categorical(range(cond_dim), cond_dim) * 1.65 - 0.7
    cond = np.tile(cond, (cond_dim, 1))
    noise = np.random.normal(size=[cond_dim, args.noise])
    noise = np.repeat(noise, cond_dim, 0)
    model.predict(cond, noise)
elif args.mode is "manual-predict":
    print("请输入 ", cond_dim, " 个属性(用空格隔开): ")
    cond = [float(x) for x in input().split(" ")]
    if len(cond) == cond_dim:
        model.predict(np.array([cond]))
elif args.mode is "train":
    if args.attr_path is None or args.img_path.__len__() is 0:
        raise ValueError("params error!")
    data = CelebA(args)
    print("\r\nImage Flows From: ", args.img_path, "   Image Count: ", data.batch_size * data.batches)
    print("\r\nUsing Attribute: ", data.label)
    repo = Repo(".")
    if repo.is_dirty() and args.test == 0:
        raise EnvironmentError("Git repo is Dirty! Please train after committed.")
    model.fit(data, args)
