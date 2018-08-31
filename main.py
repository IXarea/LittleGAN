import os.path

from git import Repo

from config import args
from ourgan import OurGAN
from utils import CelebA

attr = [15, 20, 39, 8, 9, 11, 17, 33, 2, 4, 5, 26]
cond_dim = len(attr)
channels = 3
print("Application Params:", args)
model = OurGAN(args.noise, cond_dim, args.img_size, channels, "./result-" + args.name)
if args.plot == 1:
    model.plot()
g_file = os.path.abspath("./result-" + args.name + "/model/G_weight.h5")
if os.path.isfile(g_file):
    model.generator.load_weights(g_file)
d_file = os.path.abspath("./result-" + args.name + "/model/D_weight.h5")
if os.path.isfile(d_file):
    model.discriminator.load_weights(d_file)

if args.mode == "train":
    if args.attr_path is None or args.img_path is None:
        raise ValueError("params error!")
    repo = Repo(".")
    if repo.is_dirty() and args.test == 0:
        raise EnvironmentError("Repo is Dirty! Please train after committed.")
    img_path = args.img_path.split(",")
    data = CelebA(img_path, args.attr_path, args.batch_size, (args.img_size, args.img_size, channels), args.img_ext,
                  attr)
    print("\r\nImage Flows From:", img_path, "   Image Count:", data.batch_size * data.batches)
    model.fit(args.batch_size, args.epoch, data, args.model_freq_batch, args.model_freq_epoch,
              args.img_freq)
elif args.mode == "predict":
    model.predict(args.batch_size)
