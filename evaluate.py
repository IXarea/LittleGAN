#!/usr/bin/env python3


from argparse import ArgumentParser
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = ArgumentParser()
parser.add_argument("mode")
parser.add_argument("image_path")
parser.add_argument("stats_path")
parser.add_argument("model_path")
parser.add_argument("output_file")
parser.add_argument("--gpu", default="-1")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import glob
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf
import datetime

inception_path = fid.check_or_download_inception(args.model_path)

if args.mode == "pre-calculate":
    print("load images..")
    image_list = glob.glob(os.path.join(args.image_path, '*.jpg'))
    images = np.array([imread(image).astype(np.float32) for image in image_list])
    print("%d images found and loaded" % len(images))

    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)
    print("ok")

    print("calculate FID stats..", end=" ", flush=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
        np.savez_compressed(args.stats_path, mu=mu, sigma=sigma)
    print("finished")
else:
    image_list = glob.glob(os.path.join(args.image_path, '*.jpg'))
    images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])

    f = np.load(args.stats_path)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()

    fid.create_inception_graph(inception_path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)

    fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    print("FID: %s" % fid_value)
    with open(args.output_file, "a") as f:
        print("\n", datetime.datetime.now().isoformat(), fid_value, end="\n ", file=f)
