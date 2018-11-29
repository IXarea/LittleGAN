#!/usr/bin/env python3

import os
import glob
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf

data_path = 'E:/ixarea/dataset/128'
stat_path = '../LittleGAN-test/fid_stats_celeba_128_all.npz'
inception_path = '../LittleGAN-test/'
image_path = '../LittleGAN-result'

print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(inception_path)
print("ok")

if not os.path.isfile(stat_path):
    print("load images..", end=" ", flush=True)
    image_list = glob.glob(os.path.join(data_path, '*.jpg'))
    images = np.array([imread(image).astype(np.float32) for image in image_list])
    print("%d images found and loaded" % len(images))

    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)
    print("ok")

    print("calculte FID stats..", end=" ", flush=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
        np.savez_compressed(stat_path, mu=mu, sigma=sigma)
    print("finished")

image_list = glob.glob(os.path.join(image_path, '*.jpg'))
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])

f = np.load(stat_path)
mu_real, sigma_real = f['mu'][:], f['sigma'][:]
f.close()

fid.create_inception_graph(inception_path)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)

fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
print("FID: %s" % fid_value)
