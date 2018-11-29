import sys
import tensorflow as tf
import numpy as np
from dataset import CelebA
from config import Arg

sys.argv[1] = "train"
sys.argv[2] = "auto"
sys.argv.append("--env=debug")
sys.argv.append("--debug")
tf.enable_eager_execution()
a = Arg()
d = CelebA(a)
i, c = d.iterator.get_next()
n = tf.random_normal([16, 93])
np.savez("test.npz", i=i, c=c, n=n)
