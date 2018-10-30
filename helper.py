import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str, required=False, help="is train", default="tb")
parser.add_argument("-n", "--name", type=str, required=False, help="training name", default="default")
args = parser.parse_args()
if args.mode is "tb":
    os.system("start tensorboard --host 0.0.0.0 --logdir " + os.path.abspath("../result/" + args.name + "/events"))
    print("The result path is " + os.path.abspath("../result/" + args.name))
