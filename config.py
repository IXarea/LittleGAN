import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, required=False, help="batch size", default=100)

parser.add_argument("-m", "--mode", type=str, required=False, help="is train", default="train")
parser.add_argument("-n", "--name", type=str, required=False, help="training name", default="default")
parser.add_argument("-i", "--img_path", type=str, required=False, help="image path split with ','")
parser.add_argument("-a", "--attr_path", type=str, required=False, help="attr file path")

parser.add_argument("--epoch", type=int, required=False, help="epoch times", default=100)
parser.add_argument("--start", type=int, required=False, help="start epoch times", default=1)
parser.add_argument("--plot", type=int, required=False, help="print the network info", default=0)
parser.add_argument("--test", type=int, required=False, help="ignore git repo is dirty", default=0)
parser.add_argument("--noise", type=int, required=False, help="noise dimension", default=100)
parser.add_argument("--img_ext", type=str, required=False, help="image extension (with dot)", default=".jpg")
parser.add_argument("--img_size", type=int, required=False, help="image size", default=128)
parser.add_argument("--img_freq", type=int, required=False, help="image save frequency", default=100)
parser.add_argument("--model_freq_batch", type=int, required=False, help="save model every n batches", default=500)
parser.add_argument("--model_freq_epoch", type=int, required=False, help="model save with special name every n epoch",
                    default=1)

args = parser.parse_args()
