import os
from argparse import ArgumentParser

parser = ArgumentParser()
# 操作选择
parser.add_argument("-m", "--mode", type=str, required=False, help="run mode", default="train",
                    choices=["train", "predict", "plot", "manual-predict", "visual", "predict2"])
# 必备参数
parser.add_argument("-b", "--batch_size", type=int, required=False, help="batch size", default=32)
parser.add_argument("-n", "--name", type=str, required=False, help="experience name", default="default")
parser.add_argument("-i", "--img_path", type=str, required=False, help="image path split with ','", default="")
parser.add_argument("-a", "--attr_path", type=str, required=False, help="attr file path")
parser.add_argument("-g", "--gpu", type=str, required=False, help="gpu ids, eg: 0,1,2,3", default="-1")
# 调试参数
parser.add_argument("--test", type=int, required=False, help="ignore git repo is dirty", default=0, choices=[1, 0])
parser.add_argument("--debug", type=int, required=False, help="开启debug模式将会记录详细的loss信息", default=0, choices=[1, 0])

# 其他训练控制参数
parser.add_argument("--epoch", type=int, required=False, help="epoch times", default=100)
parser.add_argument("--img_ext", type=str, required=False, help="image extension (with dot)", default=".jpg", choices=[".jpg", ".png"])
parser.add_argument("--img_freq", type=int, required=False, help="image save frequency", default=100)
parser.add_argument("--model_freq_batch", type=int, required=False, help="save model every n batches", default=500)
parser.add_argument("--model_freq_epoch", type=int, required=False, help="model save with special name every n epoch", default=1)
parser.add_argument("--start", type=int, required=False, help="start epoch times", default=1)
# 模型控制参数
parser.add_argument("--noise", type=int, required=False, help="noise dimension", default=100)
parser.add_argument("--attr", type=str, required=False, help="选择训练的属性序号", default="8,15,20,22,26,36,39")
parser.add_argument("--norm", type=str, required=False, help="选择标准化层", default="instance", choices=["instance", "batch"])
parser.add_argument("--min_filter", type=int, required=False, help="最少的卷积过滤器数量", default=16)
parser.add_argument("--img_size", type=int, required=False, help="image size", default=128)
parser.add_argument("--kernel_size", type=int, required=False, help="卷积核大小", default=5)
parser.add_argument("--img_channel", type=int, required=False, help="模型中图像通道数", default=3)
parser.add_argument("--part", type=int, required=False, help="是否使用训练（若非默认参数按情况调整分组）", default=1, choices=[1, 0])
parser.add_argument("--residual", type=int, required=False, help="instruct if use the residual blocks", default=0, choices=[1, 0])
parser.add_argument("--add_c", type=int, required=False, help="instruct if add condition into G and U-Net", default=2, choices=[0, 2, 4])

args = parser.parse_args()
if args.mode != "train":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.img_path = args.img_path.split(",")
_gpu = args.gpu.split(",")
_attr = args.attr.split(",")
args.gpu = [int(item) for item in _gpu if item.isnumeric() and int(item) >= 0]
args.attr = [int(item) for item in _attr if item.isnumeric() and int(item) >= 0]
