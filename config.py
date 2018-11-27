import os
import json
from argparse import ArgumentParser


class Arg:

    def __init__(self):
        parser = ArgumentParser(prog="LittleGAN", description="The code for paper: LittleGAN")

        parser.add_argument("mode", type=str, help="run mode", default="train", choices=["train", "plot", "visual"])
        parser.add_argument("exp_name", type=str, help="experience name")
        parser.add_argument("-e", "--env", type=str, help="config environment", default="default")
        parser.add_argument("-g", "--gpu", type=str, required=False, help="gpu ids, eg: 0,1,2,3", default="-1")
        parser.add_argument("--debug", help="use debug mode, ignore git repo is dirty", action="store_true")
        args = parser.parse_args()
        self.env_file = args.env + ".config.json"
        with open(self.env_file) as f:
            config = json.load(f)
        for item in config:
            self.__setattr__(item, config[item])
        for item in args.__dict__:
            self.__setattr__(item, getattr(args, item))

        self.cond_dim = len(self.attr)
        self.result_dir = os.path.join(self.all_result_dir, args.exp_name)

        if args.mode != "train":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
        self.gpu = self.gpu.split(",")
        self.gpu = [int(item) for item in self.gpu if item.isnumeric() and int(item) >= 0]

        self.conv_filter = [self.min_filter * 2 ** (4 - x) for x in range(5)]

    def __str__(self):
        return self.__dict__.__str__()
