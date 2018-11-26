import os
import json
from argparse import ArgumentParser


class Arg:

    def __init__(self):
        parser = ArgumentParser(prog="LittleGAN", description="The code for paper: LittleGAN")

        parser.add_argument("mode", type=str, help="run mode", default="train", choices=["train", "plot", "visual"])  #
        parser.add_argument("exp_name", type=str, help="experience name")
        parser.add_argument("-e", "--env", type=str, help="config environment", default="default")
        parser.add_argument("-g", "--gpu", type=str, required=False, help="gpu ids, eg: 0,1,2,3", default="-1")
        parser.add_argument("--debug", help="use debug mode, ignore git repo is dirty", action="store_true")
        args = parser.parse_args()
        self.__setattr__("mode", args.mode)
        self.__setattr__("exp_name", args.exp_name)
        self.__setattr__("env", args.env)
        self.__setattr__("debug", args.debug)
        config = json.load(open(args.env + ".config.json"))
        for item in config:
            self.__setattr__(item, config[item])

        self.cond_dim = len(config["attr"])
        self.result_dir = os.path.join(config['all_result_dir'], args.exp_name)

        if args.mode != "train":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.gpu = args.gpu.split(",")
        self.gpu = [int(item) for item in args.gpu if item.isnumeric() and int(item) >= 0]

        self.conv_filter = [config["min_filter"] * 2 ** (4 - x) for x in range(5)]

    def __str__(self):
        return self.__dict__.__str__()
