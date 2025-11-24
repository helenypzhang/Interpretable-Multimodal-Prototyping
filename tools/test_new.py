import argparse
import torch

from medmm.utils import setup_logger, set_random_seed, collect_env_info
from medmm.config import get_cfg_default, clean_cfg
from medmm.engine import build_trainer
from medmm.engine.mbtrain import MBTRAIN

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)



def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed
        cfg.DATASET.FOLD = str(args.seed)

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
        # cfg.TRAINER.PREC = "fp32"

        
        
def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    pass


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the method config file
    if args.config_file:

        if 'umeml' in args.config_file.lower():
            cfg.MODEL.FUSION = None

        cfg.merge_from_file(args.config_file)
    # import pdb;pdb.set_trace()
    # 2. From input arguments
    reset_cfg(cfg, args)

    # 3. From optional input arguments
    cfg.merge_from_list(args.opts)

    clean_cfg(cfg, args.trainer)
    cfg.freeze()

    return cfg

def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model_new_test(args.model_dir)
        trainer.test_new(cfg, split='test')
        return

    if not args.no_train:
        trainer.train(args.umeml_gan_test_without_omic_ratio, args.umeml_gan_test_insample_without_omic_ratio)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--umeml_gan_test_without_omic_ratio", type=float, default=0.1)
    parser.add_argument("--umeml_gan_test_insample_without_omic_ratio", type=float, default=0.1)
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument(
        "--output-dir", type=str, default="", help="output directory"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="only positive value enables a fixed seed"
    )

    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )

    parser.add_argument(
        "--trainer", type=str, default="", help="name of trainer"
    )
    parser.add_argument(
        "--backbone", type=str, default="", help="name of CNN backbone"
    )

    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument(
        "--eval-only", dest='eval_only', action="store_true", help="evaluation only"
    )
    parser.add_argument(
        "--model-dir",
        dest='model_dir',
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch",
        type=int,
        help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)