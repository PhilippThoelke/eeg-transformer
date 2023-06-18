import importlib

import torch
from lightning.pytorch.cli import LightningCLI

from eegformer.datasets import *
from eegformer.models import *


def get_num_classes(args) -> int:
    class_name = args.class_path.split(".")[-1]
    datamodule = getattr(importlib.import_module("eegformer.datasets"), class_name)
    return datamodule.num_classes(**args.init_args)


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data", "model.init_args.num_classes", compute_fn=get_num_classes)


def main():
    # optimize use of Tensor cores
    torch.set_float32_matmul_precision("medium")

    cli = CustomLightningCLI(save_config_callback=False)


if __name__ == "__main__":
    main()
