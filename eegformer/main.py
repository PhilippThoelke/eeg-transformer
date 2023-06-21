import importlib

import torch
from lightning.pytorch.cli import LightningCLI

from eegformer.datamodule import DataModule
from eegformer.datasets import *


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data",
            "model.init_args.num_classes",
            compute_fn=lambda dmod: dmod.num_classes,
            apply_on="instantiate",
        )


def main():
    # optimize use of Tensor cores
    torch.set_float32_matmul_precision("medium")

    cli = CustomLightningCLI(datamodule_class=DataModule, save_config_callback=False)


if __name__ == "__main__":
    main()
