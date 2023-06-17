from lightning.pytorch.cli import LightningCLI

from eegformer.dataset import PhysionetMotorImagery
from eegformer.model import Transformer


def main():
    cli = LightningCLI(model_class=Transformer, datamodule_class=PhysionetMotorImagery, save_config_callback=False)


if __name__ == "__main__":
    main()
