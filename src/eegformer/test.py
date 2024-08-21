from glob import glob

import numpy as np
import torch

from eegformer.data import load_data, normalize
from eegformer.module import LightningModule


@torch.inference_mode()
def test(ckpt_path):
    from matplotlib import pyplot as plt

    paths = glob(ckpt_path)
    if len(paths) > 1:
        print(f"Multiple checkpoints found. Using {paths[0]}")
    ckpt_path = paths[0]

    subjs_test = range(106, 110)
    x, pos, _, ch_names = load_data(subjs_test)

    # robust normalization of x
    x = normalize(x)

    module = LightningModule.load_from_checkpoint(ckpt_path)
    module.eval()

    x = module.encoder.to_epochs(torch.from_numpy(x).to(module.device))
    pos = torch.from_numpy(pos).to(module.device).expand(len(x), -1, -1)

    x, pos = x[:10], pos[:10]

    # get a random mask
    (x_masked, pos_masked), mask = module.mask(x, pos, rate=0.9, dim=1, return_mask=True)

    # predict masked data
    z = module.encoder(x_masked, pos_masked)
    y = module.decoder(z, pos)

    loss = module.loss(y, x)
    print(f"Test loss: {loss.item()}")

    for i in range(len(x)):
        nchan = x.shape[1]
        for j in range(nchan):
            plt.subplot(int(np.ceil(nchan / 5)), 5, j + 1)
            plt.plot(x[i, j, -1].cpu())  # plot the last epoch
            plt.plot(y[i, j, -2].cpu())  # plot the prediction for the last epoch
            plt.text(
                0,
                0.5,
                ch_names[j],
                ha="right",
                va="center",
                transform=plt.gca().transAxes,
                fontsize=10,
                color="black" if mask[j] else "red",
            )
            plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    test("")
