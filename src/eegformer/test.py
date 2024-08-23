import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import FastICA

from eegformer.data import DataModule
from eegformer.module import LightningModule
from eegformer.utils import mask_channels


def load(ckpt_path, bs=1, device="cuda"):
    model = LightningModule.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()

    data = DataModule(chunk_secs=5, overlap_secs=0, batch_size=bs)
    data.setup("test")

    return model, data.test_dataloader(), data.ch_names, data.sfreq


@torch.inference_mode()
def show_latent_progression(ckpt_path, bs=4, N=64, device="cuda"):
    model, dl, _, sfreq = load(ckpt_path, bs=bs, device=device)

    for batch in dl:
        x, pos = batch

        epochs = model.encoder.to_epochs(x.to(device))
        z = model.encoder(epochs, pos.to(device), reparametrize=False)[0].cpu().numpy()

        pc = FastICA(n_components=N).fit_transform(z.reshape(-1, z.shape[-1])).reshape(bs, -1, N)
        absmax = max(abs(pc.min()), abs(pc.max()))

        _, axes = plt.subplots(bs, 2)
        for i, (ax1, ax2) in enumerate(axes):
            ax1.imshow(pc[i].T, vmin=-absmax, vmax=absmax, cmap="bwr", aspect="auto")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel(f"Chunk {i}\nLatent components")
            ticks = ax1.get_xticks()
            ticks = ticks[(ticks >= 0) & (ticks <= z.shape[1])]
            ax1.set_xticks(ticks, ticks * model.encoder.epoch_size / sfreq)

            ax2.plot(np.arange(0, x.shape[-1] / sfreq, 1 / sfreq), x[i].T, color="black", linewidth=0.4, alpha=0.15)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("EEG amplitude")
        plt.show()


@torch.inference_mode()
def show_last_epoch(ckpt_path, mask_rate=0.5, device="cuda"):
    model, dl, ch_names, _ = load(ckpt_path, device=device)

    for batch in dl:
        x, pos = batch
        x = x.to(device)
        pos = pos.to(device)

        x = model.encoder.to_epochs(x)
        (x_masked, pos_masked), mask = mask_channels(x, pos, rate=mask_rate, dim=1, return_mask=True)

        mu, logvar = model.encoder(x_masked[:, :, :-1], pos_masked, reparametrize=False)
        x = x.squeeze(0)[:, 1:].cpu().numpy()

        def update(frame, axes):
            z = model.encoder.reparametrize(mu, logvar)
            pred = model.decoder(z, pos).squeeze(0).cpu().numpy()

            artists = []
            for i, ax in enumerate(axes.flat[: x.shape[0]]):
                ax.lines[1].set_ydata(pred[i, -1])
                artists.append(ax.lines[1])

            return tuple(artists)

        fig, axes = plt.subplots(int(np.ceil(x.shape[0] / 6)), 6, sharex=True, sharey=True)
        for i, ax in enumerate(axes.flat):
            if i < x.shape[0]:
                ax.plot(x[i, -1])
                ax.plot(x[i, -1])
                ax.set_title(ch_names[i], color="black" if mask[i] else "red")
            ax.axis("off")
        fig.legend(["True", "Predicted"], loc="upper right")
        fig.subplots_adjust(0, 0, 1, 0.95)

        _ = FuncAnimation(fig, update, fargs=(axes,), blit=True, interval=10, cache_frame_data=False)
        plt.show()


if __name__ == "__main__":
    ckpt_path = "last.ckpt"

    # show_latent_progression(ckpt_path)
    show_last_epoch(ckpt_path)
