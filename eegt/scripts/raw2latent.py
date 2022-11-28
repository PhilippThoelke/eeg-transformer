import os
from os.path import join, basename, dirname, exists, expanduser
from eegt.modules.simclr import LightningModule
from dataset import RawDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


batch_size = 256
device = "cuda"
ckpt_path = "~/scratch/eeg-transformer/lightning_logs/version_50915880/checkpoints/epoch=16-step=23562.ckpt"

if "SLURM_TMPDIR" in os.environ:
    print(f"trying to load data from {os.environ['SLURM_TMPDIR']}")
    data_path = os.environ["SLURM_TMPDIR"]
else:
    data_path = "~/scratch/eeg-transformer/data"
splits_path = expanduser(join(dirname(dirname(ckpt_path)), "splits.pt"))

print("loading model checkpoint")
module = TransformerModule.load_from_checkpoint(ckpt_path, map_location="cpu")
module.eval().freeze()
model = module.model.to(device)

hparams = module.hparams
hparams.data_path = join(data_path, basename(hparams.data_path))
hparams.label_path = join(data_path, basename(hparams.label_path))

print("loading dataset")
data = RawDataset(hparams)
dl = DataLoader(
    data, batch_size=batch_size, collate_fn=RawDataset.collate, num_workers=4
)

latent, condition, dataset = [], [], []
for batch in tqdm(dl, desc="iterating dataset"):
    x, ch_pos, mask, cond, _, dset = batch

    x = x.permute(2, 0, 1).to(device)
    ch_pos = ch_pos.permute(1, 0, 2).to(device)
    mask = mask.to(device)
    latent.append(model(x, ch_pos, mask).to("cpu"))
    condition.append(cond)
    dataset.append(dset)

print("saving latent data")
latent = torch.cat(latent)
condition = torch.cat(condition)
dataset = torch.cat(dataset)
splits = None
if exists(splits_path):
    splits = torch.load(splits_path)
else:
    print(f"couldn't find splits file at {splits_path}")
torch.save(
    dict(
        latent=latent,
        condition=condition,
        condition2name=data.condition_mapping,
        dataset=dataset,
        dataset2name=data.dataset_mapping,
        splits=splits,
    ),
    "latent.pt",
)
