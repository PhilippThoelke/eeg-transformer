import warnings
from os.path import join, basename
import glob
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from eegt import utils
from eegt.dataset import RawDataset
from eegt.utils import Attention, rollout, get_dataloader


def main(model_dir, data_path, label_path):
    # load model checkpoint
    model_path = glob.glob(join(model_dir, "checkpoints", "*"))
    if len(model_path) > 1:
        warnings.warn(f"Found multiple model checkpoints, choosing {model_path[0]}.")
    model = utils.load_lightning_module(model_path[0])
    model.eval().freeze()

    if data_path is None:
        data_path = model.hparams.data_path
    if label_path is None:
        label_path = model.hparams.label_path

    # load dataset
    data = RawDataset(model.hparams, data_path=data_path, label_path=label_path)
    val_idx = None
    if basename(model.hparams.data_path) == basename(data_path):
        val_idx = torch.load(join(model_dir, "splits.pt"))["val_idx"]
    dl = get_dataloader(model.hparams, data, indices=val_idx, weighted_sampler=False)

    # iterate over the dataset
    acc = 0
    attn, predictions, labels, subjects, datasets, confidences = [], [], [], [], [], []
    prog = tqdm(dl, desc="extracting attention weights")
    for i, (x, ch_pos, mask, condition, subject, dataset) in enumerate(prog):
        # extract attention weights
        with Attention(model) as a:
            pred = model(x, ch_pos, mask)

        # save batchwise metrics
        attn.append(rollout(a.get()))
        labels.append(condition)
        predictions.append(pred.argmax(dim=-1))
        confidences.append(pred.max(dim=-1).values)
        subjects.append(subject)
        datasets.append(dataset)

        # compute batchwise accuracy
        acc += (pred.argmax(dim=-1) == condition).float().mean().item()
        prog.set_postfix(dict(val_acc=acc / (i + 1)))
    acc /= len(dl)
    print("validtion accuracy:", acc)

    attn = torch.cat(attn)
    labels = torch.cat(labels)
    predictions = torch.cat(predictions)
    confidences = torch.cat(confidences)
    subjects = torch.cat(subjects)
    datasets = torch.cat(datasets)

    torch.save(
        (
            attn,
            confidences,
            predictions,
            labels,
            subjects,
            datasets,
            model.hparams,
            data.condition_mapping,
            data.subject_mapping,
            data.dataset_mapping,
        ),
        join(model_dir, "attention.pt"),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="path to the model checkpoint's log directory (can be glob string)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="path to the memory mapped dataset file",
    )
    parser.add_argument(
        "--label-path",
        type=str,
        help="path to the label CSV file",
    )
    args = parser.parse_args()

    if "*" in args.model_dir:
        model_dirs = glob.glob(args.model_dir)
        print(f"found {len(model_dirs)} matching models")
        for model_dir in model_dirs:
            print(f"\nloading {model_dir}")
            main(model_dir, args.data_path, args.label_path)
    else:
        main(args.model_dir, args.data_path, args.label_path)
