import warnings
from os.path import join
import glob
from functools import partial
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from module import TransformerModule
from dataset import RawDataset


class Attention:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.attn = []

    def get(self):
        return torch.stack(self.attn, dim=1)

    def __enter__(self):
        def attention_hook(module, input, output, attn):
            attn.append(output[1])

        for encoder_layer in self.model.encoder.encoder_layers:
            hook = partial(attention_hook, attn=self.attn)
            self.handles.append(encoder_layer.mha.register_forward_hook(hook))
        return self

    def __exit__(self, type, value, traceback):
        for handle in self.handles:
            handle.remove()


def rollout(attn, head_fuse="max", only_class=False):
    assert (
        attn.ndim == 5
    ), "expected attn to have 5 dimensions (batch x layers x heads x tokens x tokens)"
    assert head_fuse in [
        "max",
        "mean",
        "min",
    ], "expected head_fuse to be one of max, mean, min"
    fuse_fns = {"max": torch.max, "mean": torch.mean, "min": torch.min}

    attn = fuse_fns[head_fuse](attn, dim=2)
    if not isinstance(attn, torch.Tensor):
        attn = attn.values

    result = torch.eye(attn.size(-1))
    for layer in attn.permute(1, 0, 2, 3).flip(dims=(0,)):
        result = layer @ result

    if only_class:
        result = result[:, 0, 1:]
        return result / result.max()
    return result


def main(model_dir, data_path, label_path):
    # load model checkpoint
    model_path = glob.glob(join(model_dir, "checkpoints", "*"))
    if len(model_path) > 1:
        warnings.warn(f"Found multiple model checkpoints, choosing {model_path[0]}.")
    model = TransformerModule.load_from_checkpoint(model_path[0])
    model.eval().freeze()

    # load dataset
    data = RawDataset(
        data_path,
        label_path,
        epoch_length=model.hparams.epoch_length,
        nchannels=model.hparams.num_channels,
        low_pass=model.hparams.low_pass,
        high_pass=model.hparams.high_pass,
        notch_freq=model.hparams.notch_freq,
        sample_rate=model.hparams.sample_rate,
        stages=model.hparams.stages,
        conditions=model.hparams.conditions,
    )
    splits = torch.load(join(model_dir, "splits.pt"))["val_idx"]
    data = Subset(data, splits)
    dl = DataLoader(data, batch_size=64, num_workers=4)

    # iterate over the dataset
    acc = 0
    attn, predictions, labels, stages, subjects, confidences = [], [], [], [], [], []
    prog = tqdm(dl, desc="extracting attention weights")
    for i, (x, y, stage, subj) in enumerate(prog):
        # extract attention weights
        with Attention(model) as a:
            pred = model(x)

        # save batchwise metrics
        attn.append(rollout(a.get()))
        labels.append(y)
        predictions.append(pred.argmax(dim=-1))
        confidences.append(pred.max(dim=-1).values)
        stages.append(stage)
        subjects.append(subj)

        # compute batchwise accuracy
        acc += (pred.argmax(dim=-1) == y).float().mean().item()
        prog.set_postfix(dict(val_acc=acc / (i + 1)))
    acc /= len(dl)
    print("validtion accuracy:", acc)

    attn = torch.cat(attn)
    labels = torch.cat(labels)
    predictions = torch.cat(predictions)
    confidences = torch.cat(confidences)
    stages = torch.cat(stages)
    subjects = torch.cat(subjects)

    torch.save(
        (
            attn,
            confidences,
            predictions,
            labels,
            stages,
            subjects,
            model.hparams,
            data.dataset.condition_mapping,
            data.dataset.stage_mapping,
            data.dataset.subject_mapping,
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
        required=True,
        help="path to the memory mapped dataset file",
    )
    parser.add_argument(
        "--label-path",
        type=str,
        required=True,
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
