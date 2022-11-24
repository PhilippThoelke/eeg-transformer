from functools import partial
import torch


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