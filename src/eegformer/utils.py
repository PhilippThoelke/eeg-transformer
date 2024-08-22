import numpy as np
import torch


def mask_channels(*args, rate: float, dim: int = 1, return_mask=False):
    assert all(
        args[0].size(dim) == arg.size(dim) for arg in args
    ), f"Expected all arguments to to have the same size at dim {dim}"

    mask = torch.rand(args[0].size(dim)) >= rate
    result = tuple(arg.transpose(0, dim)[mask].transpose(0, dim) for arg in args)
    return (result, mask) if return_mask else result


def plot_gradients(module):
    from matplotlib import pyplot as plt

    layers, ave_grads, max_grads, ave_weights, max_weights = [], [], [], [], []
    for n, p in module.named_parameters():
        if (p.requires_grad) and ("bias" not in n) and ("norm" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
            ave_weights.append(p.detach().abs().mean().item())
            max_weights.append(p.detach().abs().max().item())
    ave_grads, max_grads = np.array(ave_grads), np.array(max_grads)
    ave_grads[np.isinf(ave_grads) | np.isnan(ave_grads)] = -1
    max_grads[np.isinf(max_grads) | np.isnan(max_grads)] = -1
    ave_weights, max_weights = np.array(ave_weights), np.array(max_weights)
    ave_weights[np.isinf(ave_weights) | np.isnan(ave_weights)] = -1
    max_weights[np.isinf(max_weights) | np.isnan(max_weights)] = -1

    plt.subplot(211)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, label="max-gradient")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, label="mean-gradient")
    plt.ylim(bottom=-0.01, top=np.percentile(max_grads, 95))
    plt.ylabel("average gradient")
    plt.title("Gradient magnitudes")
    plt.legend()
    plt.xticks([])
    plt.gca().xaxis.set_tick_params(labelbottom=False)

    plt.subplot(212, sharex=plt.gca())
    plt.bar(np.arange(len(max_weights)), max_weights, alpha=0.5, label="max-weight")
    plt.bar(np.arange(len(max_weights)), ave_weights, alpha=0.5, label="mean-weight")
    plt.xticks(range(0, len(ave_weights), 1), layers, rotation="vertical")
    plt.ylim(bottom=-0.01, top=np.percentile(max_weights, 95))
    plt.ylabel("average weight")
    plt.title("Weight magnitudes")
    plt.legend()
    plt.show()
