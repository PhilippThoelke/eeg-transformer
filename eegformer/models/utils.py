import torch
import torch.nn as nn


class MLP3DPositionalEmbedding(nn.Module):
    """
    MLP positional embedding for 3D data.

    ### Args:
        - `dim_model` (int): The dimensionality of the model.
        - `add_class_token` (bool): Whether to add a class token.
    """

    def __init__(self, dim_model: int, add_class_token: bool = True):
        super().__init__()
        self.dim_model = dim_model
        self.mlp = nn.Sequential(
            nn.Linear(3, dim_model // 2),
            nn.ReLU(),
            nn.Linear(dim_model // 2, dim_model),
        )
        self.class_token = torch.nn.Parameter(torch.zeros(dim_model)) if add_class_token else None

    def forward(self, x: torch.Tensor, ch_pos: torch.Tensor) -> torch.Tensor:
        """
        Embed the channel positions and optionally prepend a class token to the channel dimension.
        """
        # embed the channel positions
        out = x + self.mlp(ch_pos)

        # prepend class token
        if self.class_token is not None:
            clf_token = torch.ones(out.shape[0], 1, out.shape[-1], device=out.device) * self.class_token
            out = torch.cat([clf_token, out], dim=1)
        return out
