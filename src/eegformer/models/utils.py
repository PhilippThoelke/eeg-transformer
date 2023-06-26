from typing import Optional, Tuple

import torch
import torch.nn as nn


class MLP3DPositionalEmbedding(nn.Module):
    """
    MLP positional embedding for 3D data.

    ### Args
        - `dim_model` (int): The dimensionality of the model.
        - `add_class_token` (bool): Whether to add a class token.
        - `dropout` (float): Dropout rate.
    """

    def __init__(self, dim_model: int, add_class_token: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(3, dim_model * 2),
            nn.Tanh(),
            nn.Linear(dim_model * 2, dim_model),
            nn.Tanh(),
        )
        self.class_token = torch.nn.Parameter(torch.zeros(dim_model)) if add_class_token else None

    def forward(
        self,
        x: torch.Tensor,
        ch_pos: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embed the channel positions and optionally prepend a class token to the channel dimension.

        ### Args
            - `x` (Tensor): tensor of raw EEG signals (batch, channels, time)
            - `ch_pos` (Tensor): tensor of channel positions (batch, channels, 3)
            - `mask` (Tensor): optional attention mask (batch, channels)

        ### Returns
            Tuple[Tensor, Tensor]: embedded signal (batch, channels, time) and updated mask (batch, channels)
        """
        # embed the channel positions
        if mask is None:
            out = x + self.mlp(ch_pos)
        else:
            out = x
            out[mask] = x[mask] + self.mlp(ch_pos[mask])

        # apply dropout
        out = self.dropout(out)

        # prepend class token
        if self.class_token is not None:
            clf_token = torch.ones(out.shape[0], 1, out.shape[-1], device=out.device) * self.class_token
            out = torch.cat([clf_token, out], dim=1)

            if mask is not None:
                # include class token in mask
                mask = torch.cat([torch.ones(mask.size(0), 1, device=mask.device, dtype=torch.bool), mask], dim=1)
        return out, mask
