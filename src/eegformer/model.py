from typing import Tuple, Union

import torch
from mamba_ssm import Mamba2
from torch import nn
from torch.nn.functional import scaled_dot_product_attention


class RBFNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_rbf=64):
        super().__init__()
        self.out_channels = out_channels
        self.centers = nn.Parameter(torch.randn(num_rbf, in_channels))
        self.betas = nn.Parameter(torch.ones(num_rbf))
        self.linear = nn.Linear(num_rbf, out_channels)

    def forward(self, x):
        d = torch.cdist(x, self.centers)
        rbf = torch.exp(-self.betas * d**2)
        return self.linear(rbf)


class Temporal(nn.Module):
    """
    Residual temporal mamba block.

    ### Args
        - `hidden_channels`: Number of hidden channels in the input.
        - `headdim`: Dimension of the attention heads.
        - `nheads`: Number of attention heads.
    """

    def __init__(self, hidden_channels, headdim, nheads):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_channels)
        self.layer = Mamba2(
            hidden_channels,
            headdim=headdim,
            expand=8,  # correct would be (nheads * headdim) // hidden_channels but this raises an error
            rmsnorm=False,
        )

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return x + self.layer(self.norm(x))


class Spatial(nn.Module):
    """
    Residual spatial attention block.

    ### Args
        - `hidden_channels`: Number of hidden channels in the input.
        - `headdim`: Dimension of the attention heads.
        - `nheads`: Number of attention heads.
    """

    def __init__(self, hidden_channels, headdim, nheads):
        super().__init__()
        self.headdim = headdim
        self.nheads = nheads

        self.norm = nn.LayerNorm(hidden_channels)

        self.q_bias = nn.Parameter(torch.zeros(nheads, headdim))
        self.qkv_proj = nn.Linear(hidden_channels, headdim * nheads * 3, bias=False)
        self.o_proj = nn.Linear(headdim * nheads, hidden_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        qkv = self.qkv_proj(self.norm(x))  # (..., L, 3 * nheads * headdim)
        qkv = qkv.reshape(*shape[:-1], self.nheads, -1)  # (..., L, nheads, 3 * headdim)
        qkv[..., : self.headdim] = qkv[..., : self.headdim] + self.q_bias
        q, k, v = qkv.transpose(-3, -2).chunk(3, -1)  # 3 * (..., nheads, L, headdim)

        attn = scaled_dot_product_attention(q, k, v)  # (..., nheads, L, headdim)
        attn = attn.transpose(-3, -2)  # (..., L, nheads, headdim)
        attn = attn.reshape(*shape[:-1], -1)  # (..., L, nheads * headdim)

        return x + self.o_proj(attn)


class MLP(nn.Module):
    """
    Residual MLP block.

    ### Args
        - `hidden_channels`: Number of hidden channels in the input.
        - `mlp_expansion`: Expansion factor of the MLP block.
    """

    def __init__(self, hidden_channels, mlp_expansion):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_channels)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * mlp_expansion),
            nn.SiLU(),
            nn.Linear(hidden_channels * mlp_expansion, hidden_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class STLayer(nn.Module):
    """
    Spatio-temporal block.

    ### Args
        - `hidden_channels`: Number of hidden channels in the input.
        - `temporal_headdim`: Dimension of the attention heads in the temporal block.
        - `temporal_nheads`: Number of attention heads in the temporal block.
        - `spatial_headdim`: Dimension of the attention heads in the spatial block.
        - `spatial_nheads`: Number of attention heads in the spatial block.
        - `mlp_expansion`: Expansion factor of the MLP block.
    """

    def __init__(
        self,
        hidden_channels: int,
        temporal_headdim: int,
        temporal_nheads: int,
        spatial_headdim: int,
        spatial_nheads: int,
        mlp_expansion: int,
    ):
        super().__init__()

        # temporal block
        self.temporal = Temporal(hidden_channels, temporal_headdim, temporal_nheads)

        # spatial block
        self.spatial = Spatial(hidden_channels, spatial_headdim, spatial_nheads)

        # MLP block
        self.mlp = MLP(hidden_channels, mlp_expansion)

    def forward(self, x):
        b, s, t, d = x.shape
        x = self.temporal(x.reshape(b * s, t, d)).reshape(b, s, t, d)
        x = self.spatial(x.transpose(1, 2)).transpose(1, 2)
        x = self.mlp(x)
        return x


class Aggregator(nn.Module):
    """
    The Aggregator module transfers information from N small channels into a single large embedding.

    ### Args
        - `hidden_channels`: Number of hidden channels in the input.
        - `embedding_dim`: Dimension of the embedding vector.
        - `headdim`: Dimension of the attention heads.
        - `nheads`: Number of attention heads.
    """

    def __init__(self, hidden_channels, embedding_dim, headdim, nheads):
        super().__init__()
        self.headdim = headdim
        self.nheads = nheads

        self.q_norm = nn.LayerNorm(embedding_dim)
        self.kv_norm = nn.LayerNorm(hidden_channels)

        self.q_proj = nn.Linear(embedding_dim, headdim * nheads)
        self.kv_proj = nn.Linear(hidden_channels, headdim * 2, bias=False)
        self.o_proj = nn.Linear(headdim * nheads, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        ### Args
            - `x`: Source input channels `(..., N, hidden_channels)`.
            - `z`: Target embedding vector `(..., embedding_dim)`.
        ### Returns
            - Output tensor `(..., embedding_dim)`.
        """
        q = self.q_proj(self.q_norm(z))  # (..., nheads * headdim)
        q = q.reshape(*q.shape[:-1], self.nheads, -1)  # (..., nheads, headdim)

        kv = self.kv_proj(self.kv_norm(x))  # (..., N, headdim * 2)
        k, v = kv.chunk(2, -1)  # 2 * (..., N, headdim)

        attn = scaled_dot_product_attention(q, k, v)  # (..., nheads, headdim)
        attn = attn.reshape(*attn.shape[:-2], -1)  # (..., nheads * headdim)
        return z + self.o_proj(attn)  # (..., embedding_dim)


class Distributor(nn.Module):
    """
    The Distributor module transfers information from a large embedding into N small channels.

    ### Args
        - `hidden_channels`: Number of hidden channels in the input.
        - `embedding_dim`: Dimension of the embedding vector.
        - `headdim`: Dimension of the attention heads.
        - `nheads`: Number of attention heads.
    """

    def __init__(self, hidden_channels, embedding_dim, headdim, nheads):
        super().__init__()
        self.headdim = headdim
        self.nheads = nheads

        self.q_norm = nn.LayerNorm(hidden_channels)
        self.kv_norm = nn.LayerNorm(embedding_dim)

        self.q_proj = nn.Linear(hidden_channels, headdim)
        self.kv_proj = nn.Linear(embedding_dim, headdim * nheads * 2, bias=False)
        self.o_proj = nn.Linear(headdim, hidden_channels, bias=False)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        ### Args
            - `x`: Target output channels `(..., N, hidden_channels)`.
            - `z`: Source embedding vector `(..., embedding_dim)`.
        ### Returns
            - Output tensor `(B, N, hidden_channels)`.
        """
        q = self.q_proj(self.q_norm(x))  # (..., N, headdim)

        kv = self.kv_proj(self.kv_norm(z))  # (..., nheads * headdim * 2)
        kv = kv.reshape(*kv.shape[:-1], self.nheads, -1)  # (..., nheads, 2 * headdim)
        k, v = kv.chunk(2, -1)  # 2 * (..., nheads, headdim)

        attn = scaled_dot_product_attention(q, k, v)  # (..., N, headdim)
        return x + self.o_proj(attn)  # (..., N, hidden_channels)


class Encoder(nn.Module):
    """
    Spatio-temporal encoder network.

    Encodes an N-dimensional time series into a sequence of fixed-size embedding vectors
    by interleaving temporal (Mamba), spatial (Multi-Head Attention) and aggregation
    (Multi-Head Attention) blocks.

    The time series is split into small fixed-size epochs, which are used time steps in the
    final embedding sequence.

    ### Args
        - `epoch_size`: Number of samples in each epoch (input and output dimension).
        - `embedding_dim`: Dimension of the aggregated embedding vectors.
        - `hidden_channels`: Number of hidden channels in the network.
        - `num_layers`: Number of spatio-temporal blocks.
        - `temporal_headdim`: Dimension of the attention heads in the temporal block.
        - `temporal_nheads`: Number of attention heads in the temporal block.
        - `spatial_headdim`: Dimension of the attention heads in the spatial block.
        - `spatial_nheads`: Number of attention heads in the spatial block.
        - `aggregation_headdim`: Dimension of the attention heads in the aggregation module.
        - `aggregation_nheads`: Number of attention heads in the aggregation module.
        - `mlp_expansion`: Expansion factor of the MLP block.
        - `num_spatial_dims`: Number of spatial dimensions in the input data.
    """

    def __init__(
        self,
        epoch_size: int,
        embedding_dim: int = 512,
        hidden_channels: int = 32,
        num_layers: int = 4,
        temporal_headdim: int = 8,
        temporal_nheads: int = 4,
        spatial_headdim: int = 8,
        spatial_nheads: int = 4,
        aggregation_headdim: int = 16,
        aggregation_nheads: int = 4,
        mlp_expansion: int = 2,
        num_spatial_dims: int = 3,
    ):
        super().__init__()
        self.epoch_size = epoch_size
        self.embedding_dim = embedding_dim

        # N-dimensional spatial encoding block
        self.spatial_encoding = RBFNet(num_spatial_dims, num_spatial_dims * 4)

        # projection layer for input and spatial encoding
        self.in_proj = nn.Linear(epoch_size + self.spatial_encoding.out_channels, hidden_channels, bias=False)
        self.embedding_init = nn.Parameter(torch.randn(embedding_dim))

        # spatio-temporal update blocks
        self.update_blocks = nn.ModuleList(
            [
                STLayer(
                    hidden_channels,
                    temporal_headdim,
                    temporal_nheads,
                    spatial_headdim,
                    spatial_nheads,
                    mlp_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        # spatial aggregation blocks
        self.aggregation_blocks = nn.ModuleList(
            [
                Aggregator(
                    hidden_channels,
                    embedding_dim,
                    aggregation_headdim,
                    aggregation_nheads,
                )
                for _ in range(num_layers)
            ]
        )

        # variational layer
        self.variational = nn.Linear(embedding_dim, embedding_dim * 2, bias=False)

    def forward(
        self, x: torch.Tensor, p: torch.Tensor, reparametrize: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        ### Args
            - `x`: Spatio-temporal input data `(B, spatial_chans, time)` or `(B, spatial_chans, temporal_chans, epoch_size)`.
            - `p`: Spatial positions `(spatial_chans, spatial_dims)` or `(B, spatial_chans, spatial_dims)`.
            - `reparametrize`: Whether to reparametrize the latent space.
        ### Returns
            - Latent vector `(B, embedding_dim)` if `reparametrize` is `True`, otherwise a tuple of mean and log-variance.
        """
        # split x into epochs if necessary
        x = self.to_epochs(x)

        # compute spatial encoding
        p = self.spatial_encoding(p).unsqueeze(-2)

        if p.ndim == 3:
            # add batch dimension
            p = p.unsqueeze(0).expand(x.size(0), -1, -1, -1)

        # project input and spatial encoding to hidden dimension
        x = torch.cat([x, p.broadcast_to(*x.shape[:-1], -1)], dim=-1)
        x = self.in_proj(x)

        # get initial aggregation vector for the final embedding
        z = self.embedding_init.expand(x.size(0), x.size(2), -1)

        # apply spatio-temporal blocks
        for update, aggregate in zip(self.update_blocks, self.aggregation_blocks):
            x = update(x)
            z = aggregate(x.transpose(1, 2), z)

        mu, logvar = self.variational(z).chunk(2, dim=-1)

        if reparametrize:
            return self.reparametrize(mu, logvar)
        return mu, logvar

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        ### Args
            - `mu`: Mean of the latent space `(B, embedding_dim)`.
            - `logvar`: Log-variance of the latent space `(B, embedding_dim)`.
        ### Returns
            - Reparametrized latent vector `(B, embedding_dim)`.
        """
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def to_epochs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split an N-dimensional time series into epochs according to the model's `epoch_size`.

        ### Args
            - `x`: Spatio-temporal input data `(B, spatial_chans, time)` or `(B, spatial_chans, temporal_chans, epoch_size)`.
        ### Returns
            - Spatio-temporal input data `(B, spatial_chans, temporal_chans, epoch_size)`.
        """
        if x.ndim == 3:
            assert x.size(-1) % self.epoch_size == 0, (
                f"last input dimension ({x.size(-1)}) must be divisible " f"by inout_channels ({self.epoch_size})"
            )
            # split x into epochs
            x = x.reshape(*x.shape[:-1], -1, self.epoch_size)
        return x


class Decoder(nn.Module):
    """
    Spatio-temporal decoder network.

    Decodes a sequence of fixed-size embedding vectors into an N-dimensional time series.

    ### Args
        - `epoch_size`: Number of samples in each epoch (input and output dimension).
        - `spatial_encoding`: Spatial encoding block.
        - `embedding_dim`: Dimension of the aggregated embedding vectors.
        - `hidden_channels`: Number of hidden channels in the network.
        - `num_layers`: Number of spatio-temporal blocks.
        - `distribution_headdim`: Dimension of the attention heads in the distribution block.
        - `distribution_nheads`: Number of attention heads in the distribution block.
        - `mlp_expansion`: Expansion factor of the MLP block.
    """

    def __init__(
        self,
        epoch_size: int,
        spatial_encoding: RBFNet,
        embedding_dim: int,
        hidden_channels: int = 32,
        num_layers: int = 3,
        distribution_headdim: int = 16,
        distribution_nheads: int = 4,
        mlp_expansion: int = 2,
    ):
        super().__init__()
        self.epoch_size = epoch_size
        self.spatial_encoding = spatial_encoding

        # hidden state initializer
        self.initializer = nn.Linear(self.spatial_encoding.out_channels, hidden_channels, bias=True)

        # spatial distribution blocks
        self.distribution_blocks = nn.ModuleList(
            [
                Distributor(
                    hidden_channels,
                    embedding_dim,
                    distribution_headdim,
                    distribution_nheads,
                )
                for _ in range(num_layers)
            ]
        )
        # update blocks
        self.update_blocks = nn.ModuleList([MLP(hidden_channels, mlp_expansion) for _ in range(num_layers)])

        self.out_proj = nn.Linear(hidden_channels, self.epoch_size, bias=False)

    def forward(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        ### Args
            - `z`: Latent vectors `(B, temporal_chans, embedding_dim)`.
            - `p`: Spatial positions `(spatial_chans, spatial_dims)` or `(B, spatial_chans, spatial_dims)`.
        ### Returns
            - Reconstructed spatio-temporal data `(B, spatial_chans, temporal_chans, epoch_size)`.
        """
        # compute spatial encoding
        p = self.spatial_encoding(p)

        if p.ndim == 2:
            # add batch dimension
            p = p.unsqueeze(0).expand(x.size(0), -1, -1)

        # initialize hidden states from spatial encoding
        x = self.initializer(p).unsqueeze(1).expand(-1, z.size(1), -1, -1)

        # apply distribution and update blocks
        for distribute, update in zip(self.distribution_blocks, self.update_blocks):
            x = distribute(x, z)
            x = update(x)

        return self.out_proj(x.transpose(1, 2))

    @staticmethod
    def from_encoder(encoder: Encoder, *args, **kwargs):
        """
        Initialize a decoder from an encoder model.

        The decoder is constructed to match the encoder's epoch size, spatial encoding
        and embedding dimension.

        Use `*args` and `**kwargs` to pass additional arguments to the decoder.

        ### Args
            - `encoder`: Encoder model.
            - `*args`: Additional positional arguments for the decoder.
            - `**kwargs`: Additional keyword arguments for the decoder.
        ### Returns
            - Decoder model.
        """
        return Decoder(
            encoder.epoch_size,
            encoder.spatial_encoding,
            encoder.embedding_dim,
            *args,
            **kwargs,
        )
