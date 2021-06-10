import math

import pytorch_lightning as pl
import torch
from einops import rearrange, repeat

TEMPERATURE = 10000.0


class WordPosEnc(pl.LightningModule):
    def __init__(self, d_model: int = 512, max_len=500, temperature=10000.0) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float)
        dim_t = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = 1.0 / (temperature ** (dim_t / d_model))

        inv_freq = torch.einsum("i, j -> i j", position, div_term)

        pe[:, 0::2] = inv_freq.sin()
        pe[:, 1::2] = inv_freq.cos()
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """add positional encoding to feature

        Parameters
        ----------
        x : torch.Tensor
            [b, l, d]

        Returns
        -------
        torch.Tensor
            [b, l, d]
        """
        _, seq_len, _ = x.size()
        emb = self.pe[:seq_len, :]
        x = x + emb[None, :, :]
        return x


class ImgPosEnc(pl.LightningModule):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, d_model=512, temperature=10000, normalize=False, scale=None):
        super().__init__()
        assert d_model % 2 == 0
        self.half_d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """add image positional encoding to feature

        Parameters
        ----------
        x : torch.Tensor
            [b, h, w, d]
        mask: torch.LongTensor
            [b, h, w]

        Returns
        -------
        torch.Tensor
            [b, h, w, d]
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            0, self.half_d_model, 2, dtype=torch.float, device=self.device
        )
        inv_feq = 1.0 / (self.temperature ** (dim_t / self.half_d_model))

        pos_x = torch.einsum("b h w, d -> b h w d", x_embed, inv_feq)
        pos_y = torch.einsum("b h w, d -> b h w d", y_embed, inv_feq)

        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=4).flatten(3)
        pos = torch.cat((pos_x, pos_y), dim=3)

        x = x + pos
        return x


def rotate_every_two(x: torch.FloatTensor):
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")


class WordRotaryEmbed(pl.LightningModule):
    """
    Rotary Positional Embedding
    Ref : https://zhuanlan.zhihu.com/p/359502624
        : https://blog.eleuther.ai/rotary-embeddings/
        : https://arxiv.org/abs/2104.09864

    lucidrains implementation: https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/rotary.py
    """

    def __init__(self, d_model: int = 512) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.FloatTensor):
        """apply positional encoding to feature

        Parameters
        ----------
        x : torch.Tensor
            [b, l, d]

        Returns
        -------
        torch.Tensor
            [b, l, d]
        """
        _, n, _ = x.size()
        t = torch.arange(n, device=self.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i, j -> i j", t, self.inv_freq)
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()

        sin, cos = map(lambda t: repeat(t, "b n -> b (n j)", j=2), (sin, cos))

        x = (x * cos) + (rotate_every_two(x) * sin)
        return x


class ImageRotaryEmbed(pl.LightningModule):
    """
    2-D Generalized version of WordRotaryEmbedding
    """

    def __init__(self, d_model=512, temperature=10000, normalize=False, scale=None):
        super().__init__()
        assert d_model % 2 == 0
        self.half_d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """apply image positional encoding to feature

        Parameters
        ----------
        x : torch.Tensor
            [b, h, w, d]
        mask: torch.LongTensor
            [b, h, w]

        Returns
        -------
        torch.Tensor
            [b, h, w, d]
        """
        not_mask = ~mask
        embed_y = not_mask.cumsum(1, dtype=torch.float32)
        embed_x = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            embed_y = embed_y / (embed_y[:, -1:, :] + eps) * self.scale
            embed_x = embed_x / (embed_x[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            0, self.half_d_model, 2, dtype=torch.float, device=self.device
        )
        inv_feq = 1.0 / (self.temperature ** (dim_t / self.half_d_model))

        # [b, h, w, d_model // 4]
        pos_x = torch.einsum("b h w, d -> b h w d", embed_x, inv_feq)
        pos_y = torch.einsum("b h w, d -> b h w d", embed_y, inv_feq)

        # [b, h, w, d_model // 2]
        sin_x, cos_x, sin_y, cos_y = map(
            lambda t: repeat(t, "b h w d -> b h w (d n)", n=2),
            (pos_x.sin(), pos_x.cos(), pos_y.sin(), pos_y.cos()),
        )

        # [b, h, w, d_model]
        sin, cos = map(lambda t: torch.cat(t, dim=-1), (sin_x, sin_y), (cos_x, cos_y))

        x = (x * cos) + (rotate_every_two(x) * sin)
        return x
