"""
Based on Perceiver architecture
"""

import warnings
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from environs import Env
from torch import Tensor, nn
from torch.backends.cuda import sdp_kernel

from unicorn_baseline.vision_language.prism.configuring_prism import PerceiverConfig

env = Env()

PERCEIVER_MEM_EFF_ATTN: bool = env.bool("PERCEIVER_MEM_EFF_ATTN", default=False)
if PERCEIVER_MEM_EFF_ATTN:
    warnings.warn("Perceiver: using memory-efficient attention")

try:
    from xformers.ops import memory_efficient_attention  # type: ignore
except ImportError:
    if PERCEIVER_MEM_EFF_ATTN:
        raise Exception(
            "Memory efficient attention flag is set (PERCEIVER_MEM_EFF_ATTN) "
            "but xformers lib is not available."
        )
    pass


# Attention modules


class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: int,
        head_dim: int,
        heads: int,
        return_attn: bool = False,
        c_norm: bool = True,
    ) -> None:
        super().__init__()

        self.query_dim = query_dim
        self.context_dim = context_dim

        self.head_dim = head_dim
        self.heads = heads

        self.scale = self.head_dim**-0.5

        self.inner_dim = self.head_dim * self.heads

        self.return_attn = return_attn

        self.x_norm = nn.LayerNorm(self.query_dim)
        self.c_norm = nn.LayerNorm(self.context_dim) if c_norm is True else None

        self.to_q = nn.Linear(self.query_dim, self.inner_dim, bias=False)

        self.to_kv = nn.Linear(self.context_dim, self.inner_dim * 2, bias=False)

        self.to_out = nn.Linear(self.inner_dim, self.query_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        c: Tensor | None = None,
        kvt: tuple[Tensor, Tensor] | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        """
        Args:
            x: queries
            c: context
            kvt: key-value cache (instead of context)
            attn_mask: mask out part of context since contexts can vary in length

        Returns:
            processed output queries, KV-cache, attention weights
        """
        Bx, Nx, Dimx = x.shape

        x = self.x_norm(x)
        c = self.c_norm(c) if self.c_norm is not None else c

        q: Tensor = self.to_q(x)
        q = q.reshape(Bx, Nx, self.heads, self.head_dim)

        if c is not None and kvt is None:
            Bc, Nc, _ = c.shape
            kv: Tensor = self.to_kv(c)
            kv = kv.reshape(Bc, Nc, 2, self.heads, self.head_dim)
            k, v = kv.unbind(2)
            kvt = (k, v)
        elif kvt is not None and c is None:
            k, v = kvt
            Bc, Nc, _, _ = k.shape
            assert (Bc, Nc) == (v.shape[0], v.shape[1])
        else:
            raise Exception(f"XOR(c, kvt) but got: {type(c)} and {type(kvt)}.")

        if attn_mask is not None:
            attn_mask = attn_mask.reshape(Bc, 1, Nx, Nc).expand(-1, self.heads, -1, -1)

        if self.return_attn:
            warnings.warn("XATTN RETURNS ATTN SCORES, ONLY FOR EVAL!")
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            q = q * self.scale
            sim = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
            attn = sim.softmax(dim=-1)
            a = attn @ v
            a = a.transpose(1, 2)

        elif PERCEIVER_MEM_EFF_ATTN:
            assert q.shape == (Bx, Nx, self.heads, Dimx // self.heads)
            assert k.shape == (Bx, Nc, self.heads, Dimx // self.heads)
            assert v.shape == (Bx, Nc, self.heads, Dimx // self.heads)
            if attn_mask is not None:
                attn_bias = torch.zeros_like(attn_mask, dtype=q.dtype, device=q.device)
                attn_bias = attn_bias.masked_fill(~attn_mask, -torch.finfo(q.dtype).max)
            else:
                attn_bias = None
            a = memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attn_bias,
                p=0.0,
                scale=None,
                op=None,
                output_dtype=None,
            )
            attn = torch.empty(0)

        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            with sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                a: Tensor = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
                attn = torch.empty(0)
            a = a.transpose(1, 2)

        c = a.reshape(Bx, Nx, self.inner_dim)

        o = self.to_out(c)

        return o, kvt, attn


class MHSA(nn.Module):
    def __init__(self, *, dim: int, num_heads: int):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=dim,
            vdim=dim,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)

        with sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=PERCEIVER_MEM_EFF_ATTN,
        ):
            x, _ = self.mha(
                x,
                x,
                x,
                key_padding_mask=None,
                need_weights=False,
                attn_mask=None,
                average_attn_weights=False,
                is_causal=False,
            )

        return x


# Feed forward


class GEGLU(nn.Module):
    def forward(self, x: Tensor):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        mult: int = 1,
        dropout: float = 0.0,
        activation: str = "geglu",
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        extra_dim = 1

        if activation == "geglu":
            actfn = GEGLU
            extra_dim = 2
        elif activation == "gelu":
            actfn = nn.GELU
        else:
            raise Exception(f"{activation=} not supported.")

        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * extra_dim),
            actfn(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):
        return self.net(self.norm(x))


class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        latent_seq: int = 512,
        latent_dim: int = 768,
        context_dim: int = 2560,
        mhsa_heads: int = 8,
        perceiver_depth: int = 8,
        transformer_depth: int = 6,
        share_xattn_start_layer: int = 1,
        share_tf_start_layer: int = 0,
        xattn_heads: int = 1,
        mlp_mult: int = 1,
        mlp_activation: str = "geglu",
    ):
        super().__init__()

        assert perceiver_depth > 0
        assert share_xattn_start_layer >= 0
        assert share_tf_start_layer >= 0

        self.share_xattn_start_layer = share_xattn_start_layer
        self.share_tf_start_layer = share_tf_start_layer
        self.latent_seq = latent_seq
        self.mhsa_heads = mhsa_heads

        latent_weights = torch.randn(latent_seq, latent_dim)
        self.latents = nn.Parameter(latent_weights)

        # inspired by lucidrains Perceiver repo

        get_xattn = lambda: nn.ModuleDict(
            {
                "xattn": CrossAttention(
                    query_dim=latent_dim,
                    context_dim=context_dim,
                    head_dim=latent_dim // xattn_heads,
                    heads=xattn_heads,
                    return_attn=False,
                    # norming large contexts explodes memory
                    c_norm=False,
                ),
                "ff": FeedForward(
                    dim=latent_dim,
                    mult=mlp_mult,
                    activation=mlp_activation,
                ),
            }
        )

        get_mhsa = lambda: nn.ModuleDict(
            {
                "mhsa": MHSA(
                    dim=latent_dim,
                    num_heads=mhsa_heads,
                ),
                "ff": FeedForward(
                    dim=latent_dim,
                    mult=mlp_mult,
                    activation=mlp_activation,
                ),
            }
        )

        get_transformer = lambda: nn.ModuleList(
            [get_mhsa() for _ in range(transformer_depth)]
        )

        layers = []
        for i in range(perceiver_depth):
            layer = nn.ModuleDict(
                {
                    "xattn": (
                        get_xattn()
                        if i <= self.share_xattn_start_layer
                        else layers[-1]["xattn"]
                    ),
                    "tf": (
                        get_transformer()
                        if i <= self.share_tf_start_layer
                        else layers[-1]["tf"]
                    ),
                }
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, context: Tensor, attn_mask: Tensor) -> Tensor:
        """
        Args:
            context: input sequence that Perceiver processes with its latent queries.
                shape: (batch size, sequence length, feature dim)
            attn_mask: mask out padded area for each context example in the batch;
                necessary for batch size > 1 since contexts can have different sequence lengths.
                shape: (batch size, sequence length)

        Returns:
            processed latent queries (image latents)
        """
        B, N, _ = context.shape

        assert len(attn_mask.shape) == 2
        assert attn_mask.shape[:1] == context.shape[:1]

        attn_mask = attn_mask.reshape(B, 1, N).expand(-1, self.latent_seq, -1)

        x = self.latents.unsqueeze(0).expand(B, -1, -1)

        kvt = None
        for i, l in enumerate(self.layers):
            xattn = l["xattn"]  # type: ignore
            if i <= self.share_xattn_start_layer:
                # pass context, return key-value cache
                xattn_out, kvt, _ = xattn["xattn"](
                    x, c=context, kvt=None, attn_mask=attn_mask
                )
            else:
                # pass key-value cache since the previous layer shares W_k and W_v with this one
                xattn_out, kvt, _ = xattn["xattn"](
                    x, c=None, kvt=kvt, attn_mask=attn_mask
                )

            x = xattn_out + x
            x = xattn["ff"](x) + x

            tf = l["tf"]  # type: ignore
            for mhsa in tf:
                x = mhsa["mhsa"](x) + x
                x = mhsa["ff"](x) + x

        return x


class PerceiverResampler(nn.Module):
    def __init__(self, config: PerceiverConfig):
        super().__init__()

        self.query_dim = config.latent_dim
        self.query_seq = config.latent_seq

        latent_seq = 1 + config.latent_seq  # +1 for image embedding latent query

        self.cls_norm = nn.LayerNorm(config.latent_dim)
        self.img_norm = nn.LayerNorm(config.latent_dim)

        self.perceiver = Perceiver(
            latent_seq=latent_seq,
            latent_dim=config.latent_dim,
            context_dim=config.context_dim,
            mhsa_heads=config.mhsa_heads,
            perceiver_depth=config.perceiver_depth,
            transformer_depth=config.transformer_depth,
            share_xattn_start_layer=config.share_xattn_start_layer,
            share_tf_start_layer=config.share_tf_start_layer,
            xattn_heads=config.xattn_heads,
            mlp_mult=config.mlp_mult,
            mlp_activation=config.mlp_activation,
        )

    __call__: Callable[..., dict[str, Tensor]]

    def forward(
        self,
        tile_embeddings: Tensor,
        tile_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            tile_embeddings: sequences of tile embeddings per whole slide image
                shape: (batch size, sequence length, feature dim)
            attn_mask: mask out padded area for each sequence in the batch;
                necessary for batch size > 1 since images can have different sequence lengths.
                shape: (batch size, sequence length)

        Returns:
            dict with image embedding and image latents per sequence
        """
        batch_size = len(tile_embeddings)
        device = tile_embeddings.device

        if tile_mask is None:
            if batch_size > 1:
                raise Exception("tile pad mask must be provided with batch size>1.")

            tile_mask = torch.ones(
                tile_embeddings.shape[:2], device=device, dtype=tile_embeddings.dtype
            )

        x: Tensor = self.perceiver(context=tile_embeddings, attn_mask=tile_mask)

        cls_emb, img_emb = x[:, 0], x[:, 1:]

        assert cls_emb.shape == (len(tile_embeddings), self.query_dim)
        assert img_emb.shape == (len(tile_embeddings), self.query_seq, self.query_dim)

        cls_emb = self.cls_norm(cls_emb)
        img_emb = self.img_norm(img_emb)

        return {"image_embedding": cls_emb, "image_latents": img_emb}
