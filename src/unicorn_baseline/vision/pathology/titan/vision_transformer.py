"""
Adapted from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
             https://github.com/bytedance/ibot/blob/main/models/vision_transformer.py
             https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
"""

import math
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from timm.layers import DropPath, Mlp, trunc_normal_
from timm.models._manipulate import checkpoint_seq, named_apply
from timm.models.vision_transformer import get_init_weights_vit
from einops import repeat

from .configuration_titan import TitanVisionConfig


def preprocess_features(
    features: torch.Tensor, coords: torch.Tensor, patch_size_lv0: int
):
    # Remove extra dimensions
    features = features.squeeze(0) if features.dim() == 3 else features
    coords = coords.squeeze(0) if coords.dim() == 3 else coords

    # Offset and normalize coordinates
    offset = coords.min(dim=0).values
    grid_coords = torch.floor_divide(coords - offset, patch_size_lv0)

    # Compute grid size
    grid_offset = grid_coords.min(dim=0).values
    grid_coords = grid_coords - grid_offset
    _H, _W = grid_coords.max(dim=0).values + 1

    # Create feature and coordinate grids
    feature_grid = torch.zeros((_H, _W, features.size(-1)), device=features.device)
    coords_grid = torch.zeros((_H, _W, 2), dtype=torch.int64, device=coords.device)

    # Use scatter for more efficient placement
    indices = grid_coords[:, 0] * _W + grid_coords[:, 1]
    feature_grid.view(-1, features.size(-1)).index_add_(0, indices, features)
    coords_grid.view(-1, 2).index_add_(0, indices, coords)

    # Permute grids
    feature_grid = feature_grid.permute(2, 0, 1)
    coords_grid = coords_grid.permute(2, 0, 1)

    # Background mask
    bg_mask = torch.any(feature_grid != 0, dim=0)
    return feature_grid.unsqueeze(0), coords_grid.unsqueeze(0), bg_mask.unsqueeze(0)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        pos_encode: str = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.pos_encode = pos_encode

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop_prob = attn_drop

    def forward(self, x, attn_bias, bg_mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.pos_encode == "alibi":
            if bg_mask is not None and B > 1:
                bg_mask = bg_mask.view(B, -1)
                bg_mask = torch.cat(
                    (
                        torch.ones((B, 1), dtype=bg_mask.dtype, device=bg_mask.device),
                        bg_mask,
                    ),
                    dim=-1,
                )
                attn_mask = bg_mask.unsqueeze(2) * bg_mask.unsqueeze(1)
                # add entries on the diagonal to avoid entire rows being False
                diag_mask = torch.eye(
                    attn_mask.size(1), device=attn_mask.device, dtype=torch.bool
                ).unsqueeze(0)
                attn_mask = torch.logical_or(attn_mask, diag_mask)
                attn_mask = (1 - attn_mask.float()) * torch.finfo(q.dtype).min
                attn_mask = repeat(attn_mask, "b i j -> b h i j", h=self.num_heads)
                attn_mask = attn_mask + attn_bias
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop_prob
                )
            else:
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, dropout_p=self.attn_drop_prob
                )
        elif self.pos_encode in ["none", "abs"]:
            if bg_mask is not None and B > 1:
                bg_mask = bg_mask.view(B, -1)
                bg_mask = torch.cat(
                    (
                        torch.ones((B, 1), dtype=bg_mask.dtype, device=bg_mask.device),
                        bg_mask,
                    ),
                    dim=-1,
                )
                # add entries on the diagonal to avoid entire rows being False
                attn_mask = bg_mask.unsqueeze(2) * bg_mask.unsqueeze(1)
                diag_mask = torch.eye(
                    attn_mask.size(1), device=attn_mask.device, dtype=torch.bool
                ).unsqueeze(0)
                attn_mask = torch.logical_or(attn_mask, diag_mask)
                attn_mask = (1 - attn_mask.float()) * torch.finfo(q.dtype).min
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask.unsqueeze(1),
                    dropout_p=self.attn_drop_prob,
                )
            else:
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.attn_drop_prob
                )
        else:
            raise ValueError(f"pos_encode {self.pos_encode} not supported")

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
        pos_encode=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            pos_encode=pos_encode,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, attn_bias, bg_mask=None):
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias, bg_mask=bg_mask))
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CustomSequential(nn.Module):
    def __init__(self, *modules):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x, attn_mask, bg_mask=None):
        for i, module in enumerate(self.modules_list):
            x = module(x, attn_mask, bg_mask)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        n_head: int = 8,
        n_queries: int = 256,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(
            d_model, n_head, kdim=context_dim, vdim=context_dim
        )
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)
        self.n_head = n_head

    def forward(self, x: torch.Tensor, bg_mask: torch.Tensor = None):
        B = x.shape[0]
        if bg_mask is not None and B > 1:
            bg_mask = bg_mask.view(B, -1)
            bg_mask = torch.cat(
                (
                    torch.ones((B, 1), dtype=bg_mask.dtype, device=bg_mask.device),
                    bg_mask,
                ),
                dim=-1,
            )
            bg_mask = bg_mask.unsqueeze(1).repeat(self.n_head, self.query.shape[0], 1)
            bg_mask = ~bg_mask
        x = self.ln_k(x).permute(1, 0, 2).contiguous()  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        if B == 1:
            out, _ = self.attn(
                q.unsqueeze(1).expand(-1, N, -1), x, x, need_weights=False
            )
        else:
            out = self.attn(
                q.unsqueeze(1).expand(-1, N, -1),
                x,
                x,
                attn_mask=bg_mask,
                need_weights=False,
            )[0]
        return out.permute(1, 0, 2).contiguous()  # LND -> NLD


class VisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        grid_size=14,
        num_classes=0,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        init_values=None,
        class_token=True,
        no_embed_class=False,
        pre_norm=False,
        fc_norm=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
        return_all_tokens=False,
        masked_im_modeling=False,
        mlp_patch_embed_dim=768,
        pos_encode_type="alibi",
        #### CoCa params ####
        attentional_pool: str = None,
        attn_pooler_queries: int = 128,
        attn_pooler_heads: int = 8,
    ):
        """
        Args:
            grid_size (int): input feature grid size
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.return_all_tokens = return_all_tokens  # from ibot
        self.masked_im_modeling = masked_im_modeling  # from ibot

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        ## change embed_layer to mlp
        self.patch_embed = nn.Sequential(
            nn.Linear(mlp_patch_embed_dim, embed_dim), nn.GELU()
        )

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        # instead of embed_len = num_patches, we use grid_size
        self.pos_encode_type = pos_encode_type
        self.num_heads = num_heads
        self.pos_embed = (
            nn.Parameter(
                torch.randn(1, grid_size**2 + self.num_prefix_tokens, embed_dim)
            )
            if pos_encode_type not in ["none", "alibi"]
            else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        if pos_encode_type == "alibi":
            self.local_alibi = self.get_alibi(6, 6)
            self.global_alibi = self.get_alibi(14, 14)
            self.global_alibi_status = False
            self.local_alibi_status = False

        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = CustomSequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    pos_encode=pos_encode_type,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # classifier head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if weight_init != "skip":
            self.init_weights(weight_init)

        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

        if attentional_pool:
            self.attn_pool_type = attentional_pool
            self.pool_type = "none"
            if attentional_pool in ("parallel", "cascade"):
                self.attn_pool = AttentionalPooler(
                    embed_dim,
                    embed_dim,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = AttentionalPooler(
                    embed_dim,
                    embed_dim,
                    n_head=attn_pooler_heads,
                    n_queries=1,
                )
            else:
                raise NotImplementedError
            pool_dim = embed_dim
            scale = embed_dim**-0.5
            self.proj = nn.Parameter(scale * torch.randn(pool_dim, pool_dim))
        else:
            self.attn_pool = None
            self.attn_pool_contrastive = None

    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "moco", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        if self.pos_encode_type not in ["none", "alibi"]:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token")
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def _pos_embed(self, x, coords, w, h):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.interpolate_pos_encoding(x, coords, w, h)
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.interpolate_pos_encoding(x, coords, w, h)
        return self.pos_drop(x)

    def forward_features(self, x, coords=None, mask=None, bg_mask=None):
        B, nc, w, h = x.shape
        ## flatten x before pass into mlp patch embed
        x = x.flatten(2, 3).transpose(1, 2)
        if self.pos_encode_type == "alibi":
            if w * h == 36 and B != 1:
                if not self.local_alibi_status:
                    self.prepare_tensor(x, "local", "alibi")
                attn_bias = self.local_alibi
            elif w * h == 196 and B != 1:
                if not self.global_alibi_status:
                    self.prepare_tensor(x, "global", "alibi")
                attn_bias = self.global_alibi
            else:
                attn_bias = (
                    self.get_alibi(w, h, bg_mask) if B == 1 else self.get_alibi(w, h)
                )
                attn_bias = (
                    attn_bias.repeat(x.shape[0], 1, 1, 1).type(x.dtype).to(x.device)
                )
        else:
            attn_bias = None

        if self.masked_im_modeling:
            assert mask is not None
            x = self.patch_embed(x)
            x = self.mask_model(x, mask)
        else:
            x = self.patch_embed(x)

        x = self._pos_embed(x, coords, w, h)
        x = self.norm_pre(x)

        # mask background tokens when evaluating (batch size = 1)
        if bg_mask is not None and B == 1:
            bg_mask = torch.cat(
                (
                    torch.ones((1, 1), dtype=torch.bool, device=x.device),
                    bg_mask.view(1, -1),
                ),
                dim=1,
            )
            x = x[bg_mask].unsqueeze(0)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, attn_bias, bg_mask)
        else:
            x = self.blocks(x, attn_bias, bg_mask)

        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = (
                x[:, self.num_prefix_tokens :].mean(dim=1)
                if self.global_pool == "avg"
                else x[:, 0]
            )
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward_attn_pool(self, x, bg_mask=None):
        tokens = self.attn_pool(x, bg_mask=bg_mask)
        if self.attn_pool_type == "parallel":
            pooled = self.attn_pool_contrastive(x, bg_mask=bg_mask)
        else:
            assert self.attn_pool_type == "cascade"
            pooled = self.attn_pool_contrastive(tokens, bg_mask=bg_mask)
        pooled = pooled.squeeze(1)
        return pooled, tokens

    def forward(
        self,
        patch_features,
        patch_coords,
        patch_size_lv0,
        return_all_tokens=False,
        mask=None,
        bg_mask=None,
        no_proj=False,
    ):
        x, coords, bg_mask = preprocess_features(
            patch_features, patch_coords, patch_size_lv0
        )
        if self.masked_im_modeling:
            assert mask is not None
            x = self.forward_features(x, coords=coords, mask=mask, bg_mask=bg_mask)
        else:
            x = self.forward_features(x, coords=coords, bg_mask=bg_mask)

        if self.attn_pool is not None:  ### CoCa
            pooled, tokens = self.forward_attn_pool(x, bg_mask=bg_mask)
            if no_proj:
                return pooled
            pooled = pooled @ self.proj
            if self.return_all_tokens:
                return pooled, tokens
            else:
                return pooled
        else:  ### iBOT
            return_all_tokens = (
                self.return_all_tokens
                if return_all_tokens is None
                else return_all_tokens
            )
            if return_all_tokens:
                return x
            x = self.forward_head(x)
            return x

    def interpolate_pos_encoding(self, x, coords, w, h):
        if self.pos_encode_type in ["none", "alibi"]:
            return torch.zeros_like(x)
        elif self.pos_encode_type == "abs":
            # print('Using 2d absolute pos encode')
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            ## MODIFICATIONS: using w // 1 and h // 1 instead of w // patch_size
            w0 = w // 1
            h0 = h // 1
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim
                ).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
            )
            assert (
                int(w0) == patch_pos_embed.shape[-2]
                and int(h0) == patch_pos_embed.shape[-1]
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        else:
            raise NotImplementedError

    def mask_model(self, x, mask):
        mask = mask.view(mask.shape[0], -1)
        x[mask, :] = self.masked_embed.to(x.dtype)
        return x

    def get_alibi(self, w, h, bg_mask=None):
        x, y = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
        if bg_mask is not None:
            x = x[bg_mask.cpu().squeeze(0)]
            y = y[bg_mask.cpu().squeeze(0)]
        points = np.stack([x.ravel(), y.ravel()], axis=1)  # (w*h, 2)
        diffs = points[:, None, :] - points[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))  # (w*h, w*h)

        def get_slopes(n):
            if math.log2(n).is_integer():
                p = 2 ** (-(2 ** -(math.log2(n) - 3)))
                return [p * (p**i) for i in range(n)]
            nearest_power_of_2 = 2 ** math.floor(math.log2(n))
            base_slopes = get_slopes(nearest_power_of_2)
            if nearest_power_of_2 == n:
                return base_slopes
            extra_slopes = get_slopes(2 * nearest_power_of_2)[0::2][
                : n - nearest_power_of_2
            ]
            return base_slopes + extra_slopes

        slopes = torch.tensor(get_slopes(self.num_heads), dtype=torch.float32).view(
            self.num_heads, 1, 1
        )
        n_patches = dists.shape[-1]  # w*h or bg_mask.sum()
        dists_tensor = torch.tensor(dists, dtype=torch.float32).view(
            1, n_patches, n_patches
        )
        bias_matrix = dists_tensor * slopes * -1  # (1, num_heads, w*h, w*h)
        embed_len = n_patches + 1
        all_bias = torch.zeros(1, self.num_heads, embed_len, embed_len)
        all_bias[:, :, 1:, 1:] = bias_matrix
        return all_bias

    def prepare_tensor(self, x, view, type):
        if view == "local":
            if type == "alibi":
                self.local_alibi = (
                    self.local_alibi.repeat(x.shape[0], 1, 1, 1)
                    .type(x.dtype)
                    .to(x.device)
                )
                self.local_alibi_status = True
        elif view == "global":
            if type == "alibi":
                self.global_alibi = (
                    self.global_alibi.repeat(x.shape[0], 1, 1, 1)
                    .type(x.dtype)
                    .to(x.device)
                )
                self.global_alibi_status = True
        else:
            raise NotImplementedError


def build_vision_tower(vision_cfg: TitanVisionConfig):
    vision = VisionTransformer(
        grid_size=vision_cfg.grid_size,
        global_pool=vision_cfg.global_pool,
        embed_dim=vision_cfg.embed_dim,
        depth=vision_cfg.depth,
        num_heads=vision_cfg.num_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        qkv_bias=vision_cfg.qkv_bias,
        mlp_patch_embed_dim=vision_cfg.mlp_patch_embed_dim,
        pos_encode_type=vision_cfg.pos_encode_type,
        attentional_pool=vision_cfg.attentional_pool,
        attn_pooler_queries=vision_cfg.attn_pooler_queries,
        attn_pooler_heads=vision_cfg.attn_pooler_heads,
    )
    return vision
