import copy
from typing import Any

from transformers import PretrainedConfig


class ConchConfig(PretrainedConfig):
    model_type = "conch"

    def __init__(
        self,
        patch_size: int = 16,
        context_dim: int = 1024,
        embed_dim: int = 768,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values: float = 1e-6,
        pooler_n_queries_contrast: int = 1,
        **kwargs: Any,
    ):
        self.patch_size = patch_size
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.init_values = init_values
        self.pooler_n_queries_contrast = pooler_n_queries_contrast

        super().__init__(**kwargs)


class TitanVisionConfig(PretrainedConfig):
    model_type = "titan_vision"

    def __init__(
        self,
        grid_size: int = 14,
        global_pool: str = "token",
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        mlp_patch_embed_dim: int = 768,
        pos_encode_type: str = "alibi",
        #### CoCa params ####
        attentional_pool: str = None,
        attn_pooler_queries: int = 128,
        attn_pooler_heads: int = 8,
        **kwargs: Any,
    ):
        self.grid_size = grid_size
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.mlp_patch_embed_dim = mlp_patch_embed_dim
        self.pos_encode_type = pos_encode_type
        self.attentional_pool = attentional_pool
        self.attn_pooler_queries = attn_pooler_queries
        self.attn_pooler_heads = attn_pooler_heads

        super().__init__(**kwargs)


class TitanTextConfig(PretrainedConfig):
    model_type = "titan_text"

    def __init__(
        self,
        embed_dim: int = 768,
        context_length: int = 128,
        vocab_size: int = 32007,
        width: int = 768,
        heads: int = 12,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        embed_cls: bool = True,
        pad_id: int = 0,
        pool_type: str = "argmax",
        proj_bias: bool = False,
        output_tokens: bool = True,
        **kwargs: Any,
    ):
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.vocab_size = vocab_size

        self.width = width
        self.heads = heads
        self.layers = layers
        self.mlp_ratio = mlp_ratio
        self.embed_cls = embed_cls
        self.pad_id = pad_id
        self.pool_type = pool_type
        self.proj_bias = proj_bias
        self.output_tokens = output_tokens

        super().__init__(**kwargs)


class TitanConfig(PretrainedConfig):
    model_type = "titan"
    is_composition = True

    def __init__(
        self,
        vision_config: TitanVisionConfig = TitanVisionConfig(),
        text_config: TitanTextConfig = TitanTextConfig(),
        conch_config: ConchConfig = ConchConfig(),
        **kwargs: Any,
    ):
        if isinstance(conch_config, dict):
            self.conch_config = ConchConfig(**conch_config)
        else:
            self.conch_config = conch_config
        if isinstance(vision_config, dict):
            self.vision_config = TitanVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config
        ### for CoCa ###
        self.vision_config.attentional_pool = "parallel"
        if isinstance(text_config, dict):
            self.text_config = TitanTextConfig(**text_config)
        else:
            self.text_config = text_config

        super().__init__(**kwargs)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        for k, v in output.items():
            if isinstance(v, PretrainedConfig):
                output[k] = v.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
