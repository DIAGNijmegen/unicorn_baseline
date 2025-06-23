import torch
from pathlib import Path
from transformers import PreTrainedModel

from .vision_transformer import build_vision_tower
from .conch_v1_5 import build_conch
from .configuration_titan import TitanConfig


class Titan(PreTrainedModel):
    config_class = TitanConfig

    def __init__(self, config: TitanConfig, *model_args, **model_kwargs):
        super().__init__(config)

        self.vision_encoder = build_vision_tower(config.vision_config)
        self.conch_config = config.conch_config

    def return_conch(self, model_dir: Path):
        model, eval_transform = build_conch(self.conch_config, model_dir)
        return model, eval_transform

    def encode_slide_from_patch_features(
        self,
        patch_features: torch.Tensor,
        patch_coords: torch.Tensor,
        patch_size_lv0: int,
    ) -> torch.Tensor:
        """
        encode whole-slide image using patch features
        Args:
            patch_features: torch.Tensor, shape (1, N, C)
            patch_coords: torch.Tensor, shape (1, N, 2)
            patch_size_lv0: int, patch size at level 0 (1024 if slide is 40x, 512 if slide is 20x)
        """
        slide_embedding = self.vision_encoder(
            patch_features, patch_coords, patch_size_lv0, no_proj=True
        )
        return slide_embedding
